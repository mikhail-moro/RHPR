import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


class MovieMultimodalTest(unittest.TestCase):
    def test_fixed_normalization_equalizes_constant_predictor_loss_without_missing_bias(self):
        from rhpr.gr.multimodal_rqvae import normalize_modalities
        x={'text':np.array([[1.,2.],[3.,7.],[999.,999.]],dtype='float32'),'people':np.arange(12,dtype='float32').reshape(3,4)*100}
        m={'text':np.array([True,True,False]),'people':np.ones(3,dtype=bool)}
        values,stats=normalize_modalities(x,m)
        for key in x:
            self.assertAlmostEqual(float(np.square(values[key][m[key]]).mean()),1.,places=5)
            recovered=values[key][m[key]]*stats[key]['scale']+stats[key]['mean']
            np.testing.assert_allclose(recovered,x[key][m[key]],atol=1e-4)
        self.assertTrue((values['text'][2]==0).all())
        with self.assertRaises(ValueError):normalize_modalities({'bad':np.ones((3,2))},{'bad':np.ones(3,dtype=bool)})

    def test_exact_four_stage_three_seed_protocol(self):
        from rhpr.gr.movie_suite import validate
        from rhpr.gr.ae_suite import assignments
        c=json.loads((ROOT/'experiments/movielens100k_multimodal_2t4.json').read_text());validate(c)
        self.assertEqual([len(j['trials']) for j in assignments(c,2)],[6,6])
        self.assertEqual(c['fields'][-1],'tmdb_keywords');self.assertEqual(len(c['trials']),12)
        self.assertEqual(c['replicate_seeds'],[42,101,202])
        for key in ['training','generator','data','movie_input']:
            old=json.loads((ROOT/'experiments/movielens100k_text_ladder_2t4.json').read_text());self.assertEqual(c[key],old[key])
        bad=copy.deepcopy(c);bad['trials'][-1]['modalities']=['text','plot']
        with self.assertRaises(ValueError):validate(bad)

    def test_walk_corpus_is_reproducible_and_respects_shared_film_edges(self):
        from scripts.prepare_movie_people_deepwalk import PeopleWalks
        walks=PeopleWalks([[1,2],[2,3,4]],walks=2,length=10)
        first=list(walks);self.assertEqual(first,list(walks));self.assertEqual(len(first),8)
        edges={(1,2),(2,1),(2,3),(3,2),(2,4),(4,2),(3,4),(4,3)}
        for path in first:
            self.assertEqual(len(path),10)
            self.assertTrue(all((int(a),int(b)) in edges for a,b in zip(path,path[1:])))

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires pinned native GRID')
    def test_native_heads_inactive_channels_gradients_and_saved_codes(self):
        script=r'''
import json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.sid_study import state_digest
from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
root=Path(sys.argv[1]);source=prepare_grid_source(sys.argv[2],root/'grid');recipe=Path(sys.argv[2])/'configs/experiment/rqvae_train_flat.yaml';torch.set_num_threads(1)
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=8,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=4)
schema=dict(text=8,poster=6,plot=5,people=4)
torch.manual_seed(42);m,_=build_rqvae(source,recipe,16,spec,modality_schema=schema,active_modalities=['text'])
torch.manual_seed(42);other,_=build_rqvae(source,recipe,16,spec,modality_schema=schema,active_modalities=list(schema))
assert state_digest(dict(m.named_parameters()))==state_digest(dict(other.named_parameters()))
f={k:torch.randn(16,d) for k,d in schema.items()};f.update({k+'_observed':torch.ones(16,dtype=torch.bool) for k in schema})
m.eval();a=m.encode_modalities(f);g={k:v.clone() for k,v in f.items()};g['poster'].fill_(999);g['plot'].fill_(-999);g['people'].fill_(500)
assert torch.equal(a,m.encode_modalities(g)),'inactive information leaked'
loss,heads=m.reconstruction(a,f);assert torch.equal(loss,heads['text']);loss.backward()
assert m.encoder_heads['text'][0].weight.grad.abs().sum()>0
assert m.decoder_heads['text'][0].weight.grad.abs().sum()>0
assert m.encoder_heads['poster'][0].weight.grad is None and m.decoder_heads['poster'][0].weight.grad is None
arrays={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()}
masks={k:np.ones(16,dtype=bool) for k in schema};masks['plot'][0:3]=False
codes,report=fit_multimodal_rqvae(arrays,masks,spec,42,'cpu',source,recipe,root/'fit',list(schema))
assert codes.shape==(16,3) and report['steps']==8
assert set(report['head_reconstruction_mse'])==set(schema)
assert (root/'fit/quantizer.pt').is_file() and (root/'fit/codebook-learning-curves.json').is_file()
assert all(np.isfinite(v) for v in report['head_reconstruction_mse'].values())

from rhpr.gr.movie_data import TemporalUser
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.movie_multimodal import run_multimodal_trial
c=json.loads(Path(sys.argv[3]).read_text());c['quantizer']=spec
c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=8,evaluation_batch_size=2,max_seconds=120)
data=root/'modalities';data.mkdir();np.savez_compressed(data/'features.npz',**arrays,**{k+'_observed':v for k,v in masks.items()})
(data/'source-ids.json').write_text(json.dumps(list(range(16))));(data/'provenance.json').write_text('{}')
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16)
generator=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(generator.named_parameters()));del generator
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=sys.argv[2],initial_state_sha256_by_seed={'42':initial})
r=run_multimodal_trial(dict(id='all',seed=42,stage=4,modalities=list(schema)),c,plan,users,source,'cpu')
assert r['selection']['training_epochs']==20 and r['selection']['validation_checks']==4 and r['test']['evaluation_examples']==2
assert (root/'suite/trials/all/test-predictions.npz').is_file()
'''
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(ROOT/'experiments/movielens100k_multimodal_2t4.json')],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
