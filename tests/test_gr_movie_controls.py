"""Parameter-matched ID control, no graph leakage, and shared temporal trainer."""
import copy,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_names_sasrec_controls_2t4.json'


class MovieControlsTest(unittest.TestCase):
    def test_matched_protocol_and_mixed_initialization_groups(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        old=json.loads((ROOT/'experiments/movielens100k_multimodal_ae_2t4.json').read_text())
        for key in ['training','generator','data','representation','quantizer']:
            self.assertEqual(c[key],old[key])
        self.assertEqual(c['fields'][-2:],['cast','directors'])
        jobs=movie_assignments(c,2);self.assertEqual([len(j['trials']) for j in jobs],[3,3])
        self.assertEqual(sum(t['predictor']=='tiger_names' for t in jobs[0]['trials']),2)
        self.assertTrue(c['compare_initialization_by_shape'])
        for trial in c['trials']:self.assertNotIn('people',trial['modalities'])
        bad=copy.deepcopy(c);bad['trials'][0]['modalities'].append('people')
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['sasrec']['n_layers']=2
        with self.assertRaises(ValueError):validate(bad)
        with self.assertRaises(RuntimeError):movie_assignments(c,1)

    def test_item_adapter_keeps_context_order_causality_and_full_catalogue_loss(self):
        import torch
        from rhpr.gr.movie_sasrec import ItemIDIndex,build_movie_sasrec
        torch.set_num_threads(1)
        c=dict(data=dict(items=8,history_items=4),sasrec=dict(d_model=8,n_heads=2,n_layers=1,ffn_multiplier=2,dropout=0.))
        model=build_movie_sasrec(c,42,'cpu').eval();index=ItemIDIndex(8)
        x=np.array([[-1,-1,0,3],[-1,2,5,7]])
        b=index.model_batch(x,np.array([1,2]),np.array([4,6]),'cpu')
        self.assertEqual(b['item_ids'].tolist(),[[1,4,0,0],[3,6,8,0]])
        self.assertEqual(b['valid_mask'].sum(1).tolist(),[2,3])
        logits=model.catalogue_logits(b);self.assertEqual(tuple(logits.shape),(2,8))
        self.assertTrue(torch.equal(model.rhpr_loss(b),torch.nn.functional.cross_entropy(logits,b['target_item_ids'])))
        # ID-only: user hashes are never consumed, seen items remain candidates.
        changed=index.model_batch(x,np.array([999,1234]),np.array([4,6]),'cpu')
        self.assertTrue(torch.equal(logits,model.catalogue_logits(changed)))
        pred=model.recommend(b).indices;self.assertEqual(pred.shape[1],8)
        encoded=model.encode(b['item_ids'],b['valid_mask'])[0]
        future={k:v.clone() for k,v in b.items()};future['item_ids'][1,2]=1
        torch.testing.assert_close(encoded[1,:2],model.encode(future['item_ids'],future['valid_mask'])[0][1,:2])
        with self.assertRaises(ValueError):index.model_batch(np.array([[1,-1,2,-1]]),np.array([1]),np.array([1]),'cpu')

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires pinned native GRID')
    def test_native_capacity_names_replacement_and_both_twenty_epoch_controls(self):
        script=r'''
import copy,json,sys,torch,numpy as np,hashlib
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.sid_study import state_digest
from rhpr.gr.movie_controls import prepare_names_modalities,capacity_preflight,run_control_trial
from rhpr.gr.movie_data import TemporalUser
from rhpr.gr.rqvae import build_rqvae
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid');torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text())
report=capacity_preflight(c,source,[],root)
assert report['tiger']['active_nonembedding_parameters']==4590464
assert report['sasrec']['active_nonembedding_parameters']==4609536
assert report['relative_parameter_difference']<.01
assert not report['sasrec']['inactive_nonembedding_names'] and not report['tiger']['inactive_nonembedding_names']
assert report['tiger']['supported_forward_flops']>report['sasrec']['supported_forward_flops']>0
# Tiny schema, but the real names replacement, native AE/RQ, and shared trainer.
c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=8,evaluation_batch_size=2,max_seconds=120)
c['data'].update(items=16,history_items=20);c['sasrec'].update(d_model=8,n_heads=2,n_layers=1,ffn_multiplier=2,dropout=0.)
c.pop('expected_sasrec_initial_state_sha256_by_seed');c['benchmark']=dict(batch_sizes=[1,2],warmup=1,iterations=2)
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=8,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=4,ae_warmup=dict(max_steps=8,probe_interval=2,stopping=dict(kind='water_filling_plateau',min_steps=2,patience_probes=2)))
schema=dict(text=8,poster=6,plot=5,people=4);rng=np.random.default_rng(42)
arrays={k:rng.normal(size=(16,d)).astype('float32') for k,d in schema.items()};arrays.update({k+'_observed':np.ones(16,dtype=bool) for k in schema});arrays['plot_observed'][0]=False
base=root/'base';base.mkdir();np.savez_compressed(base/'features.npz',**arrays);(base/'source-ids.json').write_text(json.dumps(list(range(16))));(base/'provenance.json').write_text('{}')
parent=root/'parent';pt=parent/'trials/stage_07';pt.mkdir(parents=True);(parent/'run-manifest.json').write_text('{"git_sha":"frozen"}')
text=rng.normal(size=(16,8)).astype('float32');np.save(pt/'item-embeddings.npy',text)
(pt/'representation.json').write_text(json.dumps(dict(fields=c['fields'],model=c['representation'],truncated_items=0)))
(pt/'sid-catalogue.json').write_text(json.dumps(dict(original_item_ids=list(range(16)))))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c['names_input']=dict(path=str(parent),legacy_path='absent',parent_git_sha='frozen',parent_manifest_sha256=sha(parent/'run-manifest.json'),trial_directory='trials/stage_07',embedding_sha256=sha(pt/'item-embeddings.npy'),representation_sha256=sha(pt/'representation.json'))
data=prepare_names_modalities(c,base,list(range(16)),root/'modalities')
with np.load(data/'features.npz') as f:
 np.testing.assert_array_equal(f['text'],text)
 for k in ['poster','plot','people']:np.testing.assert_array_equal(f[k],arrays[k])
assert not json.loads((data/'provenance.json').read_text())['text_replacement']['graph_people_used']
torch.manual_seed(42);m,_=build_rqvae(source,original/'configs/experiment/rqvae_train_flat.yaml',16,spec,modality_schema=schema,active_modalities=['text','poster','plot']);spec['expected_initial_state_sha256_by_seed']={'42':state_digest(dict(m.named_parameters()))};del m;c['quantizer']=spec
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16)
g=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(g.named_parameters()));del g
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=str(original),initial_state_sha256_by_seed={'42':initial})
for predictor in ['tiger_names','sasrec_id']:
 trial=dict(id=predictor,seed=42,predictor=predictor,stage=3 if predictor=='tiger_names' else 0,modalities=['text','poster','plot'] if predictor=='tiger_names' else [])
 result=run_control_trial(trial,c,plan,users,source,'cpu');p=root/'suite/trials'/predictor
 assert result['selection']['training_epochs']==20 and result['selection']['validation_checks']==4 and result['selection']['optimizer_updates']==20
 assert result['test']['evaluation_examples']==2
 with np.load(p/'validation-epoch-005.npz') as f:
  assert ('item_ids' if predictor=='sasrec_id' else 'semantic_ids') in f.files
 bench=json.loads((p/'inference-benchmark.json').read_text());assert len(bench['measurements'])==2
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=240)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
