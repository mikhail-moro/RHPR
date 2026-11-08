import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_people_roles_2t4.json'


class PeopleRolesTest(unittest.TestCase):
    def test_protocol_and_assignment(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        jobs=movie_assignments(c,2)
        self.assertEqual(sorted(len(j['trials']) for j in jobs),[4,5])
        for key in ['people_role','modalities','reconstruction_weights']:
            bad=copy.deepcopy(c)
            bad['trials'][1][key]={'people_role':'unknown','modalities':['text'],
                'reconstruction_weights':{'people':.1}}[key]
            with self.assertRaises(ValueError):validate(bad)

    def test_selected_role_normalization_has_no_inactive_information(self):
        from rhpr.gr.multimodal_rqvae import normalize_modalities
        from rhpr.gr.movie_people_roles import role_columns
        values={'people':np.random.default_rng(7).normal(size=(20,128)).astype('float32'),
                'text':np.random.default_rng(8).normal(size=(20,8)).astype('float32')}
        masks={k:np.ones(20,dtype=bool) for k in values}
        original,_=normalize_modalities(values,masks)
        for role in ['cast','directors']:
            columns=role_columns(role);inactive=sorted(set(range(128))-set(columns))
            a,stats=normalize_modalities(values,masks,{'people':columns})
            changed={k:v.copy() for k,v in values.items()};changed['people'][:,inactive]+=5000
            b,_=normalize_modalities(changed,masks,{'people':columns})
            np.testing.assert_array_equal(a['people'],b['people'])
            np.testing.assert_array_equal(a['text'],original['text'])
            self.assertFalse(np.any(a['people'][:,inactive]))
            self.assertAlmostEqual(float(np.mean(a['people'][:,columns]**2)),1.,places=6)
            self.assertEqual(stats['people']['selected_columns'],columns)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires native GRID')
    def test_native_role_loss_gradients_initialization_and_fit(self):
        script=r'''
import json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.sid_study import state_digest
from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid');recipe=original/'configs/experiment/rqvae_train_flat.yaml';torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text());schema=dict(text=768,poster=768,plot=768,people=128)
for seed in c['replicate_seeds']:
 torch.manual_seed(seed);model,_=build_rqvae(source,recipe,512,c['quantizer'],modality_schema=schema,active_modalities=list(schema))
 assert state_digest(dict(model.named_parameters()))==c['quantizer']['expected_initial_state_sha256_by_seed'][str(seed)]
 del model
schema=dict(text=8,poster=6,plot=5,people=128)
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=4,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=2,ae_warmup=dict(max_steps=4,probe_interval=2,stopping=dict(kind='fixed_steps')))
spec['reconstruction_columns']={'people':list(range(64,128))}
torch.manual_seed(42);model,_=build_rqvae(source,recipe,16,spec,modality_schema=schema,active_modalities=list(schema));model.eval()
fields={k:torch.randn(16,d) for k,d in schema.items()};fields.update({k+'_observed':torch.ones(16,dtype=torch.bool) for k in schema});fields['people'][:,:64]=0
z=model.encode_modalities(fields);total,loss=model.reconstruction(z,fields)
total.backward();grad=model.decoder_heads['people'][-1].weight.grad
assert torch.count_nonzero(grad[:64])==0 and torch.count_nonzero(grad[64:])>0
with torch.no_grad():
 model.decoder_heads['people'][-1].bias[:64]+=1000
 _,new=model.reconstruction(z,fields);torch.testing.assert_close(loss['people'],new['people'])
arrays={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()};masks={k:np.ones(16,dtype=bool) for k in schema}
codes,report=fit_multimodal_rqvae(arrays,masks,spec,42,'cpu',source,recipe,root/'fit',list(schema))
assert codes.shape==(16,3) and report['reconstruction_batches']==4
assert report['warmup']['neural_transfer_verified'] and report['warmup']['fresh_rq_optimizer_verified']
assert report['warmup']['codebooks_unchanged']
stats=json.loads((root/'fit/modality-normalization.json').read_text());assert stats['people']['selected_columns']==list(range(64,128))
from rhpr.gr.movie_multimodal import run_multimodal_trial
from rhpr.gr.movie_data import TemporalUser
from rhpr.gr.grid import build_grid_generator
c['quantizer']=dict(spec);c['quantizer'].pop('reconstruction_columns')
c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=4,evaluation_batch_size=2,max_seconds=120,min_epochs=26,max_epochs=26)
data=root/'modalities';data.mkdir();np.savez_compressed(data/'features.npz',**arrays,**{k+'_observed':v for k,v in masks.items()});(data/'source-ids.json').write_text(json.dumps(list(range(16))));(data/'provenance.json').write_text('{}')
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16);g=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(g.named_parameters()));del g
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=str(original),initial_state_sha256_by_seed={'42':initial})
result=run_multimodal_trial(c['trials'][2],c,plan,users,source,'cpu');p=root/'suite/trials'/result['trial']
assert result['people_role']=='directors' and result['selection']['resumed_from_epoch']==20 and result['selection']['training_epochs']==26
assert (p/'test-predictions.npz').exists() and json.loads((p/'resume-verification.json').read_text())['rng_restored_and_verified']
assert json.loads((p/'quantizer.json').read_text())['config']['reconstruction_columns']['people']==list(range(64,128))
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
