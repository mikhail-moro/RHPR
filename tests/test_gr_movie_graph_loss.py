import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_graph_loss_weight_2t4.json'


class MovieGraphLossTest(unittest.TestCase):
    def test_paired_weights_fixed_warmup_and_full_training_contract(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        jobs=movie_assignments(c,2);self.assertEqual([len(j['trials']) for j in jobs],[1,1])
        old=json.loads((ROOT/'experiments/movielens100k_multimodal_ae_2t4.json').read_text())
        for key in ['generator','data','movie_input','modality_input','fields','representation']:
            self.assertEqual(c[key],old[key],key)
        with self.assertRaises(RuntimeError):movie_assignments(c,1)
        for change in ['weight','text_weight','seed','warmup','generator_epochs','vectors']:
            bad=copy.deepcopy(c)
            if change=='weight':bad['trials'][1]['reconstruction_weights']['people']=.3
            elif change=='text_weight':bad['trials'][1]['reconstruction_weights']['text']=.1
            elif change=='seed':bad['trials'][1]['seed']=101
            elif change=='warmup':bad['quantizer']['ae_warmup']['max_steps']=5000
            elif change=='generator_epochs':bad['training']['max_epochs']=20
            else:bad['modality_input']['features_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):validate(bad)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires pinned native GRID')
    def test_native_weighted_gradients_fixed_ae_rq_and_full_training(self):
        script=r'''
import copy,json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.sid_study import state_digest
from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
from rhpr.gr.movie_multimodal import run_multimodal_trial
from rhpr.gr.movie_data import TemporalUser
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid');recipe=original/'configs/experiment/rqvae_train_flat.yaml';torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text());schema=dict(text=768,poster=768,plot=768,people=128)
pins=[]
for trial in c['trials']:
 spec=copy.deepcopy(c['quantizer']);spec['reconstruction_weights']=trial['reconstruction_weights'];torch.manual_seed(42)
 m,_=build_rqvae(source,recipe,512,spec,modality_schema=schema,active_modalities=list(schema))
 pins.append(state_digest(dict(m.named_parameters())));assert pins[-1]==spec['expected_initial_state_sha256_by_seed']['42'];del m
assert pins[0]==pins[1]
schema=dict(text=8,poster=6,plot=5,people=4)
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=8,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=4,ae_warmup=dict(max_steps=8,probe_interval=2,stopping=dict(kind='fixed_steps')))
build=lambda sp:build_rqvae(source,recipe,16,sp,modality_schema=schema,active_modalities=list(schema))[0]
torch.manual_seed(42);a=build(spec);weighted=copy.deepcopy(spec);weighted['reconstruction_weights']=dict(text=1.,poster=1.,plot=1.,people=.1);torch.manual_seed(42);b=build(weighted)
assert state_digest(dict(a.named_parameters()))==state_digest(dict(b.named_parameters()))
f={k:torch.randn(16,d) for k,d in schema.items()};f.update({k+'_observed':torch.ones(16,dtype=torch.bool) for k in schema});a.eval();b.eval()
za=a.encode_modalities(f);zb=b.encode_modalities(f);assert torch.equal(za,zb)
la,ha=a.reconstruction(za,f);lb,hb=b.reconstruction(zb,f)
assert all(torch.equal(ha[k],hb[k]) for k in schema)
assert torch.allclose(lb,ha['text']+ha['poster']+ha['plot']+.1*ha['people'])
la.backward();lb.backward()
for name in ['text','people']:
 ga=a.decoder_heads[name][0].weight.grad;gb=b.decoder_heads[name][0].weight.grad
 torch.testing.assert_close(gb,ga*(.1 if name=='people' else 1.))
for value in [-1.,float('nan'),float('inf'),True]:
 bad=copy.deepcopy(spec);bad['reconstruction_weights']={'people':value}
 try:build(bad)
 except Exception:pass
 else:raise AssertionError('invalid reconstruction weight accepted')
arrays={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()};masks={k:np.ones(16,dtype=bool) for k in schema};masks['plot'][:3]=False
codes,report=fit_multimodal_rqvae(arrays,masks,weighted,42,'cpu',source,recipe,root/'fit',list(schema))
assert report['warmup']['steps']==8 and report['warmup']['stopping']['reason']=='fixed_steps'
assert report['warmup']['reconstruction_weights']['people']==.1 and report['reconstruction_weights']['people']==.1
assert report['warmup']['codebooks_unchanged'] and report['warmup']['fresh_rq_optimizer_verified'] and report['warmup']['neural_transfer_verified']
assert report['reconstruction_batches']==8 and report['initialization_latent_audit']['max_absolute_difference']<1e-5
q=json.loads((root/'fit/codebook-learning-curves.json').read_text())[-1]
assert abs(q['weighted_reconstruction_mse']-sum(weighted['reconstruction_weights'][k]*v for k,v in q['reconstruction_mse_by_head'].items()))<1e-8
c['quantizer']=weighted;c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=4,evaluation_batch_size=2,max_seconds=120,min_epochs=26,max_epochs=26)
data=root/'modalities';data.mkdir();np.savez_compressed(data/'features.npz',**arrays,**{k+'_observed':v for k,v in masks.items()});(data/'source-ids.json').write_text(json.dumps(list(range(16))));(data/'provenance.json').write_text('{}')
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16);g=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(g.named_parameters()));del g
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=str(original),initial_state_sha256_by_seed={'42':initial})
r=run_multimodal_trial(c['trials'][1],c,plan,users,source,'cpu');p=root/'suite/trials'/r['trial'];rows=json.loads((p/'training-curves.json').read_text())
assert r['selection']['training_epochs']==26 and r['selection']['resumed_from_epoch']==20
assert [x['epoch'] for x in rows if 'validation' in x]==[5,10,15,20]+list(range(21,27))
assert (p/'prefix-training-curves.json').exists() and (p/'test-predictions.npz').exists()
assert json.loads((p/'resume-verification.json').read_text())['rng_restored_and_verified']
assert r['reconstruction_weights']['people']==.1 and r['ae_steps']==8
assert r['budget_censored']==(r['selection']['stopping_reason']=='epoch_limit')
assert c['quantizer']['reconstruction_weights']['people']==.1
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
