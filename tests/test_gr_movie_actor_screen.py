import copy,gzip,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_actor_screen_2t4.json'


class ActorScreenTest(unittest.TestCase):
    def test_protocol_and_source_matrix(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        jobs=movie_assignments(c,2)
        self.assertEqual([len(j['trials']) for j in jobs],[5,5])
        self.assertEqual([{r['seed'] for r in j['trials']} for j in jobs],[{42},{101}])
        for change in ['epochs','weights','source','scheduler']:
            bad=copy.deepcopy(c)
            if change=='epochs':bad['training']['max_epochs']=100
            elif change=='weights':bad['trials'][-1]['reconstruction_weights']['people']=.7
            elif change=='source':bad['trials'][-1]['people_source']='shared_graph'
            else:bad['training']['scheduler']={'kind':'reduce_on_plateau'}
            with self.assertRaises(ValueError):validate(bad)

    def test_actor_graph_excludes_director_edges_and_pools_only_cast(self):
        from scripts.prepare_movie_people_deepwalk import prepare
        rows=[dict(item_id=1,cast=[dict(person_id=1),dict(person_id=2)],directors=[dict(person_id=999)]),
              dict(item_id=2,cast=[dict(person_id=2),dict(person_id=3)],directors=[dict(person_id=999)])]
        seen=[]
        class FakeWord2Vec:
            def __init__(self,sentences,**kwargs):
                seen.extend(sentences.people);self.wv={str(p):np.full(64,p,dtype=np.float32) for p in sentences.people}
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp,mock.patch.dict(os.environ,PYTHONHASHSEED='42'),mock.patch.dict(sys.modules,{'gensim.models':mock.Mock(Word2Vec=FakeWord2Vec)}):
            p=Path(tmp);(p/'catalog.gz').write_bytes(gzip.compress(json.dumps(rows).encode()));prepare(p/'catalog.gz',p/'out',role='cast')
            self.assertEqual(seen,[1,2,3]);x=np.load(p/'out/people.npy')
            np.testing.assert_array_equal(x[:,:64],np.array([[1.5]*64,[2.5]*64]));self.assertFalse(x[:,64:].any())
            self.assertEqual(json.loads((p/'out/people-provenance.json').read_text())['graph_role'],'cast')

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires native GRID')
    def test_native_new_graph_weight_and_fixed30_training(self):
        script=r'''
import json,hashlib,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.sid_study import state_digest
from rhpr.gr.movie_multimodal import run_multimodal_trial
from rhpr.gr.movie_data import TemporalUser
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid');torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text());schema=dict(text=8,poster=6,plot=5,people=128)
c['quantizer']=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=4,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=2,ae_warmup=dict(max_steps=4,probe_interval=2,stopping=dict(kind='fixed_steps')))
c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=4,evaluation_batch_size=2,max_seconds=120)
arrays={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()};masks={k:np.ones(16,dtype=bool) for k in schema};ids=list(range(16))
data=root/'modalities';data.mkdir();np.savez_compressed(data/'features.npz',**arrays,**{k+'_observed':v for k,v in masks.items()});(data/'source-ids.json').write_text(json.dumps(ids));(data/'provenance.json').write_text('{}')
a=root/'actor';a.mkdir();x=np.random.default_rng(88).normal(size=(16,128)).astype('float32');x[:,64:]=0;np.save(a/'people.npy',x);(a/'source-ids.json').write_text(json.dumps(ids));(a/'people-provenance.json').write_text(json.dumps(dict(graph_role='cast',no_interactions_used=True,seed=42)))
spec=dict(path=str(a));spec.update({key:hashlib.sha256((a/name).read_bytes()).hexdigest() for name,key in [('people.npy','people_sha256'),('source-ids.json','source_ids_sha256'),('people-provenance.json','provenance_sha256')]});c['actor_graph_input']=spec
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16);g=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(g.named_parameters()));del g
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=ids,original=str(original),initial_state_sha256_by_seed={'42':initial})
trial=next(t for t in c['trials'] if t['candidate']=='actor_w03' and t['seed']==42)
r=run_multimodal_trial(trial,c,plan,users,source,'cpu');p=root/'suite/trials'/r['trial'];q=json.loads((p/'quantizer.json').read_text());curves=json.loads((p/'training-curves.json').read_text())
assert r['candidate']=='actor_w03' and r['people_source']=='actor_graph' and r['fixed_epoch_screen'] and not r['budget_censored']
assert r['selection']['training_epochs']==30 and r['selection']['validation_checks']==6
assert [x['epoch'] for x in curves if 'validation' in x]==[5,10,15,20,25,30]
assert q['reconstruction_weights']['people']==.3 and q['warmup']['reconstruction_weights']['people']==.3
assert q['config']['reconstruction_columns']['people']==list(range(64))
assert (p/'actor-input-provenance.json').exists() and (p/'test-predictions.npz').exists()
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
