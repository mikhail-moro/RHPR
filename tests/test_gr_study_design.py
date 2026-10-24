"""Crossed initialization controls, fixed-index imports and repair ceilings."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

READY = all(importlib.util.find_spec(k) for k in ('numpy','torch','pyarrow','duckdb','sklearn'))
SOURCE = os.environ.get('RHPR_GENPAS_SOURCE')
NATIVE = READY and SOURCE and all(importlib.util.find_spec(k) for k in ('transformers','hydra','pytorch_lightning'))


@unittest.skipUnless(READY, 'Requires SID dependencies')
class StudyDesignTests(unittest.TestCase):
    def test_crossed_jobs_are_unique_paired_and_balanced_over_cards(self):
        from rhpr.gr.study_design import study_jobs, initial_hash_for
        from rhpr.gr.study_parallel import seed_assignments
        c=dict(quantizer_seeds=[11,22],model_seed=101,model_seeds=[101,202,303],
            training=dict(device='cuda'),execution=dict(worker_devices=['cuda:0','cuda:1'],cpu_threads_per_worker=1))
        jobs=study_jobs(c)
        self.assertEqual(len({j['run_id'] for j in jobs}),6)
        self.assertEqual({(j['quantizer_seed'],j['model_seed']) for j in jobs},
            {(q,m) for q in [11,22] for m in [101,202,303]})
        a=seed_assignments(c)
        self.assertEqual([[j['quantizer_seed'] for j in x['jobs']] for x in a],[[11,22,11],[22,11,22]])
        self.assertEqual(initial_hash_for({'101':'hash'},101),'hash')
        for bad in [[101,101],[202,101],[-1,101],[]]:
            with self.assertRaises(ValueError):study_jobs({**c,'model_seeds':bad})

    def test_import_is_exact_and_rejects_changed_bytes_and_invalid_suffix(self):
        import numpy as np
        from rhpr.gr.sid import SidIndex
        from rhpr.gr.data import file_sha256
        from rhpr.gr.study_design import load_frozen_index
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original';original.mkdir();out=root/'out';out.mkdir()
            index=SidIndex(np.array([[0,0,0],[0,0,0],[1,1,1]]),width=2)
            np.save(original/'semantic-ids.npy',index.sids)
            np.save(original/'quantizer-centroids.npy',np.zeros((3,2,4),dtype=np.float32))
            (original/'quantizer.json').write_text(json.dumps(dict(seed=11,levels=3,codebook_size=2)))
            source=dict(quantizer_seed=11,files={name:dict(path=str(original/name),sha256=file_sha256(original/name))
                for name in ['semantic-ids.npy','quantizer-centroids.npy','quantizer.json']})
            imported,_=load_frozen_index(np.zeros((3,4)),dict(levels=3,codebook_size=2),source,out)
            np.testing.assert_array_equal(imported.sids,index.sids)
            for name in source['files']:self.assertEqual((out/name).read_bytes(),(original/name).read_bytes())
            (original/'quantizer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                load_frozen_index(np.zeros((3,4)),dict(levels=3,codebook_size=2),source,out)

    def test_local_repairs_preserve_mass_and_are_maximal_for_true_parent(self):
        import numpy as np
        from rhpr.gr.study_ceiling import repair_true_decisions,target_rank
        # P(root0)=.2, P(root1)=.8; root0 children [.1,.9]. Target0.
        sids=np.array([[0,0],[0,1],[1,0]])
        scores=np.log([.02,.18,.8]);lp=np.log([.2,.1])
        second=repair_true_decisions(scores,sids,0,lp,[1])
        np.testing.assert_allclose(np.exp(second),[.2,0,.8])
        first=repair_true_decisions(scores,sids,0,lp,[0])
        np.testing.assert_allclose(np.exp(first),[.1,.9,0])
        all_=repair_true_decisions(scores,sids,0,lp,[0,1])
        np.testing.assert_allclose(np.exp(all_),[1,0,0])
        for repaired in [scores,second,first,all_]:
            self.assertAlmostEqual(np.exp(repaired).sum(),1.)
            self.assertLessEqual(target_rank(repaired,0),target_rank(scores,0))
        # A repaired child boosts ALL its descendant items, not only target.
        ids=np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0]])
        raw=np.log([.03,.06,.11,.8]);repaired=repair_true_decisions(raw,ids,0,np.log([.2,.45,1/3]),[1])
        self.assertAlmostEqual(repaired[0]-repaired[1],raw[0]-raw[1])
        self.assertTrue(np.isneginf(repaired[2]))
        self.assertEqual(repaired[3],raw[3])
        self.assertEqual(target_rank(np.array([0.,0.,-1.]),1),2)


@unittest.skipUnless(NATIVE,'Requires pinned native GenPAS source')
class NativeCrossedStudyTests(unittest.TestCase):
    def test_frozen_crossed_workers_and_probes_preserve_training_trajectory(self):
        script=r'''
import sys,json,copy,torch,numpy as np
from pathlib import Path
from rhpr.gr.sid_study import run_sid_study,fit_quantizer
from rhpr.gr.study_store import StudyStore
from rhpr.gr.study_analysis import audit
from rhpr.gr.data import file_sha256
root=Path(sys.argv[2]);data=root/'data';data.mkdir()
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item {i}') for i in range(16)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,(i+1)%16,(i+2)%16,(i+3)%16],validation=(i+4)%16,test=(i+5)%16) for i in range(4)]))
values=np.random.default_rng(3).normal(size=(16,8)).astype('float32');np.save(root/'vectors.npy',values)
quantizer=dict(levels=3,codebook_size=4,n_init=1,max_iter=5,cpu_threads=1)
frozen={}
for seed in [11,22]:
 folder=root/f'frozen-{seed}';folder.mkdir();fit_quantizer(values,quantizer,seed,folder)
 frozen[str(seed)]=dict(quantizer_seed=seed,files={name:dict(path=str(folder/name),sha256=file_sha256(folder/name))
   for name in ['semantic-ids.npy','quantizer-centroids.npy','quantizer.json']})
c=dict(name='crossed-smoke',kind='gr_sid_index_study',model_seed=5,model_seeds=[5,6],data_seed=123,quantizer_seeds=[11,22],frozen_indices=frozen,
 source_path=sys.argv[1],data_path=str(data),data={},
 representation=dict(path=str(root/'vectors.npy'),sha256=file_sha256(root/'vectors.npy'),catalog_sha256=file_sha256(data/'catalog.json')),
 quantizer=quantizer,generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1),
 training=dict(device='cpu',precision='fp32',cpu_threads=1,max_epochs=2,validation_interval_epochs=1,batch_size=4,
  source_user_batch_size=2,max_windows_per_source_batch=7,context_items=3,input_tokens=16,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.),
 diagnostics=dict(context_items=3,evaluation_batch_size=4,exact_panel_size=1,exact_epochs=[2],exact_chunk_size=4,
  save_top_k=16,main_beam_width=10,panel_beam_widths=[10]))
(root/'config.json').write_text(json.dumps(c))
parallel=copy.deepcopy(c);parallel['execution']=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1)
parallel['diagnostics']['training_probe']=dict(target_items=[5,6],interval_updates=2,train_examples_per_item=2,validation_examples_per_item=2,batch_size=2)
r=run_sid_study(parallel,root/'parallel');assert r['completed_runs']==4
store=StudyStore(root/'parallel');assert audit(store,verify_files=True)['passed']
runs=store.read('runs');assert len(runs)==4
for m in [5,6]:assert len({r['initial_state_sha256'] for r in runs if r['model_seed']==m})==1
assert len({r['initial_state_sha256'] for r in runs})==2
assert not (root/'parallel/initial-state.pt').exists()
for m in [5,6]:assert (root/f'parallel/initial-state-ms-{m}.pt').exists()
for r in runs:
 assert list((root/'parallel'/r['run_id']/'probes').glob('step-*.parquet'))
 assert (root/'parallel'/r['run_id']/'probes/step-000000.parquet').is_file()
'''
        compare=r'''
import sys,json,torch,numpy as np
from pathlib import Path
from rhpr.gr.sid_study import run_sid_study
from rhpr.gr.study_store import StudyStore
from rhpr.gr.study_analysis import audit
root=Path(sys.argv[2]);c=json.loads((root/'config.json').read_text());run_sid_study(c,root/'serial')
assert audit(StudyStore(root/'serial'),verify_files=True)['passed']
for q in [11,22]:
 for m in [5,6]:
  run=f'rq-{q}-ms-{m}'
  a=torch.load(root/'parallel'/run/'epoch-002.pt',weights_only=False)
  b=torch.load(root/'serial'/run/'epoch-002.pt',weights_only=False)
  for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
  assert torch.equal(a['torch_rng'],b['torch_rng'])
  assert a['examples']==b['examples'] and a['effective_updates']==b['effective_updates']
  for name in ['semantic-ids.npy','quantizer-centroids.npy']:
   assert (root/f'parallel/rq-{q}'/name).read_bytes()==(root/f'frozen-{q}'/name).read_bytes()
'''
        with tempfile.TemporaryDirectory() as tmp:
            for code in [script,compare]:
                p=subprocess.run([sys.executable,'-c',code,SOURCE,tmp],text=True,capture_output=True,timeout=240)
                self.assertEqual(p.returncode,0,p.stdout[-2000:]+'\n'+p.stderr[-6000:])
