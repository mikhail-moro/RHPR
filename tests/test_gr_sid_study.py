"""SID evidence invariants: search objectives, paired sampling, source replay."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

READY = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "pyarrow", "sklearn", "duckdb"))
SOURCE = os.environ.get("RHPR_GENPAS_SOURCE")
NATIVE = READY and SOURCE and all(importlib.util.find_spec(name) for name in ("transformers", "hydra", "pytorch_lightning"))


@unittest.skipUnless(READY, "Requires SID analysis dependencies")
class EvidenceTests(unittest.TestCase):
    def test_exact_local_and_raw_are_distinct_and_beams_use_matching_scores(self):
        import numpy as np
        import torch
        from rhpr.gr.sid import SidIndex
        from rhpr.gr.study_search import exact_catalogue, beam_search
        # One branch has one valid child, another two. Invalid mass radically
        # differs. A local singleton must be probability 1, NOT its raw value.
        index = SidIndex(np.array([[0], [1], [1]]), width=4)

        class Scorer:
            batch_size = 1
            def __init__(self):
                self.index = index
            def logits(self, prefixes, owners):
                result = []
                for prefix in prefixes:
                    if not len(prefix):
                        result.append([math.log(.4), math.log(.5), math.log(.05), math.log(.05)])
                    elif prefix[0] == 0:
                        result.append([math.log(.01), math.log(.33), math.log(.33), math.log(.33)])
                    else:
                        result.append([math.log(.5), math.log(.3), math.log(.1), math.log(.1)])
                return torch.tensor(result)

        scorer = Scorer()
        scores = exact_catalogue(scorer, 0, chunk_size=1)
        np.testing.assert_allclose(np.exp(scores["raw"]), [.004, .25, .15], rtol=1e-6)
        np.testing.assert_allclose(np.exp(scores["local"]), [.4/.9, .5/.9*.5/.8, .5/.9*.3/.8], rtol=1e-6)
        self.assertEqual(int(scores["raw"].argmax()), 1)
        self.assertEqual(int(scores["local"].argmax()), 0)
        for scoring in ("raw", "local"):
            beams, trace = beam_search(scorer, 20, scoring, targets=index.sids[[0]])
            self.assertEqual(len(beams[0]), 3)  # No invalid beams if width > catalogue.
            for sid, score in beams[0]:
                self.assertAlmostEqual(score, scores[scoring][index.sid_to_item[sid]], places=6)
            self.assertTrue(any(row["target_prefix"] for row in trace))
        beams, trace = beam_search(scorer, 1, "raw", targets=index.sids[[0]])
        self.assertTrue(any(row["target_prefix"] and row["rank"] is None for row in trace))

    def test_shared_train_only_graph_features_panel_and_atomic_store(self):
        import numpy as np
        from rhpr.gr.data import UserSequence
        from rhpr.gr.study_index import prepare_shared, record_index, compare_indices, extract_features
        from rhpr.gr.study_store import StudyStore
        from rhpr.gr.sid import SidIndex
        from rhpr.gr.study_analysis import audit
        users = [UserSequence(7, (0, 1, 0), 2, 3), UserSequence(8, (1, 0, 1, 0), 2, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyStore(tmp)
            cases = prepare_shared(store, ["Title: T; Brand: Unknown; Categories: ['A']; Price: 2.5; "]*4,
                                   list(range(4)), users, context_items=2, panel_size=1, panel_seed=42)
            self.assertEqual(len(cases), 4)
            self.assertEqual(sum(row["exact_panel"] for row in cases), 2)
            for row in cases:
                self.assertEqual(row["panel_probability"], .5)
                self.assertEqual(len(row["visible_history"]), 2)
            self.assertFalse(any(row["target_item"] in {2, 3} for row in store.read("train_edges")))
            self.assertEqual(store.read("items")[2]["train_events"], 0)
            self.assertEqual(extract_features("bad")[0][2], "unparsed")
            (Path(tmp)/"tables/items/incomplete.pending").write_text("broken")
            self.assertEqual(len(store.read("items")), 4)
            with self.assertRaises(ValueError):
                store.write("items", [{"item_id": 0}])
            a = SidIndex(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]), width=2)
            b = SidIndex(1-a.semantic_codes, width=2)
            for name, index in (("a", a), ("b", b)):
                store.write("runs", [{"run_id": name, "quantizer_seed": 1, "model_seed": 42,
                    "data_seed": 42, "initial_state_sha256": "same", "config_json": "{}"}])
                record_index(store, name, index, users, cases,
                    np.random.default_rng(1).normal(size=(4, 3)).astype("float32"), np.ones((3, 2, 3), dtype="float32"))
            compare_indices(store, {"a": a, "b": b})
            for row in store.read("index_comparisons"):
                self.assertEqual(row["prefix_ari"], 1.)
                self.assertEqual(row["token_ari"], 1.)
            self.assertTrue(audit(store)["passed"])
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM items").fetchone()[0], 4)
                self.assertEqual(db.execute("SELECT count(*) FROM item_quality").fetchone()[0], 0)
            store.inventory()

    def test_disjoint_writer_shards_are_one_queryable_store(self):
        from rhpr.gr.study_store import StudyStore
        with tempfile.TemporaryDirectory() as tmp:
            parent = StudyStore(tmp)
            original = (Path(tmp)/"schema.json").read_bytes()
            a, b = StudyStore(tmp, shard="worker-0"), StudyStore(tmp, shard="worker-1")
            for writer, run in ((a, "a"), (b, "b")):
                writer.write("runs", [{"run_id": run, "quantizer_seed": 1, "model_seed": 42,
                    "data_seed": 42, "initial_state_sha256": "same", "config_json": "{}"}])
                writer.flush()
            self.assertEqual((Path(tmp)/"schema.json").read_bytes(), original)
            self.assertEqual({r["run_id"] for r in parent.read("runs")}, {"a", "b"})
            with parent.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM runs").fetchone()[0], 2)
            with self.assertRaises(FileExistsError):
                StudyStore(tmp, shard="worker-0")
            with self.assertRaises(ValueError):
                StudyStore(tmp, shard="../bad")

    def test_worker_assignment_device_guards_and_inherited_gpu_mapping(self):
        from copy import deepcopy
        from unittest.mock import patch
        from rhpr.gr.study_parallel import seed_assignments, worker_environment, check_devices
        config = {"quantizer_seeds": [11,22,33,44,55,66], "training": {"device":"cuda"},
                  "execution": {"worker_devices":["cuda:0","cuda:1"],"cpu_threads_per_worker":1}}
        assignments = seed_assignments(config)
        self.assertEqual([a["seeds"] for a in assignments], [[11,33,55],[22,44,66]])
        environment = worker_environment(config, assignments[1], environment={"CUDA_VISIBLE_DEVICES":"GPU-a,GPU-b"}, python_paths=["/bundle.zip","/offline/site"])
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-b")
        self.assertEqual(environment["PYTHONPATH"], os.pathsep.join(["/bundle.zip","/offline/site"]))
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        with patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(RuntimeError,"no fallback"):
                check_devices(config)
        config["execution"]["worker_devices"]=["cuda:0","cuda:0"]
        with self.assertRaises(ValueError):
            seed_assignments(config)


@unittest.skipUnless(NATIVE, "Requires pinned GenPAS source and analysis dependencies")
class NativeStudyTests(unittest.TestCase):
    def test_native_end_to_end_frozen_initialization_checkpoints_replay_and_permutation(self):
        script = r'''
import json,sys,torch,numpy as np,copy
from pathlib import Path
from types import SimpleNamespace
from rhpr.gr.sid_study import run_sid_study,make_batch
from rhpr.gr.data import file_sha256,UserSequence
from rhpr.gr.study_store import StudyStore
from rhpr.gr.study_analysis import audit
from rhpr.gr.study_index import window_manifest
from rhpr.gr.study_search import ModelScorer,exact_catalogue,beam_search,mask_history_events
from rhpr.gr.sid import SidIndex
root=Path(sys.argv[2]);data=root/'data';data.mkdir()
catalog=[dict(item_id=i,text=f"Title: item{i}; Brand: B; Categories: ['C']; Price: 1.0; ") for i in range(16)]
sequences=[dict(user_id=i,train=[i%16,(i+1)%16,(i+2)%16,(i+3)%16,(i+4)%16],validation=(i+5)%16,test=(i+6)%16) for i in range(4)]
(data/'catalog.json').write_text(json.dumps(catalog));(data/'sequences.json').write_text(json.dumps(sequences))
np.save(root/'embeddings.npy',np.random.default_rng(3).normal(size=(16,8)).astype('float32'))
config=dict(name='smoke',kind='gr_sid_index_study',model_seed=42,data_seed=123,quantizer_seeds=[11,22],source_path=sys.argv[1],data_path=str(data),data={},
 representation=dict(path=str(root/'embeddings.npy'),sha256=file_sha256(root/'embeddings.npy'),catalog_sha256=file_sha256(data/'catalog.json')),
 quantizer=dict(levels=3,codebook_size=4,n_init=1,max_iter=10,cpu_threads=1),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1),
 training=dict(device='cpu',precision='fp32',cpu_threads=1,max_epochs=2,validation_interval_epochs=1,batch_size=4,source_user_batch_size=2,
 max_windows_per_source_batch=7,context_items=4,input_tokens=20,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.),
 diagnostics=dict(context_items=3,evaluation_batch_size=3,exact_panel_size=2,exact_epochs=[2],exact_chunk_size=4,save_top_k=16,main_beam_width=10,panel_beam_widths=[10,50]))
output=root/'output';metrics=run_sid_study(config,output);store=StudyStore(output)
assert metrics['runs']==2 and all(np.isfinite(v) for v in metrics.values())
assert audit(store,verify_files=True)['passed']
runs=store.read('runs');assert len({r['initial_state_sha256'] for r in runs})==1
assert len(store.read('checkpoints'))==4
assert len([r for r in store.read('case_metrics') if r['case_id'].startswith('test:') and r['epoch']==1])==0
assert len(store.read('model_inputs'))==16
assert all(row['exact_top10_overlap']==1. for row in store.read('case_metrics') if row['search']=='beam' and row['beam_width']==50)
saved=torch.load(output/'rq-11/epoch-002.pt',weights_only=False)
assert saved['examples']==24  # Duplicate source draws are emitted only once.
# Actual collated input and target parity with the pinned source.
from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate,collate_fn_train
from src.data.loading.components.label_function import NextKTokenMasking
from rhpr.gr.genpas_xla import kwargs_from_collation
labels={'sequence_data':SimpleNamespace(transform=NextKTokenMasking(next_k=4))}
index=SidIndex(saved['sids'][:,:-1],width=4)
users=[UserSequence(row['user_id'],tuple(row['train']),row['validation'],row['test']) for row in sequences]
manifest=window_manifest(users,seed=123,epoch=1,source_size=4,maximum=7,context_items=4)
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(123)
 # Source user ordering must also match the manifest.
 from rhpr.gr.genpas_epochs import epoch_order
 order=epoch_order(4,123,0)
 rows=[{'sequence_data':torch.tensor(index.sids[list(users[i].train)].reshape(-1)), 'user_id':torch.tensor([users[i].user_id])} for i in order]
 native=kwargs_from_collation(collate_with_sid_causal_duplicate(rows,sequence_field_name='sequence_data',sid_hierarchy=4,
  labels=labels,sequence_length=20,max_batch_size=7,padding_token=-1,masking_token=-1))
ours=make_batch([users[r['user_id']].train[r['visible_start']:r['end']-1] for r in manifest],
 [r['target_item'] for r in manifest],[r['user_id'] for r in manifest],index,20,'cpu')
for key in ('input_ids','attention_mask_encoder','future_ids'):
 torch.testing.assert_close(ours[key],native[key],rtol=0,atol=0)
# Long contexts must be truncated identically; sampled duplicates remain unique.
for sampler_seed in (0,42):
 long_users=[UserSequence(i,tuple(j%16 for j in range(n)),1,2) for i,n in enumerate((3,6,40))]
 long_order=epoch_order(3,sampler_seed,0)
 long_rows=[{'sequence_data':torch.tensor(index.sids[list(long_users[i].train)].reshape(-1)),
  'user_id':torch.tensor([long_users[i].user_id])} for i in long_order]
 for maximum in (1,7,512,10000):
  picked=window_manifest(long_users,seed=sampler_seed,epoch=1,source_size=3,maximum=maximum,context_items=4)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(sampler_seed)
   expected=kwargs_from_collation(collate_with_sid_causal_duplicate(long_rows,sequence_field_name='sequence_data',
    sid_hierarchy=4,labels=labels,sequence_length=20,max_batch_size=maximum,padding_token=-1,masking_token=-1))
  actual=make_batch([long_users[r['user_id']].train[r['visible_start']:r['end']-1] for r in picked],
   [r['target_item'] for r in picked],[r['user_id'] for r in picked],index,20,'cpu')
  for key in ('input_ids','attention_mask_encoder','future_ids'):
   torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
# Reload native model. Same trained weights must produce invariant item scores
# under a global per-level token relabeling with corresponding parameter rows.
from rhpr.gr.sid_study import build_study_model
import src
source=Path(next(iter(src.__path__))).parent
model=build_study_model(index,source,root/'replay',config['generator']);model.load_state_dict(saved['model']);model.eval()
batch={k:v[:1] for k,v in ours.items()}
scores=exact_catalogue(ModelScorer(model,batch,index),0,chunk_size=8)
model.top_k_for_generation=2
with torch.no_grad():
 native_ids,native_scores=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
beams,_=beam_search(ModelScorer(model,batch,index),2,'local')
np.testing.assert_array_equal(np.array([p for p,s in beams[0]]),native_ids[0].numpy())
np.testing.assert_allclose(np.exp([s for p,s in beams[0]]),native_scores[0].numpy(),rtol=2e-5)
perms=[np.array([2,0,3,1]),np.array([1,0,3,2]),np.array([3,1,2,0]),np.arange(4)]
semantic=np.stack([perms[l][index.sids[:,l]] for l in range(3)],1)
changed_index=SidIndex(semantic,width=4)
changed_model=copy.deepcopy(model)
with torch.no_grad():
 for l,p in enumerate(perms):
  changed_model.item_sid_embedding_table_encoder.weight[l*4+torch.tensor(p)]=model.item_sid_embedding_table_encoder.weight[l*4:l*4+4]
  changed_model.decoder.decoder_mlp[l].weight[torch.tensor(p)]=model.decoder.decoder_mlp[l].weight
changed_batch=make_batch([users[manifest[0]['user_id']].train[manifest[0]['visible_start']:manifest[0]['end']-1]],
 [manifest[0]['target_item']],[manifest[0]['user_id']],changed_index,20,'cpu')
changed_scores=exact_catalogue(ModelScorer(changed_model,changed_batch,changed_index),0,chunk_size=8)
for scoring in scores: np.testing.assert_allclose(scores[scoring],changed_scores[scoring],atol=3e-6)
maskable={k:v[ours['attention_mask_encoder'].sum(-1)>4] for k,v in ours.items()}
masked=mask_history_events(maskable,[0,0],4)
assert torch.equal(masked['input_ids'][:,4:],maskable['input_ids'][:,4:])
assert not masked['attention_mask_encoder'][:,:4].any()
(root/'config.json').write_text(json.dumps(config))
'''
        replay = r'''
import sys,json
from rhpr.gr.study_analysis import replay_intervention,audit
from rhpr.gr.study_store import StudyStore
root=sys.argv[2]+'/output'
result=replay_intervention(root,'rq-11',2,'validation:0:5',[0],sys.argv[1])
assert len(result)==2 and {r['scoring'] for r in result}=={'raw','local'}
assert audit(StudyStore(root),verify_files=True)['passed']
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp, USE_TF="0", USE_FLAX="0",
                       PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            for source in (script, replay):
                result = subprocess.run([sys.executable, "-c", source, SOURCE, tmp],
                    env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-7000:])

    def test_two_fresh_worker_processes_match_serial_training_and_preserve_partial_results(self):
        script = r'''
import json,sys,torch,numpy as np,copy
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.data import file_sha256
from rhpr.gr.sid_study import run_sid_study
from rhpr.gr.study_store import StudyStore
from rhpr.gr.study_analysis import audit
root=Path(sys.argv[2]);data=root/'data';data.mkdir()
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item {i}') for i in range(16)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,(i+1)%16,(i+2)%16,(i+3)%16],validation=(i+4)%16,test=(i+5)%16) for i in range(4)]))
np.save(root/'vectors.npy',np.random.default_rng(3).normal(size=(16,8)).astype('float32'))
c=dict(name='parallel-smoke',kind='gr_sid_index_study',model_seed=42,data_seed=123,quantizer_seeds=[11,22,33,44],
 source_path=sys.argv[1],data_path=str(data),data={},
 representation=dict(path=str(root/'vectors.npy'),sha256=file_sha256(root/'vectors.npy'),catalog_sha256=file_sha256(data/'catalog.json')),
 quantizer=dict(levels=3,codebook_size=4,n_init=1,max_iter=5,cpu_threads=1),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1),
 training=dict(device='cpu',precision='fp32',cpu_threads=1,max_epochs=2,validation_interval_epochs=1,batch_size=4,
 source_user_batch_size=2,max_windows_per_source_batch=7,context_items=3,input_tokens=16,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.),
 diagnostics=dict(context_items=3,evaluation_batch_size=4,exact_panel_size=1,exact_epochs=[2],exact_chunk_size=4,
 save_top_k=16,main_beam_width=10,panel_beam_widths=[10]))
config_file=root/'config.json';config_file.write_text(json.dumps(c))
parallel=copy.deepcopy(c);parallel['execution']=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1)
r=run_sid_study(parallel,root/'parallel');assert r['completed_runs']==4 and r['parallel_workers']==2
store=StudyStore(root/'parallel');assert audit(store,verify_files=True)['passed']
plan=json.loads((root/'parallel/workers/plan.json').read_text())
assert [a['seeds'] for a in plan['assignments']]==[[11,33],[22,44]]
a=json.loads((root/'parallel/workers/worker-0.json').read_text());b=json.loads((root/'parallel/workers/worker-1.json').read_text())
assert a['pid']!=b['pid'] and a['status']==b['status']=='complete'
assert len(store.read('runs'))==4 and len({x['initial_state_sha256'] for x in store.read('runs')})==1
# Both workers start before either has completed all its assigned seeds.
assert len(list((root/'parallel/tables/checkpoints').glob('worker-*')))==2
# Direct failure-isolation exercise with the real worker loop: lose one index,
# still train the next assigned seed and retain complete/failed status honestly.
from rhpr.gr.study_parallel import launch_workers
windows={int(e):root/'parallel'/p for e,p in plan['windows'].items()}
failure_root=root/'failure';failure_root.mkdir()
import shutil
for name in ('schema.json','study-config.json'): shutil.copyfile(root/'parallel'/name,failure_root/name)
shutil.copytree(root/'parallel/inputs',failure_root/'inputs')
failed_store=StudyStore(failure_root)
for name in ('items','cases','events','case_history','training_windows'):
 failed_store.write(name,store.read(name))
failed_store.flush()
failed_windows={}
import pyarrow.parquet as pq,pyarrow as pa
for epoch in (1,2):
 rows=[row for row in failed_store.read('training_windows') if row['epoch']==epoch]
 path=failure_root/f'windows-{epoch}.parquet';pq.write_table(pa.Table.from_pylist(rows),path);failed_windows[epoch]=path
for seed in (33,22,44):
 (failure_root/f'rq-{seed}').mkdir();shutil.copyfile(root/'parallel'/f'rq-{seed}/semantic-ids.npy',failure_root/f'rq-{seed}/semantic-ids.npy')
# rq-11 exists without a SID file: the first seed fails, the second on its GPU
# still executes; the other process finishes both assigned seeds.
(failure_root/'rq-11').mkdir()
try: launch_workers(parallel,failure_root,Path(sys.argv[1]),failed_windows,plan['initial_state_sha256'])
except RuntimeError as error: assert '3/4 seeds' in str(error),error
else: raise AssertionError('partial study was incorrectly declared complete')
status=json.loads((failure_root/'workers/worker-0.json').read_text())
assert status['status']=='failed' and '33' in status['results'] and status['failures'][0]['seed']==11
'''
        serial = r'''
import sys,json,torch,numpy as np
from pathlib import Path
from rhpr.gr.sid_study import run_sid_study
from rhpr.gr.study_store import StudyStore
root=Path(sys.argv[2]);c=json.loads((root/'config.json').read_text());run_sid_study(c,root/'serial')
for seed in c['quantizer_seeds']:
 a=torch.load(root/f'parallel/rq-{seed}/epoch-002.pt',weights_only=False)
 b=torch.load(root/f'serial/rq-{seed}/epoch-002.pt',weights_only=False)
 for name in a['model']: torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
 assert torch.equal(a['torch_rng'],b['torch_rng'])
 assert a['examples']==b['examples'] and a['effective_updates']==b['effective_updates']
pa=StudyStore(root/'parallel');pb=StudyStore(root/'serial')
with pa.connect() as a,pb.connect() as b:
 query='SELECT run_id,epoch,case_id,scoring,search,beam_width,target_rank,ndcg10 FROM case_metrics ORDER BY ALL'
 assert a.execute(query).fetchall()==b.execute(query).fetchall()
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp, USE_TF="0", USE_FLAX="0",
                       PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            for source in (script, serial):
                result = subprocess.run([sys.executable, "-c", source, SOURCE, tmp], env=env,
                    capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-5000:])


if __name__ == "__main__":
    unittest.main()
