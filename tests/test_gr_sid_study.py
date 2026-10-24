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


if __name__ == "__main__":
    unittest.main()
