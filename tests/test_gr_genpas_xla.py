"""CPU equivalence for TPU portability changes; never claims TPU compatibility."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = os.environ.get("RHPR_GENPAS_SOURCE")
READY = SOURCE and all(importlib.util.find_spec(name) for name in
    ("torch", "numpy", "transformers", "hydra", "pytorch_lightning", "pyarrow"))


@unittest.skipUnless(READY, "Requires pinned GenPAS source and GR dependencies")
class GenpasStaticAdaptersTest(unittest.TestCase):
    def test_windows_prefixes_beams_padding_and_epoch_selection_match(self):
        script = r'''
import json,sys,torch,numpy as np,copy
from pathlib import Path
from types import SimpleNamespace
from rhpr.gr.genpas import activate_source
from rhpr.gr.genpas_tpu import build_model,make_rows,train,fit_rkmeans
from rhpr.gr.genpas_xla import fast_windows,kwargs_from_collation,adapt_beams,epoch_batches,padded_batch,per_example_loss
from rhpr.gr.sid import SidIndex
source=activate_source(sys.argv[1]); output=Path(sys.argv[2]); torch.set_num_threads(1)
from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate,collate_fn_train
from src.data.loading.components.label_function import NextKTokenMasking
labels={'sequence_data':SimpleNamespace(transform=NextKTokenMasking(next_k=4))}
index=SidIndex(np.array([[i%16,i//16,0] for i in range(32)]),width=16)
rows=[{'sequence_data':torch.tensor(index.sids[np.arange(n)%32].reshape(-1)),
       'user_id':torch.tensor([i])} for i,n in enumerate([3,6,12,31])]
for seed in [0,1,42,101]:
 for maximum in [1,7,512,10000]:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed)
   expected=kwargs_from_collation(collate_with_sid_causal_duplicate(rows,sequence_field_name='sequence_data',
       sid_hierarchy=4,labels=labels,sequence_length=100,max_batch_size=maximum,padding_token=-1,masking_token=-1))
  state=torch.get_rng_state().clone()
  actual=kwargs_from_collation(fast_windows(rows,generator=torch.Generator().manual_seed(seed),
       collate=collate_fn_train,maximum=maximum,labels=labels))
  assert torch.equal(state,torch.get_rng_state())
  for name in actual:
   torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
model,cfg=build_model(index.sids,source,output,tiny=True);model.eval()
old=model._check_valid_prefix
batch=kwargs_from_collation(collate_fn_train(rows[:2],labels=labels,sequence_length=100,masking_token=-1,padding_token=-1))
with torch.no_grad():
 original=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
adapt_beams(model)
for depth in range(1,5):
 prefixes=torch.cat([torch.randint(0,16,(256,depth)),torch.tensor(index.sids[:,:depth].copy())])
 torch.testing.assert_close(model._check_valid_prefix(prefixes),old(prefixes),rtol=0,atol=0)
with torch.no_grad():
 adapted=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
for actual,expected in zip(adapted,original): torch.testing.assert_close(actual,expected,rtol=0,atol=0)
# Padding must contribute neither loss nor gradients.
reference=copy.deepcopy(model)
native_loss=per_example_loss(reference,batch).mean();native_loss.backward()
padded,mask=padded_batch(batch,8)
loss=(per_example_loss(model,padded)*mask).sum()/mask.sum();loss.backward()
for actual,expected in zip(model.parameters(),reference.parameters()):
 if actual.grad is not None: torch.testing.assert_close(actual.grad,expected.grad,rtol=2e-4,atol=2e-5)
users=[SimpleNamespace(user_id=i,train=(i,(i+1)%32,(i+2)%32),validation=(i+3)%32,test=(i+4)%32) for i in range(5)]
all_rows=make_rows(users,index.sids)
batches=list(epoch_batches(all_rows['train'],seed=42,source_size=2,physical_size=4,collate=collate_fn_train,labels=labels))
assert [len(b['future_ids']) for b in batches]==[4,4,4,3]
for b in batches:
 for row in b['future_ids']: assert tuple(row.tolist()) in index.sid_to_item
# Exercise the TPU runner's epoch orchestration on CPU with a CPU optimizer shim.
# This is a protocol test, explicitly not a hardware compatibility result.
class CpuOnly:
 @staticmethod
 def mark_step(): pass
 @staticmethod
 def optimizer_step(optimizer,barrier=True): optimizer.step()
 @staticmethod
 def save(value,path): torch.save(value,path)
model,cfg=build_model(index.sids,source,output,tiny=True)
config={'seed':42,'training':dict(max_epochs=20,validation_interval_epochs=10,patience_checks=10,
 source_user_batch_size=2,physical_batch_size=4,accumulate_grad_batches=2,evaluation_batch_size=3,max_seconds=60)}
result=train(model,cfg,all_rows,index,config,output,'cpu',CpuOnly)
curves=json.loads((output/'learning-curves.json').read_text())
assert [row['epoch'] for row in curves if 'validation' in row]==[10,20]
assert result['model_examples_consumed']==300 and result['test_evaluations']==1,result
assert result['optimizer_updates']==40,result
selected=torch.load(output/'generator-best.pt',weights_only=False)
for name,value in model.state_dict().items(): torch.testing.assert_close(value,selected['model'][name],rtol=0,atol=0)
# The replacement quantizer is explicitly declared, independent of the encoder.
small=fit_rkmeans(np.random.default_rng(42).normal(size=(32,16)).astype('float32'),
 dict(levels=3,codebook_size=4,n_init=1,max_iter=5,cpu_threads=1),output,42)
assert len(small.sid_to_item)==32
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            result = subprocess.run([sys.executable, "-c", script, SOURCE, tmp], env=env,
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-6000:])


if __name__ == "__main__":
    unittest.main()
