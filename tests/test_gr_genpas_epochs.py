"""Finite epochs exercise the actual GenPAS model in an isolated namespace."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rhpr.gr.genpas import prepare_source
from rhpr.gr.genpas_epochs import epoch_order

SOURCE = os.environ.get("RHPR_GENPAS_SOURCE")
READY = SOURCE and all(importlib.util.find_spec(name) for name in
    ("torch", "transformers", "hydra", "pytorch_lightning", "pyarrow"))


class GenpasIntegrityTests(unittest.TestCase):
    def test_unpinned_source_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
                prepare_source(tmp)

    def test_epoch_visits_all_users_and_reshuffles_reproducibly(self):
        first = epoch_order(101, 42, 0)
        self.assertEqual(sorted(first), list(range(101)))
        self.assertEqual(first, epoch_order(101, 42, 0))
        self.assertNotEqual(first, epoch_order(101, 42, 1))


@unittest.skipUnless(READY, "Set RHPR_GENPAS_SOURCE and install GR dependencies")
class NativeGenpasEpochTests(unittest.TestCase):
    def test_native_model_window_epochs_sparse_validation_and_selected_test(self):
        script = r'''
import json, sys, torch
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from hydra.utils import instantiate
from rhpr.gr.genpas import activate_source, compose_beauty
from rhpr.gr.genpas_epochs import EpochWindows, fit_generator, make_datamodule
root, output = Path(sys.argv[1]), Path(sys.argv[2])
prepared = activate_source(root)
changed = [p.relative_to(root).as_posix() for p in (root/'src').rglob('*.py')
           if p.read_bytes() != (prepared/p.relative_to(root)).read_bytes()]
assert changed == ['src/data/loading/components/interfaces.py'], changed
assert (prepared/'License+Attribution.txt').read_bytes() == (root/'License+Attribution.txt').read_bytes()
from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate, collate_fn_train
from src.data.loading.components.label_function import NextKTokenMasking
torch.set_num_threads(1); torch.manual_seed(42)
sids = torch.tensor([[i,0,0,0] for i in range(16)])
(output/'sids').mkdir()
torch.save(sids.T, output/'sids/flan-t5-xxl_rkmeans_3_256_seed42.pt')
cfg = compose_beauty(prepared, output, output)
assert cfg.model.huggingface_model.config.num_layers == 4
assert cfg.model.decoder.config.num_layers == 4
assert cfg.model.mlp_layers == 2 and cfg.model.num_user_bins == 0
assert cfg.optim.optimizer._target_ == 'torch.optim.Adam'
assert cfg.optim.scheduler.scheduler_steps == 320000
for key, value in dict(vocab_size=16,d_model=8,d_ff=16,d_kv=4,num_heads=2,num_layers=1,dropout_rate=0.).items():
    cfg.model.huggingface_model.config[key] = value
cfg.model.decoder.config.num_layers = 1
model = instantiate(cfg.model)
evaluation_dtypes = []
model.decoder.decoder_mlp[0].register_forward_hook(
    lambda module, args, value: evaluation_dtypes.append(value.dtype) if not module.training else None)
rows = [{'sequence_data':sids[[i,(i+1)%16,(i+2)%16]].reshape(-1),
         'user_id':torch.tensor([i])} for i in range(5)]
labels = {'sequence_data':SimpleNamespace(transform=NextKTokenMasking(next_k=4))}
common = dict(labels=labels,sequence_length=12,padding_token=-1,masking_token=-1,oov_token=None)
windows = partial(collate_with_sid_causal_duplicate,sequence_field_name='sequence_data',
                  sid_hierarchy=4,max_batch_size=512,**common)
evaluation = partial(collate_fn_train,**common)
isolated = EpochWindows(windows,42)
state = torch.get_rng_state().clone()
actual = isolated(rows)
assert torch.equal(state,torch.get_rng_state())
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42); expected = windows(rows)
for key in actual[0].transformed_sequences:
    torch.testing.assert_close(actual[0].transformed_sequences[key],expected[0].transformed_sequences[key],rtol=0,atol=0)
config = {'seed':42,'training':dict(accelerator='cpu',precision='bf16-mixed',source_user_batch_size=2,
    evaluation_batch_size=3,accumulate_grad_batches=2,max_epochs=20,validation_interval_epochs=10,patience_checks=10)}
data = {stage:rows for stage in ['train','validation','test']}
dm = make_datamodule(data,windows,evaluation,config['training'],42)
batches = list(dm.loader('train'))
assert [len(b[0].mask) for b in batches] == [6,6,3]  # both source and optimizer remainders retained
config['training']['max_epochs'] = 19
try:
    fit_generator(model,data,windows,evaluation,config,output)
    raise AssertionError('off-boundary epoch cap accepted')
except ValueError as e:
    assert 'validation boundary' in str(e)
config['training']['max_epochs'] = 20
result = fit_generator(model,data,windows,evaluation,config,output)
exposure = json.loads((output/'training-exposure.json').read_text())
curves = json.loads((output/'learning-curves.json').read_text())
assert [row['epoch'] for row in curves] == [10,20], curves
assert [row['model_examples_consumed'] for row in curves] == [150,300], curves
assert result['completed_epochs'] == 20 and result['optimizer_updates'] == 40, result
assert result['model_examples_consumed'] == 300 and result['test_evaluations'] == 1
assert result['selected_epoch'] in [10,20]
assert exposure['source_rows_consumed'] == 100 and exposure['last_epoch_examples'] == 15
assert exposure['validation_epochs'] == [10,20]
selected = torch.load(result['selected_checkpoint'],map_location='cpu',weights_only=False)
for name,value in model.state_dict().items():
    if name in selected['state_dict'] and not any(tag in name for tag in ['loss','metric']):
        torch.testing.assert_close(value,selected['state_dict'][name],rtol=0,atol=0)
assert exposure['learning_rate'] > .0003  # native scheduler advances at completed epochs
assert evaluation_dtypes and set(evaluation_dtypes) == {torch.float32}, evaluation_dtypes
(output/'verified.json').write_text(json.dumps(result))
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLCONFIGDIR=tmp, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            result = subprocess.run([sys.executable, "-c", script, SOURCE, tmp], env=env,
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-6000:])
            self.assertEqual(json.loads((Path(tmp) / "verified.json").read_text())["validation_checks"], 2)


if __name__ == "__main__":
    unittest.main()
