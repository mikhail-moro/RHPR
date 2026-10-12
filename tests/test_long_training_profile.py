from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy required")
class LongTrainingProfileTest(unittest.TestCase):
    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(100+i, (np.arange(30+i) % 11 + 1).astype(np.int32),
                                     np.arange(30+i, dtype=np.uint32)) for i in range(4))
        return LongHistoryCohort(users, np.arange(11, dtype=np.int64), {})

    def test_targets_and_negatives_are_identical_across_contexts_and_batch_boundaries(self):
        from rhpr.sasrec.long_training_profile import profile_batch, window_schedule
        cohort = self.cohort()
        windows = window_schedule(cohort, seed=42, count=8, raw_context_length=16)
        self.assertEqual(windows, window_schedule(cohort, seed=42, count=8, raw_context_length=16))
        self.assertNotEqual(windows, window_schedule(cohort, seed=43, count=8, raw_context_length=16))
        long = profile_batch(cohort, windows, context_length=16, num_negatives=7)
        short = profile_batch(cohort, windows, context_length=4, num_negatives=7)
        self.assertTrue(torch.equal(long["targets"][:, -4:], short["targets"]))
        self.assertTrue(torch.equal(long["item_ids"][:, -4:], short["item_ids"]))
        self.assertTrue(torch.equal(long["negative_ids"], short["negative_ids"]))
        singles = [profile_batch(cohort, (window,), context_length=4, num_negatives=7) for window in windows]
        self.assertTrue(torch.equal(short["negative_ids"], torch.cat([row["negative_ids"] for row in singles])))
        for window in windows:
            self.assertGreaterEqual(window.end, 16)
            self.assertLess(window.end, len(cohort.users[window.user_index].train))

    def test_fixed_negatives_produce_same_loss_and_gradients_as_seeded_sets(self):
        from rhpr.sasrec.model import SASRecModel
        from rhpr.sasrec.losses import next_item_loss
        from rhpr.sasrec.long_training_profile import profile_batch, window_schedule
        torch.manual_seed(3)
        model = SASRecModel(num_items=11, max_seq_len=4, d_model=8, n_layers=1,
                            n_heads=1, ffn_multiplier=1, dropout=0.)
        batch = profile_batch(self.cohort(), window_schedule(self.cohort(), seed=42,count=2,raw_context_length=16),
                              context_length=4, num_negatives=7)
        ids, valid, targets = batch["item_ids"],batch["valid_mask"],batch["targets"]
        generator = torch.Generator().manual_seed(31)
        fixed = torch.randint(1,12,(2,7),generator=generator)
        losses, gradients = [],[]
        for use_fixed in [False,True]:
            model.zero_grad(set_to_none=True)
            encoded, mask, sources = model.encode(ids,valid)
            loss = next_item_loss(name="sampled_softmax", model=model,encoded=encoded,valid_mask=mask,
                source_positions=sources,targets=targets,num_negatives=7,loss_tail_size=4,
                original_lengths=valid.sum(1),negative_ids=fixed if use_fixed else None,
                negative_generator=torch.Generator().manual_seed(31))
            loss.backward()
            losses.append(loss.detach())
            gradients.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        self.assertTrue(torch.equal(*losses))
        self.assertTrue(all(torch.equal(gradients[0][k],gradients[1][k]) for k in gradients[0]))
        with self.assertRaises(ValueError):
            next_item_loss(name="sampled_softmax",model=model,encoded=encoded,valid_mask=mask,
                source_positions=sources,targets=targets,num_negatives=7,negative_ids=torch.zeros((2,7),dtype=torch.long))

    def test_profile_requires_all_models_real_cuda_and_valid_accumulation(self):
        from rhpr.sasrec.long_training_profile import validate_long_profile_config
        config=json.loads((ROOT/'experiments/yambda_long_training_profile.json').read_text())
        validate_long_profile_config(config)
        for path,value in [(('architectures',),['sasrec']), (('profile','device'),'cpu'),
                           (('profile','physical_batch_sizes'),[3]), (('profile','tail_size'),32768),
                           (('model','use_time_features'),True), (('profile','learning_rate'),float('nan'))]:
            bad=copy.deepcopy(config);cursor=bad
            for key in path[:-1]:cursor=cursor[key]
            cursor[path[-1]]=value
            with self.assertRaises(ValueError):validate_long_profile_config(bad)

    def test_short_model_shapes_and_major_flops_do_not_use_raw16k_width(self):
        from rhpr.sasrec.long_training_profile import build_profile_model,_major_flops
        configuration=dict(d_model=16,n_heads=1,n_layers=1,ffn_multiplier=4,dropout=0.)
        for architecture in ['sasrec','hstu']:
            model=build_profile_model(architecture,11,4,configuration)
            ids=torch.tensor([[1,2,3,4]])
            output,mask,sources=model.encode(ids,ids!=0)
            self.assertEqual(output.shape,(1,4,16))
            self.assertTrue(torch.equal(sources,torch.arange(4)[None,:]))
            self.assertTrue(_major_flops(model,architecture,4)['available'])
            self.assertGreater(_major_flops(model,architecture,4)['per_user'],0)


if __name__=='__main__':unittest.main()
