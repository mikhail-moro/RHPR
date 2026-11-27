from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(find_spec('torch'), 'PyTorch required')
class YambdaSidMaskTest(unittest.TestCase):
    def test_routes_next_code_with_exact_recent_boundary_and_padding(self):
        import torch

        from rhpr.gr.yambda_masked_decoder import sid_attention_mask

        observed = torch.ones((2, 517), dtype=torch.bool)
        observed[1, :8] = False
        mask = sid_attention_mask(observed, observed.shape[-1])[:, 0]
        # Query 513 predicts code 1 of item 128. Item 97 is exactly 31 back.
        self.assertTrue(mask[0, 513, 2 + 4 * 97 + 3])
        self.assertTrue(mask[0, 513, 2 + 4 * 96])
        self.assertFalse(mask[0, 513, 2 + 4 * 96 + 3])
        self.assertFalse(mask[0, 513, 514])
        # On the suffix query, the entire valid prefix is visible.
        torch.testing.assert_close(mask[:, 516], observed)
        self.assertTrue(mask[1, 513, 8:10].all())
        self.assertFalse(mask[1, 513, :8].any())
        for position in range(513, 517):
            cached = sid_attention_mask(observed[:, :position+1], 1)[:, 0, 0]
            torch.testing.assert_close(cached, mask[:, position, :position+1])

    def test_packed_prefill_covers_every_allowed_pair_once(self):
        import torch

        from rhpr.gr.yambda_masked_decoder import packed_query_plan, sid_attention_mask

        length = 514
        mask = sid_attention_mask(torch.ones((1, length), dtype=torch.bool), length)[0, 0]
        plan = packed_query_plan(length, 'cpu')
        torch.testing.assert_close(torch.cat([rows for rows, _ in plan]).sort().values, torch.arange(length))
        materialized = torch.zeros_like(mask)
        for rows, columns in plan:
            materialized[rows[:, None], columns] = True
        self.assertTrue((~mask | materialized).all())
        self.assertLess(materialized.count_nonzero(), length**2)
