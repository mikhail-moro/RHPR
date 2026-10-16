from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch required")
class OldPrefixNativeAuditTest(unittest.TestCase):
    def summary_backward(self, conditioning="recent", query_source="old_suffix"):
        from rhpr.sasrec.query_ssm import QuerySSMCompressor
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(918273)
            model = QuerySSMCompressor(d_model=8, summary_tokens=2, tail_size=6,
                query_context_size=4, n_heads=2, d_state=4, d_inner=8, gate_rank=4,
                write_conditioning=conditioning, dropout=0., query_source=query_source)
            mask = torch.arange(13)[None, :] < torch.tensor([13, 11])[:, None]
            inputs = torch.randn(2, 13, 8).masked_fill(~mask[:, :, None], 0).requires_grad_()
            outputs = model(inputs, mask)[0]
            (outputs[:, :2] * torch.randn(2, 2, 8)).sum().backward()
        return model, inputs

    def test_old_suffix_summary_gradient_reaches_scan_and_query_without_raw_tail(self):
        from rhpr.sasrec.query_profile import _critical_gradient_norms
        for conditioning in ("recent", "slot_only"):
            with self.subTest(conditioning=conditioning):
                model, inputs = self.summary_backward(conditioning)
                norms = _critical_gradient_norms(model, inputs, (7, 5), 4, "old_suffix")
                self.assertGreater(norms["old_input_before_query"], 0)
                self.assertTrue(all(value > 0 for value in norms.values()))
                self.assertEqual(torch.count_nonzero(inputs.grad[0, 7:]).item(), 0)
                self.assertEqual(torch.count_nonzero(inputs.grad[1, 5:]).item(), 0)

    def test_old_suffix_audit_rejects_disconnected_history_before_query(self):
        from rhpr.sasrec.query_profile import _critical_gradient_norms
        model, inputs = self.summary_backward()
        inputs.grad[0, :3] = 0
        inputs.grad[1, :1] = 0
        with self.assertRaisesRegex(RuntimeError, "zero or nonfinite"):
            _critical_gradient_norms(model, inputs, (7, 5), 4, "old_suffix")

    def test_old_suffix_audit_rejects_any_raw_tail_gradient(self):
        from rhpr.sasrec.query_profile import _critical_gradient_norms
        model, inputs = self.summary_backward()
        inputs.grad[1, 5, 0] = 1e-12
        with self.assertRaisesRegex(RuntimeError, "future raw tail"):
            _critical_gradient_norms(model, inputs, (7, 5), 4, "old_suffix")

    def test_recent_start_audit_still_rejects_gradient_after_query_support(self):
        from rhpr.sasrec.query_profile import _critical_gradient_norms
        model, inputs = self.summary_backward(query_source="recent_start")
        _critical_gradient_norms(model, inputs, (7, 5), 4)
        inputs.grad[0, 11, 0] = 1e-12
        with self.assertRaisesRegex(RuntimeError, "future raw tail"):
            _critical_gradient_norms(model, inputs, (7, 5), 4)

    def test_invalid_source_fails_before_cuda_import(self):
        from rhpr.sasrec.query_profile import audit_native_query_scan
        with self.assertRaisesRegex(ValueError, "query_source"):
            audit_native_query_scan(torch.device("cpu"), query_source="future")


if __name__ == "__main__":
    unittest.main()
