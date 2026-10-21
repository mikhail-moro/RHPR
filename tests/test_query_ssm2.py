from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SSDReferenceTest(unittest.TestCase):
    def test_closed_form_two_event_recurrence_and_bias(self):
        from rhpr.sasrec.query_ssm2 import ssd_scan_reference

        values = {
            "x": torch.tensor([[[[2.]], [[3.]]]]),
            "dt": torch.tensor([[[.2], [.4]]]),
            "A": torch.tensor([-.7]),
            "B": torch.tensor([[[[.5]], [[-.6]]]]),
            "C": torch.tensor([[[[.8]], [[1.1]]]]),
        }
        first = .2 * .5 * 2
        second = torch.exp(torch.tensor(-.7 * .4)) * first + .4 * -.6 * 3
        expected = torch.tensor([[[[.8 * first]], [[1.1 * second]]]])
        actual = ssd_scan_reference(**values, dt_softplus=False)
        torch.testing.assert_close(actual, expected)
        biased = ssd_scan_reference(**values, dt_bias=torch.tensor([.2]))
        values["dt"] = values["dt"] + .2
        torch.testing.assert_close(biased, ssd_scan_reference(**values), rtol=0, atol=0)

    def test_scalar_identity_matches_independent_diagonal_scan_and_gradients(self):
        from rhpr.sasrec.query_ssm import selective_scan_reference
        from rhpr.sasrec.query_ssm2 import ssd_scan_reference

        torch.manual_seed(61)
        values = {
            "x": torch.randn(2, 5, 2, 3),
            "dt": torch.randn(2, 5, 2) * .1,
            "A": -torch.rand(2) - .2,
            "B": torch.randn(2, 5, 1, 3) * .2,
            "C": torch.randn(2, 5, 1, 3) * .2,
            "dt_bias": torch.full((2,), -2.),
        }
        left = {key: value.clone().requires_grad_() for key, value in values.items()}
        right = {key: value.clone().requires_grad_() for key, value in values.items()}
        actual = ssd_scan_reference(**left)
        expected = selective_scan_reference(
            u=right["x"].reshape(2, 5, 6).transpose(1, 2),
            delta=right["dt"].repeat_interleave(3, dim=2).transpose(1, 2),
            A=right["A"].repeat_interleave(3)[:, None].expand(-1, 3),
            B=right["B"][:, :, 0].transpose(1, 2),
            C=right["C"][:, :, 0].transpose(1, 2),
            delta_bias=right["dt_bias"].repeat_interleave(3),
        ).transpose(1, 2).reshape_as(actual)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        weights = torch.linspace(-.7, 1.4, 3)
        (actual[:, -1] * weights).square().sum().backward()
        (expected[:, -1] * weights).square().sum().backward()
        for key in values:
            with self.subTest(input=key):
                self.assertTrue(bool(torch.isfinite(left[key].grad).all()))
                self.assertGreater(float(left[key].grad.abs().sum()), 0)
                torch.testing.assert_close(left[key].grad, right[key].grad, rtol=2e-5, atol=2e-7)
        self.assertGreater(float(left["x"].grad[:, 0].abs().sum()), 0)

    def test_grouped_reads_and_writes_match_separate_heads(self):
        from rhpr.sasrec.query_ssm2 import ssd_scan_reference

        torch.manual_seed(71)
        x, dt = torch.randn(1, 4, 4, 2), torch.randn(1, 4, 4)
        A, B, C = -torch.rand(4), torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3)
        actual = ssd_scan_reference(x, dt, A, B, C)
        for head in range(4):
            expected = ssd_scan_reference(x[:, :, head:head + 1], dt[:, :, head:head + 1],
                                          A[head:head + 1], B[:, :, head // 2:head // 2 + 1],
                                          C[:, :, head // 2:head // 2 + 1])
            torch.testing.assert_close(actual[:, :, head:head + 1], expected)
        with self.assertRaisesRegex(ValueError, "divisibility"):
            ssd_scan_reference(x, dt, A, B[:, :, :1].expand(-1, -1, 3, -1),
                               C[:, :, :1].expand(-1, -1, 3, -1))


@unittest.skipIf(torch is None, "torch is not installed")
class QuerySSM2Test(unittest.TestCase):
    def compressor(self, **kwargs):
        from rhpr.sasrec.query_ssm2 import make_query_ssm2_compressor

        defaults = dict(d_model=8, summary_tokens=2, tail_size=3,
                        query_context_size=2, n_heads=2, d_state=3,
                        d_inner=4, gate_rank=3, head_dim=2, chunk_size=4)
        defaults.update(kwargs)
        return make_query_ssm2_compressor(**defaults).eval()

    def test_scan_shapes_query_conditioning_and_no_extra_mamba_features(self):
        torch.manual_seed(107)
        model = self.compressor()
        memory, queries = torch.randn(2, 7, 8), torch.randn(2, 2, 8)
        before = model.build_scan_inputs(memory, queries)
        changed = model.build_scan_inputs(memory, queries + 2 * torch.randn_like(queries))
        self.assertEqual(before["x"].shape, (4, 7, 2, 2))
        self.assertEqual(before["dt"].shape, (4, 7, 2))
        self.assertEqual(before["B"].shape, (4, 7, 1, 3))
        self.assertEqual(before["C"].shape, (4, 7, 1, 3))
        self.assertEqual(before["A"].shape, (2,))
        for key in ("x", "A", "dt_bias"):
            torch.testing.assert_close(before[key], changed[key], rtol=0, atol=0)
        for key in ("B", "dt", "C"):
            self.assertFalse(torch.allclose(before[key], changed[key]), key)
        torch.testing.assert_close(before["C"][:, :1].expand_as(before["C"]), before["C"])
        metadata = model.architecture_metadata()
        self.assertFalse(metadata["diagonal_qssm_checkpoint_compatible"])
        self.assertEqual(metadata["bc_groups"], 1)
        self.assertFalse(metadata["convolution"])
        self.assertFalse(metadata["d_skip"])
        self.assertFalse(metadata["output_gate"])
        self.assertEqual(before["dt_bias"].dtype, torch.float32)
        steps = torch.nn.functional.softplus(before["dt_bias"])
        self.assertTrue(bool(((steps >= 1e-4 - 1e-10) & (steps <= 1e-2 + 1e-8)).all()))

    def test_joint_c_changes_with_history_but_query_c_does_not(self):
        torch.manual_seed(113)
        memory, queries = torch.randn(1, 5, 8), torch.randn(1, 2, 8)
        changed = memory + torch.randn_like(memory) * 3
        for mode in ("query", "joint"):
            model = self.compressor(c_conditioning=mode)
            before = model.build_scan_inputs(memory, queries)
            after = model.build_scan_inputs(changed, queries)
            if mode == "query":
                torch.testing.assert_close(before["C"], after["C"], rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(before["C"], after["C"]))

    def test_state128_is_state_width_not_head_width_and_backpropagates(self):
        torch.manual_seed(117)
        model = self.compressor(summary_tokens=32, d_inner=64, head_dim=64,
                                d_state=128, chunk_size=64)
        memory = torch.randn(1, 5, 8, requires_grad=True)
        queries = torch.randn(1, 32, 8, requires_grad=True)
        inputs = model.build_scan_inputs(memory, queries)
        self.assertEqual(inputs["x"].shape, (32, 5, 1, 64))
        self.assertEqual(inputs["dt"].shape, (32, 5, 1))
        self.assertEqual(inputs["B"].shape, (32, 5, 1, 128))
        self.assertEqual(inputs["C"].shape, (32, 5, 1, 128))
        self.assertEqual(inputs["A"].shape, (1,))
        outputs = model.scan_output(inputs)
        self.assertEqual(outputs.shape, (32, 5, 1, 64))
        # An independent final-output formula sums event contributions after
        # decay; its state contraction must use all 128 B/C coordinates.
        steps = torch.nn.functional.softplus(inputs["dt"] + inputs["dt_bias"])
        final_c = inputs["C"][:, -1, 0]
        contributions = []
        for position in range(memory.shape[1]):
            later_time = steps[:, position + 1:, 0].sum(dim=1)
            decay = torch.exp(later_time * inputs["A"][0])
            similarity = (inputs["B"][:, position, 0] * final_c).sum(dim=-1)
            contributions.append(inputs["x"][:, position, 0]
                                 * (steps[:, position, 0] * decay * similarity)[:, None])
        expected = torch.stack(contributions).sum(dim=0)
        torch.testing.assert_close(outputs[:, -1, 0], expected, rtol=2e-5, atol=2e-7)
        (outputs[:, -1].square() * torch.linspace(.3, 1.7, 64)).sum().backward()
        self.assertGreater(float(memory.grad[:, 0].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(queries.grad).all()))
        self.assertGreater(float(queries.grad.abs().sum()), 0)
        for parameter in (model.write_projection.weight, model.read_projection.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertTrue(bool((parameter.grad.abs().sum(dim=1) > 0).all()))
        metadata = model.architecture_metadata()
        self.assertEqual(metadata["d_state"], 128)
        self.assertEqual(metadata["head_dim"], 64)
        self.assertEqual(metadata["scan_heads"], 1)

    def test_old_suffix_summaries_ignore_raw_tail_and_future_padding(self):
        torch.manual_seed(127)
        model = self.compressor()
        inputs = torch.randn(1, 12, 8, requires_grad=True)
        mask = torch.arange(12)[None, :] < 10
        outputs, output_mask, sources = model(inputs, mask)
        changed = inputs.detach().clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 20
        other = model(changed, mask)[0]
        torch.testing.assert_close(outputs[:, :2], other[:, :2], rtol=0, atol=0)
        self.assertEqual(sources.tolist(), [[6, 6, 7, 8, 9]])
        self.assertEqual(output_mask.sum().item(), 5)
        (outputs[:, :2].square() * torch.linspace(.3, 1.8, 8)).sum().backward()
        self.assertGreater(float(inputs.grad[:, 0].abs().sum()), 0)
        self.assertEqual(int(inputs.grad[:, 7:].count_nonzero()), 0)
        for name in ("A_log", "input_gate", "query_gate", "delta_projection",
                     "write_projection", "read_projection", "query_generator"):
            gradients = [value.grad for key, value in model.named_parameters()
                         if key == name or key.startswith(name + ".")]
            self.assertTrue(gradients, name)
            self.assertTrue(all(grad is not None and bool(torch.isfinite(grad).all())
                                for grad in gradients), name)
            self.assertGreater(sum(float(grad.abs().sum()) for grad in gradients), 0, name)

    def test_ragged_and_zero_old_history_preserve_original_tail(self):
        torch.manual_seed(131)
        model = self.compressor()
        lengths = torch.tensor([11, 7, 3, 1])
        inputs = torch.randn(4, 13, 8)
        mask = torch.arange(13)[None, :] < lengths[:, None]
        changed = inputs.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 100
        with torch.no_grad():
            output, output_mask, sources = model(inputs, mask)
            torch.testing.assert_close(output, model(changed, mask)[0], rtol=0, atol=0)
            expected_lengths = (lengths - 3).clamp(min=0, max=2) + lengths.clamp(max=3)
            self.assertEqual(output_mask.sum(dim=1).tolist(), expected_lengths.tolist())
            self.assertEqual(int(output[~output_mask].count_nonzero()), 0)
            for row, length in enumerate(lengths.tolist()):
                solo = model(inputs[row:row + 1, :length], mask[row:row + 1, :length])
                end = int(expected_lengths[row])
                torch.testing.assert_close(output[row, :end], solo[0][0], rtol=2e-5, atol=2e-6)
                self.assertEqual(sources[row, :end].tolist(), solo[2][0].tolist())
                tail = min(length, 3)
                torch.testing.assert_close(output[row, end - tail:end], inputs[row, length - tail:length])

    def test_factory_preserves_rng_contract_validation_and_cpu_native_guard(self):
        from rhpr.sasrec.query_ssm2 import audit_native_query_ssm2

        torch.manual_seed(137)
        before = torch.get_rng_state()
        self.compressor()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for kwargs in ({"head_dim": 3}, {"chunk_size": 3}, {"d_state": 0},
                       {"d_state": True}, {"d_state": 257},
                       {"c_conditioning": "invalid"}, {"dt_min": 0.},
                       {"dt_max": float("nan")}, {"head_dim": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compressor(**kwargs)
        model = self.compressor()
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            model.scan_output(model.build_scan_inputs(torch.randn(1, 2, 8),
                                                       torch.randn(1, 2, 8)), backend="native")
        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            audit_native_query_ssm2("cpu")
        for state in (0, True, 257, 128.0):
            with self.subTest(d_state=state), self.assertRaisesRegex(ValueError, "d_state"):
                # Validation occurs before any CUDA allocation or kernel call.
                audit_native_query_ssm2("cuda", head_dim=64, d_state=state)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_native_ssd_matches_cpu_and_final_output_gradients(self):
        from rhpr.sasrec.query_ssm2 import audit_native_query_ssm2, native_ssd_available

        if not native_ssd_available():
            self.skipTest("native SSD kernel is not installed")
        for head_dim, d_state in ((32, 16), (64, 16), (64, 128)):
            with self.subTest(head_dim=head_dim, d_state=d_state):
                result = audit_native_query_ssm2("cuda", head_dim=head_dim, d_state=d_state)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["d_state"], d_state)
                self.assertGreater(result["early_x_gradient_norm"], 0)
                self.assertTrue(result["multi_chunk_checked"])
                self.assertTrue(result["partial_chunk_checked"])
                self.assertEqual(result["chunk_count"], 2)
                self.assertEqual(result["length"], result["chunk_size"] + 1)


if __name__ == "__main__":
    unittest.main()
