"""Numerical and memory-contract checks for the optional GRID distance kernel."""
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rhpr.gr.distances import (
    TorchCdistSquaredEuclideanDistance,
    distance_backend_report,
    make_grid_distance,
)


_HAS_TORCH = importlib.util.find_spec("torch") is not None


class GridDistanceConfigurationTests(unittest.TestCase):
    def test_default_preserves_upstream_object(self):
        upstream = object()
        grid = SimpleNamespace(SquaredEuclideanDistance=lambda: upstream)
        self.assertIs(make_grid_distance(grid, {}), upstream)
        self.assertEqual(distance_backend_report({}), {
            "distance_backend": "upstream",
            "distance_compute_mode": "upstream_broadcast_squared_l2",
            "initialization_distance_backend": "upstream",
        })

    def test_optional_backend_and_provenance_are_explicit(self):
        config = {"distance_backend": "torch_cdist"}
        self.assertIsInstance(make_grid_distance(None, config), TorchCdistSquaredEuclideanDistance)
        self.assertEqual(distance_backend_report(config), {
            "distance_backend": "torch_cdist",
            "distance_compute_mode": "donot_use_mm_for_euclid_dist",
            "initialization_distance_backend": "upstream",
        })

    def test_unknown_backend_fails_instead_of_silently_falling_back(self):
        for backend in ("faiss", "", None):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "distance_backend"):
                make_grid_distance(None, {"distance_backend": backend})


@unittest.skipUnless(_HAS_TORCH, "PyTorch is not installed")
class TorchCdistDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.distance = TorchCdistSquaredEuclideanDistance()

    def assert_matches_explicit_metric(self, x, y, *, batch_size=256):
        torch = self.torch
        expected = (x[:, None, :] - y[None, :, :]).square().sum(dim=-1)
        actual = self.distance.compute(x, y, batch_size=batch_size)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-6)
        self.assertTrue(torch.equal(actual.argmin(dim=-1), expected.argmin(dim=-1)))
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.device, x.device)
        return actual

    def test_normalized_fp32_and_fp64_centroid_assignments(self):
        torch = self.torch
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(42)
                x = torch.nn.functional.normalize(torch.randn(19, 2048, dtype=dtype, generator=generator), dim=-1)
                y = torch.nn.functional.normalize(torch.randn(9, 2048, dtype=dtype, generator=generator), dim=-1)
                self.assert_matches_explicit_metric(x, y, batch_size=7)

    def test_duplicate_and_zero_vectors_keep_exact_zero_and_first_tie(self):
        torch = self.torch
        x = torch.tensor([[0., 0.], [1., 2.], [1., 2.], [3., 4.]])
        y = torch.tensor([[1., 2.], [0., 0.], [1., 2.]])
        actual = self.assert_matches_explicit_metric(x, y, batch_size=2)
        self.assertEqual(actual[0, 1].item(), 0.0)
        self.assertEqual(actual[1, 0].item(), 0.0)
        self.assertEqual(actual[1, 2].item(), 0.0)
        self.assertEqual(actual[1].argmin().item(), 0)
        identical = torch.ones(5, 2048)
        self.assertTrue(torch.equal(self.distance.compute(identical, identical), torch.zeros(5, 5)))

    def test_noncontiguous_views(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(7, 19, generator=generator).t()
        y = torch.randn(9, 14, generator=generator)[:, ::2]
        self.assertFalse(x.is_contiguous())
        self.assertFalse(y.is_contiguous())
        self.assert_matches_explicit_metric(x, y, batch_size=5)

    def test_large_common_offset_preserves_small_distances(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(43)
        x = 1000 + 1e-3 * torch.randn(32, 2048, generator=generator)
        actual = self.assert_matches_explicit_metric(x, x, batch_size=9)
        self.assertTrue(torch.equal(actual.diag(), torch.zeros(32)))
        self.assertTrue(torch.equal(actual.argmin(dim=-1), torch.arange(32)))

    def test_row_chunking_limits_native_calls_and_preserves_all_rows(self):
        torch = self.torch
        x = torch.arange(33, dtype=torch.float32).reshape(11, 3)
        y = torch.tensor([[0., 1., 2.], [27., 28., 29.]])
        with patch.object(torch, "cdist", wraps=torch.cdist) as cdist:
            chunked = self.assert_matches_explicit_metric(x, y, batch_size=4)
        self.assertEqual([call.args[0].shape[0] for call in cdist.call_args_list], [4, 4, 3])
        self.assertTrue(all(
            call.kwargs["compute_mode"] == "donot_use_mm_for_euclid_dist"
            for call in cdist.call_args_list
        ))
        torch.testing.assert_close(chunked, self.distance.compute(x, y, batch_size=None))

    def test_empty_rows_keep_the_distance_matrix_shape(self):
        torch = self.torch
        self.assertEqual(tuple(self.distance.compute(torch.empty(0, 3), torch.empty(4, 3)).shape), (0, 4))
        self.assertEqual(tuple(self.distance.compute(torch.empty(4, 3), torch.empty(0, 3)).shape), (4, 0))

    def test_half_precision_and_invalid_row_chunks_are_rejected(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float32 or float64"):
                x = torch.zeros(2, 3, dtype=dtype)
                self.distance.compute(x, x)
        x = torch.zeros(2, 3)
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, "batch_size"):
                self.distance.compute(x, x, batch_size=batch_size)

    @unittest.skipUnless(
        os.environ.get("RHPR_TEST_CUDA_DISTANCES") == "1",
        "Set RHPR_TEST_CUDA_DISTANCES=1 for an explicitly requested CUDA check",
    )
    def test_optional_cuda_normalized_and_shifted_assignments(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        generator = torch.Generator().manual_seed(91)
        x = torch.randn(32, 2048, generator=generator)
        for values in (torch.nn.functional.normalize(x, dim=-1), 1000 + 1e-3 * x):
            values = values.cuda()
            self.assert_matches_explicit_metric(values, values, batch_size=7)


if __name__ == "__main__":
    unittest.main()
