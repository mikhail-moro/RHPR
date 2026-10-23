"""CPU arithmetic parity for GRID's rank-local distributed RKMeans update."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import unittest


_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_REQUIRED = (
    "torch", "numpy", "transformers", "lightning", "pytorch_lightning",
    "torchmetrics", "pyarrow", "hydra",
)
_CAN_INTEGRATE = bool(_SOURCE) and all(
    importlib.util.find_spec(name) is not None for name in _REQUIRED
)


@unittest.skipUnless(
    _CAN_INTEGRATE,
    "Set RHPR_GRID_SOURCE and install GR dependencies for actual GRID integration",
)
class LogicalRankQuantizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from rhpr.gr.grid import load_grid
        from rhpr.gr.paper_quantizer_profile import logical_rank_step

        cls.torch = torch
        cls.grid = load_grid(Path(_SOURCE))
        cls.logical_rank_step = staticmethod(logical_rank_step)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def model(self, centroids, history, manual=False):
        model = self.grid.MiniBatchKMeans(
            n_clusters=centroids.shape[0],
            n_features=centroids.shape[1],
            distance_function=self.grid.SquaredEuclideanDistance(),
            update_manually=manual,
        )
        model.is_initialized = True
        with self.torch.no_grad():
            model.centroids.copy_(centroids)
        model.cluster_counts = history.clone()
        return model

    def run_step(self, replicas, batches, optimizer=None):
        if optimizer is None:
            optimizer = self.torch.optim.SGD(replicas[0].parameters(), lr=0.5)
        report = self.logical_rank_step(replicas, optimizer, batches)
        self.assertEqual(report["logical_ranks"], len(replicas))
        self.assertEqual(report["points"], sum(len(batch) for batch in batches))
        self.assertTrue(report["centroid_grad_finite"])
        self.assertTrue(math.isfinite(report["loss"]))
        return optimizer

    def assert_close(self, actual, expected):
        self.torch.testing.assert_close(actual, expected)

    def test_unequal_rank_history_matches_ddp_and_differs_from_pooled_batch(self):
        torch = self.torch
        centroids = torch.tensor([[0.0]])
        histories = [torch.tensor([1.0]), torch.tensor([9.0])]
        batches = [torch.tensor([[2.0]]), torch.tensor([[4.0]])]
        replicas = [self.model(centroids, history) for history in histories]
        self.run_step(replicas, batches)
        for replica in replicas:
            self.assert_close(replica.centroids, torch.tensor([[0.7]]))

        pooled = self.model(centroids, sum(histories), manual=True)
        pooled.model_step(torch.cat(batches))
        self.assert_close(pooled.centroids, torch.tensor([[0.5]]))
        self.assertGreater(float((replicas[0].centroids - pooled.centroids).detach().abs().max()), 0.19)

    def test_counts_remain_rank_local_and_persist_between_updates(self):
        torch = self.torch
        initial = torch.tensor([[0.0], [20.0]])
        replicas = [
            self.model(initial, torch.tensor([1.0, 9.0])),
            self.model(initial, torch.tensor([7.0, 2.0])),
        ]
        counts = [replica.cluster_counts for replica in replicas]
        batches = [torch.tensor([[1.0], [2.0]]), torch.tensor([[19.0]])]
        optimizer = self.run_step(replicas, batches)
        self.assert_close(replicas[0].cluster_counts, torch.tensor([3.0, 9.0]))
        self.assert_close(replicas[1].cluster_counts, torch.tensor([7.0, 3.0]))
        self.assertIsNot(replicas[0].cluster_counts, replicas[1].cluster_counts)
        for replica, local_counts in zip(replicas, counts):
            self.assertIs(replica.cluster_counts, local_counts)
            self.assertNotIn("cluster_counts", replica.state_dict())
            self.assertNotIn("cluster_counts", dict(replica.named_buffers()))

        self.run_step(replicas, batches, optimizer)
        self.assert_close(replicas[0].cluster_counts, torch.tensor([5.0, 9.0]))
        self.assert_close(replicas[1].cluster_counts, torch.tensor([7.0, 4.0]))
        self.assert_close(replicas[0].centroids, replicas[1].centroids)

    def test_every_rank_assigns_against_the_common_preupdate_centroids(self):
        torch = self.torch
        initial = torch.tensor([[0.0], [10.0]])
        replicas = [self.model(initial, torch.zeros(2)) for _ in range(2)]
        # A sequential manual update on rank 0 would move centroid 10 to 8.
        # Rank 1's item 4.5 would then choose it. DDP chooses centroid 0 before
        # either rank's update, and both deltas are averaged afterwards.
        batches = [torch.tensor([[8.0]]), torch.tensor([[4.5]])]
        self.run_step(replicas, batches)
        self.assert_close(replicas[0].cluster_counts, torch.tensor([0.0, 1.0]))
        self.assert_close(replicas[1].cluster_counts, torch.tensor([1.0, 0.0]))
        for replica in replicas:
            self.assert_close(replica.centroids, torch.tensor([[2.25], [9.0]]))

    def test_bounded_runtime_proxy_keeps_rng_and_includes_all_residual_stages(self):
        from rhpr.gr.paper_profile import PaperProfileBudget
        from rhpr.gr.paper_quantizer_profile import benchmark_logical_rank_quantizer

        torch = self.torch
        rng = torch.get_rng_state().clone()
        report = benchmark_logical_rank_quantizer(Path(_SOURCE), {
            "seed": 42,
            "quantizer": {"logical_ranks": 2, "batch_size_per_rank": 4,
                "embedding_dim": 8, "codebook_size": 4, "num_hierarchies": 3,
                "init_buffer_size": 8, "steps_per_level": 1000},
        }, {"quantizer_iterations": 2, "warmup": 1}, PaperProfileBudget(30), "cpu")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["total_assignment_stages_per_update"], 3)
        self.assertEqual(report["points_per_update"], 8)
        self.assertTrue(report["quality_not_evaluated"])
        self.assertTrue(report["matches_initialized_ddp_gradient_arithmetic"])
        self.assertFalse(report["matches_eight_gpu_wall_time"])
        self.assertGreater(report["seconds_per_logical_update"], 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_eight_rank_updates_match_actual_source_gradients_over_two_steps(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(42)
        initial = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
        histories = [
            torch.randint(1, 20, (3,), generator=generator).float()
            for _ in range(8)
        ]
        replicas = [self.model(initial, history) for history in histories]
        references = [self.model(initial, history) for history in histories]
        optimizer = torch.optim.SGD(replicas[0].parameters(), lr=0.5)
        expected = initial
        for _ in range(2):
            batches = [
                torch.randn(3 + rank, 2, generator=generator) * 3
                for rank in range(8)
            ]
            independent_gradients = []
            for reference, batch in zip(references, batches):
                reference.zero_grad(set_to_none=True)
                with torch.no_grad():
                    reference.centroids.copy_(expected)
                _, _, loss = reference.model_step(batch)
                loss.backward()
                independent_gradients.append(reference.centroids.grad.detach().clone())
            expected = expected - 0.5 * torch.stack(independent_gradients).mean(0)
            self.run_step(replicas, batches, optimizer)
            for replica, reference in zip(replicas, references):
                self.assert_close(replica.centroids, expected)
                self.assert_close(replica.cluster_counts, reference.cluster_counts)


if __name__ == "__main__":
    unittest.main()
