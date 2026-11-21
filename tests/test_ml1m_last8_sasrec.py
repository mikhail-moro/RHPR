"""Causal multi-target SASRec must reproduce independently truncated prefixes."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'requires torch and numpy')
class MovieLensLast8SASRecTest(unittest.TestCase):
    def test_grouped_logits_match_each_prefix(self):
        import numpy as np
        import torch

        from rhpr.gr.movie_sasrec import Last8ItemIDIndex, build_movie_sasrec, grouped_sasrec_logits

        cfg = {'data': {'items': 40, 'history_items': 12},
               'sasrec': {'d_model': 8, 'n_layers': 1, 'n_heads': 2, 'ffn_multiplier': 2, 'dropout': 0.0}}
        index = Last8ItemIDIndex(cfg['data']['items'])
        model = build_movie_sasrec(cfg, 101, 'cpu', grouped=True).eval()
        histories = np.array([[-1, -1, 5, 3, 8], [10, 11, 12, 13, 14]])
        users = np.array([1, 2])
        targets = np.arange(16, 32).reshape(2, 8)
        with torch.inference_mode():
            batch = index.model_batch(histories, users, targets, 'cpu')
            grouped = grouped_sasrec_logits(model, batch)
            for step in range(targets.shape[1]):
                visible = np.concatenate((histories, targets[:, :step]), axis=1)
                prefix = np.pad(visible, ((0, 0), (12-visible.shape[1], 0)), constant_values=-1)
                separate = index.model_batch(prefix, users, targets[:, step], 'cpu')
                torch.testing.assert_close(grouped[:, step], model.catalogue_logits(separate))
