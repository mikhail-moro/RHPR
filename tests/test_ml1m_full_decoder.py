"""Check causal training and cached generation on the pinned GRID T5 stack."""
import os
import unittest


class GroupedTrainingCasesTest(unittest.TestCase):
    def test_last16_preserves_last8_and_has_causal_prefixes_for_both_blocks(self):
        import numpy as np

        from rhpr.gr.ml1m_last8 import build_last8_cases
        from rhpr.gr.ml1m_window_tiger import group_training_cases

        histories = {3: np.arange(20), 7: np.arange(200, 360)}
        raw8 = build_last8_cases(histories)['train'][:3]
        raw16 = build_last8_cases(histories, targets_per_user=16)['train'][:3]
        old = group_training_cases(raw8)
        expanded = group_training_cases(raw16, targets_per_user=16)
        for a, b in zip(old, expanded):
            np.testing.assert_array_equal(a, b[1::2])
        np.testing.assert_array_equal(expanded[1], [3, 3, 7, 7])
        for prefix, user, targets in zip(*expanded):
            sequence = histories[user]
            first = np.flatnonzero(sequence == targets[0])[0]
            np.testing.assert_array_equal(prefix[prefix >= 0], sequence[max(0, first - 121):first])
            np.testing.assert_array_equal(targets, sequence[first:first + 8])
            self.assertLessEqual(first + 8, len(sequence) - 2)


@unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'), 'requires pinned GRID and modern Transformers')
class FullHistoryDecoderTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        from rhpr.gr.experiment import _model_batch
        from rhpr.gr.ml1m_full_decoder import build_full_decoder

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        source = os.environ['RHPR_GRID_SOURCE']
        cfg = dict(codebook_size=4, beam_width=3, d_model=16, d_ff=32, num_heads=2, d_kv=8,
                   encoder_layers=1, decoder_layers=1, num_user_bins=8, dropout=0.0, attn_implementation='sdpa')
        sids = np.array(list(np.ndindex(4, 4, 4, 4)), dtype=np.int64)
        histories = np.array([[1, 2, 3, 12], [-1, -1, 15, 16], [-1, -1, -1, -1]])
        targets = np.arange(32, 56, dtype=np.int64).reshape(3, 8)
        self.batch = _model_batch(histories, np.array([0, 1, 10]), targets, sids, 'cpu')
        self.model = build_full_decoder(sids, cfg, 101, source).eval()

    def test_grouped_teacher_forcing_matches_every_unpadded_prefix(self):
        import torch

        from rhpr.gr.ml1m_full_decoder import embed_prefix, full_decoder_loss, target_hidden

        model, batch = self.model, self.batch
        with torch.inference_mode():
            grouped = target_hidden(model, batch).reshape(3, 32, -1)
            for row in range(3):
                history = batch['input_ids'][row:row + 1]
                history = history[history >= 0].reshape(1, -1)
                wanted = batch['future_ids'][row:row + 1].flatten(1)
                for step in range(wanted.shape[1]):
                    codes = torch.cat((history, wanted[:, :step]), dim=1)
                    tape, mask = embed_prefix(model, codes, torch.ones_like(codes), batch['user_id'][row:row + 1])
                    hidden = model.decoder(sequence_embedding=tape, attention_mask=mask, encoder_output=None,
                                           encoder_attention_mask=None, past_key_values=None)
                    torch.testing.assert_close(grouped[row, step], hidden[0, -1])
        loss = full_decoder_loss(model, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(p.grad.isfinite().all(), name)

    def test_cached_beam_matches_full_prefix_recomputation(self):
        import torch

        from rhpr.gr.ml1m_full_decoder import embed_prefix

        model, batch = self.model, self.batch
        history, mask, users = (batch[key] for key in ('input_ids', 'attention_mask_encoder', 'user_id'))
        with torch.inference_mode():
            cached, cached_scores = model.generate(input_ids=history, attention_mask=mask, user_id=users)
            ids = scores = None
            for level, head in enumerate(model.decoder.decoder_mlp):
                codes, observed, who = history, mask, users
                if level:
                    beam = model.top_k_for_generation
                    prefix = ids.reshape(-1, level)
                    codes = torch.cat((history.repeat_interleave(beam, dim=0), prefix), dim=1)
                    observed = torch.cat((mask.repeat_interleave(beam, dim=0), torch.ones_like(prefix)), dim=1)
                    who = users.repeat_interleave(beam, dim=0)
                tape, valid = embed_prefix(model, codes, observed, who)
                hidden = model.decoder(sequence_embedding=tape, attention_mask=valid, encoder_output=None,
                                       encoder_attention_mask=None, past_key_values=None)
                ids, scores, _ = model._beam_search_one_step(head(hidden[:, -1]), ids, scores, None, level, len(history))
            torch.testing.assert_close(cached, ids, atol=0, rtol=0)
            torch.testing.assert_close(cached_scores, scores)
