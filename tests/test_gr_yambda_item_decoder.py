from importlib.util import find_spec
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(find_spec('torch') and find_spec('transformers'), 'PyTorch and Transformers required')
class ItemDecoderCausalityTest(unittest.TestCase):
    def test_parallel_targets_match_their_individual_causal_histories(self):
        import torch
        from torch import nn
        from transformers import T5Config, __version__
        from transformers.models.t5.modeling_t5 import T5Stack

        if __version__ != '5.22.1':
            self.skipTest('Requires the experiment’s pinned T5 stack')
        from rhpr.gr.yambda_item_decoder import MODES, teacher_forced_logits

        config = T5Config(d_model=8, d_kv=4, d_ff=16, num_heads=2, num_layers=4,
                          is_decoder=True, is_encoder_decoder=False, dropout_rate=0.0,
                          attn_implementation='sdpa')
        stack = T5Stack(config).eval()
        codes = nn.Embedding(36, 8)
        model = SimpleNamespace(
            decoder=SimpleNamespace(decoder=stack, decoder_mlp=nn.ModuleList(nn.Linear(8, 9) for _ in range(4)),
                                    bos_token=torch.zeros(1, 8)),
            get_embedding_table=lambda _name: codes,
            num_embeddings_per_hierarchy=9, user_embedding=nn.Embedding(5, 8),
            item_sid_fusion=nn.Linear(16, 8),
        )
        history = torch.tensor([[-1, -1, -1, -1, 0, 1, 2, 3]])
        targets = torch.tensor([[[4, 5, 6, 7], [0, 7, 8, 1], [1, 3, 4, 2]]])
        grouped = dict(input_ids=history, attention_mask_encoder=history.ge(0),
                       user_id=torch.tensor([[3]]), future_ids=targets)
        with torch.inference_mode():
            for mode in MODES:
                with self.subTest(mode=mode):
                    model.item_history_mode = mode
                    group_logits = teacher_forced_logits(model, grouped)
                    for position in range(targets.shape[1]):
                        single_history = torch.cat((history, targets[:, :position].flatten(1)), dim=1)
                        single = {**grouped, 'input_ids': single_history,
                                  'attention_mask_encoder': single_history.ge(0),
                                  'future_ids': targets[:, position:position+1]}
                        for batched, individual in zip(group_logits, teacher_forced_logits(model, single), strict=True):
                            torch.testing.assert_close(batched[:, position], individual[:, 0])
