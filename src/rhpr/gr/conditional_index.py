"""Known length branch: one GRID generator, two frozen item/SID mappings."""
import hashlib
from types import MethodType

import numpy as np

from rhpr.gr.sid import evaluate_ranked_sids


class ConditionalSidIndex:
    """Choose a complete mapping per available prefix, including target labels."""
    def __init__(self, short, long, threshold):
        if (type(threshold) is not int or threshold < 1 or short.width != long.width
                or short.sids.shape != long.sids.shape):
            raise ValueError('compatible SID indices and positive threshold required')
        # Both mappings must describe exactly the same complete path set.
        if set(short.sid_to_item) != set(long.sid_to_item):
            raise ValueError('conditional indices must preserve the same complete paths')
        self.indices = (short, long)
        self.threshold = threshold
        self.num_items, self.sid_length = short.sids.shape
        self.width = short.width
        self.sids = np.stack([short.sids, long.sids])
        self.sids.flags.writeable = False

    def branches(self, contexts):
        c = np.asarray(contexts)
        if (c.ndim != 2 or c.dtype.kind not in 'iu' or c.shape[1] < self.threshold
                or np.any(c < -1) or np.any(c >= self.num_items)):
            raise ValueError('valid observed prefixes with width covering the threshold required')
        return ((c >= 0).sum(1) >= self.threshold).astype(np.int64)

    def model_batch(self, contexts, users, targets, device):
        import torch
        c, targets = np.asarray(contexts), np.asarray(targets)
        branch = self.branches(c)
        if (targets.shape != (len(c),) or targets.dtype.kind not in 'iu'
                or np.any(targets < 0) or np.any(targets >= self.num_items)
                or np.asarray(users).shape != (len(c),)):
            raise ValueError('one valid target/user per context required')
        observed = c >= 0
        inputs = self.sids[branch[:, None], np.maximum(c, 0)].copy()
        inputs[~observed] = -1
        # First raw column transports a branch ID. The model adapter strips it
        # BEFORE applying native hierarchy offsets and inserts its own embedding.
        raw = np.c_[branch, inputs.reshape(len(c), -1)]
        mask = np.c_[np.ones(len(c), dtype=np.int64), np.repeat(observed, self.sid_length, axis=1)]
        return dict(input_ids=torch.as_tensor(raw, dtype=torch.long, device=device),
                    attention_mask_encoder=torch.as_tensor(mask, dtype=torch.long, device=device),
                    user_id=torch.as_tensor(np.asarray(users)[:, None], dtype=torch.long, device=device),
                    future_ids=torch.as_tensor(self.sids[branch, targets], dtype=torch.long, device=device))

    def prediction_metadata(self, contexts):
        return dict(branch_ids=self.branches(contexts), observed_context_lengths=(contexts >= 0).sum(1))

    def evaluate(self, predictions, targets, contexts, cutoffs=(5, 10)):
        branches = self.branches(contexts)
        parts = []
        for branch, index in enumerate(self.indices):
            selected = branches == branch
            if selected.any():
                parts.append((branch, evaluate_ranked_sids(index, predictions[selected], targets[selected], cutoffs)))
        counts = ['evaluation_examples', 'prediction_count', 'invalid_predictions', 'duplicate_predictions']
        result = {key: sum(p[key] for _, p in parts) for key in counts}
        for name in ['invalid', 'duplicate']:
            result[name + '_prediction_fraction'] = result[name + '_predictions'] / max(result['prediction_count'], 1)
        for k in cutoffs:
            for metric in ['recall', 'ndcg']:
                key = f'{metric}@{k}'
                result[key] = sum(p[key] * p['evaluation_examples'] for _, p in parts) / result['evaluation_examples']
        for branch, label in [(0, 'short'), (1, 'long')]:
            part = next((p for b, p in parts if b == branch), None)
            result['gate_' + label + '_examples'] = 0 if part is None else part['evaluation_examples']
            if part is not None and 10 in cutoffs:
                result['gate_' + label + '_ndcg10'] = part['ndcg@10']
        return result

    def provenance(self):
        return dict(threshold=self.threshold, branch_token='encoder_prefix_before_user',
                    branch_names=['SHORT', 'LONG'], target_mapping='same branch as entire observed context',
                    sid_matrix_shape=list(self.sids.shape),
                    sid_matrix_sha256=hashlib.sha256(self.sids.tobytes()).hexdigest(),
                    changed_item_mappings=int(np.any(self.sids[0] != self.sids[1], axis=1).sum()))


def _encoder_with_branch(self, attention_mask, input_ids, user_id):
    """Native GRID embedding preparation plus one known branch prefix token.

    Item hierarchy offsets apply ONLY to item SIDcolumns. The original user
    hash, padding, item embeddings and decoder/beam implementation are retained.
    """
    import torch
    if input_ids.shape[1] % self.num_hierarchies != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError('expected branch column followed by complete item SIDs')
    branch = input_ids[:, 0]
    if torch.any((branch < 0) | (branch > 1)) or torch.any(attention_mask[:, 0] != 1):
        raise ValueError('branch token must be SHORT/LONG and observed')
    items, mask = input_ids[:, 1:], attention_mask[:, 1:]
    shifted = self._add_repeating_offset_to_rows(items, self.num_embeddings_per_hierarchy,
                                                self.num_hierarchies, mask)
    embeds = self.get_embedding_table(table_name='encoder')(shifted)
    if user_id is not None and self.user_embedding is not None:
        user = self.user_embedding(torch.remainder(user_id[:, 0], self.user_embedding.num_embeddings))
        embeds = torch.cat([user[:, None], embeds], dim=1)
        mask = torch.cat([torch.ones((len(items), 1), device=mask.device, dtype=mask.dtype), mask], dim=1)
    embeds = torch.cat([self.index_branch_embedding(branch)[:, None], embeds], dim=1)
    mask = torch.cat([torch.ones((len(items), 1), device=mask.device, dtype=mask.dtype), mask], dim=1)
    return self.encoder(sequence_embedding=embeds, attention_mask=mask), mask


def add_branch_token(model):
    """Register two embeddings; use the same adapter for control and switched SID."""
    import torch
    if model.sep_token is not None or model.should_check_prefix:
        raise ValueError('conditional adapter requires the no-SEP/free-beam GRID protocol')
    if hasattr(model, 'index_branch_embedding'):
        raise ValueError('branch token was already installed')
    device = model.get_embedding_table('encoder').weight.device
    model.index_branch_embedding = torch.nn.Embedding(2, model.embedding_dim).to(device)
    model.encoder_forward_pass = MethodType(_encoder_with_branch, model)
    model.rhpr_grid_provenance = {**model.rhpr_grid_provenance,
                                 'index_branch_token': 'two learned encoder prefix embeddings before user;known,not predicted',
                                 'extra_parameters': 2 * model.embedding_dim}
    return model
