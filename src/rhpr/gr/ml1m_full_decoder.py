"""GRID's SID heads over one causal stream of observed items and target codes."""
from functools import partial

from rhpr.gr.ml1m_window_tiger import SID_LEVELS, TARGETS_PER_USER


def build_full_decoder(sids, config, seed, source):
    from rhpr.gr.grid import build_grid_generator

    model = build_grid_generator(sids, config, seed, 'cpu', source)
    # GRID also registers its original T5 encoder under the base module.
    del model.model
    del model.encoder
    # Pinned T5Stack uses block.layer[0] for self-attention and [-1] for FFN.
    # The cross-attention branch is only taken when encoder memory is supplied.
    for block in model.decoder.decoder.block:
        del block.layer[1]
    model.generate = partial(generate_next, model)
    return model


def embed_prefix(model, codes, observed, user_id):
    import torch

    shifted = model._add_repeating_offset_to_rows(
        codes, model.num_embeddings_per_hierarchy, SID_LEVELS, observed)
    embedding = model.get_embedding_table('decoder')(shifted)
    user = model.user_embedding(user_id.reshape(-1) % model.user_embedding.num_embeddings)
    bos = model.decoder.bos_token.expand(len(codes), -1)
    prefix = torch.stack((user, bos), dim=1)

    # Insert USER/BOS just before the first observed SID. Keeping padding
    # before both tokens preserves their relative distances on ragged histories.
    prefix_slots = (observed == 0).sum(dim=1, keepdim=True) + torch.arange(2, device=codes.device)
    tape = torch.cat((torch.zeros_like(prefix), embedding), dim=1)
    tape = tape.scatter(1, prefix_slots.unsqueeze(-1).expand_as(prefix), prefix)
    mask = torch.cat((torch.zeros_like(prefix_slots), observed), dim=1).scatter(1, prefix_slots, 1)
    return tape, mask


def target_hidden(model, batch):
    import torch

    labels = batch['future_ids'].flatten(1)
    shifted_targets = labels[:, :-1]
    codes = torch.cat((batch['input_ids'], shifted_targets), dim=1)
    observed = torch.cat((batch['attention_mask_encoder'], torch.ones_like(shifted_targets)), dim=1)
    tape, mask = embed_prefix(model, codes, observed, batch['user_id'])
    hidden = model.decoder(sequence_embedding=tape, attention_mask=mask,
                           encoder_output=None, encoder_attention_mask=None, past_key_values=None)
    return hidden[:, -TARGETS_PER_USER * SID_LEVELS:].reshape(-1, SID_LEVELS, hidden.shape[-1])


def full_decoder_loss(model, batch):
    import torch.nn.functional as F

    hidden = target_hidden(model, batch)
    labels = batch['future_ids'].reshape(-1, SID_LEVELS)
    return sum(F.cross_entropy(head(hidden[:, level]), labels[:, level])
               for level, head in enumerate(model.decoder.decoder_mlp))


def generate_next(model, attention_mask, input_ids, user_id):
    import torch

    tape, mask = embed_prefix(model, input_ids, attention_mask, user_id)
    hidden, cache = model.decoder(sequence_embedding=tape, attention_mask=mask,
                                  encoder_output=None, encoder_attention_mask=None,
                                  use_cache=True, past_key_values=None)
    batch_size = len(input_ids)
    beam = model.top_k_for_generation
    ids, scores, _ = model._beam_search_one_step(
        model.decoder.decoder_mlp[0](hidden[:, -1]), None, None, None, 0, batch_size)
    cache.reorder_cache(torch.arange(batch_size, device=input_ids.device).repeat_interleave(beam))
    mask = mask.repeat_interleave(beam, dim=0)
    embedding = model.get_embedding_table('decoder')
    for level in range(1, SID_LEVELS):
        last_code = ids[:, :, -1].reshape(-1, 1) + (level - 1) * model.num_embeddings_per_hierarchy
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        hidden, cache = model.decoder(sequence_embedding=embedding(last_code), attention_mask=mask,
                                      encoder_output=None, encoder_attention_mask=None,
                                      use_cache=True, past_key_values=cache)
        ids, scores, cache = model._beam_search_one_step(
            model.decoder.decoder_mlp[level](hidden[:, -1]), ids, scores, cache, level, batch_size)
    return ids, scores
