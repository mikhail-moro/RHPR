"""Build shared Last-16 windows and a fixed audio SID catalogue from Yambda-50M."""
import hashlib
from pathlib import Path

import numpy as np

from rhpr.gr.codebook_diagnostics import semantic_statistics
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex


def validate_pilot(config):
    seeds = config.get('generator_seeds', [config['seed']])
    expected = {(name, window, seed) for name, window in (('full_decoder', None), ('window16', 16))
                for seed in seeds}
    trials = [(trial['predictor'], trial['decoder_window'], trial['seed']) for trial in config['trials']]
    if (not seeds or len(seeds) != len(set(seeds)) or len(trials) != len(expected) or set(trials) != expected
            or config['data']['history_items'] not in (128, 256)
            or config['targets_per_user'] != 16 or not config.get('compare_initialization_by_shape')):
        raise ValueError('Expected paired Last-16 D4 and Window16 fits for every generator seed')


def verify_embedding_footer(path, spec):
    with path.open('rb') as stream:
        if stream.seek(0, 2) != spec['embedding_bytes']:
            raise ValueError('Unexpected Yambda embedding file size')
        stream.seek(-8, 2)
        tail = stream.read()
        footer_bytes = int.from_bytes(tail[:4], 'little')
        if tail[4:] != b'PAR1' or not 0 < footer_bytes < 1_000_000:
            raise ValueError('Invalid Yambda embedding Parquet footer')
        stream.seek(-8-footer_bytes, 2)
        if hashlib.sha256(stream.read()).hexdigest() != spec['embedding_footer_sha256']:
            raise ValueError('Yambda source embedding footer changed')


def read_audio_catalogue(parquet, observed_items):
    import pyarrow.compute as pc

    embedding_ids = parquet.read(columns=['item_id'])['item_id'].combine_chunks().to_numpy()
    if np.any(embedding_ids[1:] <= embedding_ids[:-1]):
        raise ValueError('Yambda embedding ids must be sorted and unique')
    source_ids = np.intersect1d(observed_items, embedding_ids, assume_unique=True)
    values = np.empty((len(source_ids), 128), dtype=np.float32)
    cursor = 0
    for batch in parquet.iter_batches(batch_size=16_384, columns=['item_id', 'normalized_embed']):
        ids = batch.column(0).to_numpy()
        right = np.searchsorted(source_ids, ids[-1], side='right')
        if right == cursor:
            continue
        selected = source_ids[cursor:right]
        indices = np.searchsorted(ids, selected)
        np.testing.assert_array_equal(ids[indices], selected)
        vectors = pc.take(batch.column(1), indices)
        if vectors.null_count or not np.all(pc.list_value_length(vectors).to_numpy() == 128):
            raise ValueError('Expected one complete 128-dimensional vector per selected track')
        part = vectors.flatten().to_numpy().reshape(-1, 128)
        if not np.isfinite(part).all():
            raise ValueError('Nonfinite Yambda embedding')
        values[cursor:right] = part
        cursor = right
    if cursor != len(source_ids):
        raise ValueError('Yambda embedding projection omitted catalogue items')
    return source_ids, values


def build_cases(users, offsets, items, timestamps, source_ids, history_items):
    """Hold out distinct time groups and keep two complete causal blocks per user."""
    dense = np.full(int(items.max()) + 1, -1, dtype=np.int32)
    dense[source_ids] = np.arange(len(source_ids), dtype=np.int32)
    training_prefix = np.full((2 * len(users), history_items - 7), -1, dtype=np.int32)
    training_targets = np.empty((2 * len(users), 8), dtype=np.int32)
    validation = np.full((len(users), history_items), -1, dtype=np.int32)
    test = np.full_like(validation, -1)
    kept_users = np.empty(len(users), dtype=np.int64)
    targets = np.empty((len(users), 2), dtype=np.int32)
    time_anchors = np.empty((len(users), 3), dtype=np.uint32)
    lengths = np.empty(len(users), dtype=np.int64)

    kept = 0
    for uid, start, end in zip(users, offsets[:-1], offsets[1:], strict=True):
        ids = dense[items[start:end]]
        has_embedding = ids >= 0
        ids = ids[has_embedding]
        times = timestamps[start:end][has_embedding]
        if len(ids) < 19:
            continue
        if np.any(times[1:] < times[:-1]):
            raise ValueError('Yambda source histories must be chronological')
        val_pos = int(np.searchsorted(times, times[-1])) - 1
        if val_pos < 0:
            continue
        cut = int(np.searchsorted(times, times[val_pos]))
        if cut < 17:
            continue
        train = ids[:cut]
        for block, end_pos in enumerate((cut - 8, cut)):
            row = 2 * kept + block
            start_pos = end_pos - 8
            prefix = train[max(0, start_pos - training_prefix.shape[1]):start_pos]
            training_prefix[row, -len(prefix):] = prefix
            training_targets[row] = train[start_pos:end_pos]
        val_context = train[-history_items:]
        validation[kept, -len(val_context):] = val_context
        test_context = train[-(history_items - 1):]
        test[kept, -(len(test_context)+1):-1] = test_context
        test[kept, -1] = ids[val_pos]
        targets[kept] = ids[[val_pos, -1]]
        time_anchors[kept] = times[[cut - 1, val_pos, -1]]
        lengths[kept] = len(train) + 1
        kept_users[kept] = uid
        kept += 1

    kept_users = kept_users[:kept]
    splits = {
        'train': (training_prefix[:2*kept], np.repeat(kept_users, 2), training_targets[:2*kept]),
        'validation': (validation[:kept], kept_users, targets[:kept, 0]),
        'test': (test[:kept], kept_users, targets[:kept, 1]),
    }
    return splits, time_anchors[:kept], lengths[:kept]


def _normalized_rows(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def _nearest_available(vectors, centers, capacity):
    if len(vectors) > capacity.sum():
        raise ValueError('Insufficient SID prefix capacity')
    center_norm = np.square(centers).sum(axis=1)
    codes = np.empty(len(vectors), dtype=np.int64)
    for start in range(0, len(vectors), 4096):
        batch = vectors[start:start+4096]
        # The squared input norm is common to all candidates for an item.
        preference = np.argsort(center_norm - 2 * (batch @ centers.T), axis=1, kind='stable')
        for i, ranked in enumerate(preference, start):
            code = ranked[np.flatnonzero(capacity[ranked])[0]]
            codes[i] = code
            capacity[code] -= 1
    return codes


def _by_code(rows, codes):
    for code in np.unique(codes[rows]):
        yield code, rows[codes[rows] == code]


def reassign_overflow(codes, vectors, centers, width):
    """Retain canonical members and route only excess tracks to free siblings."""
    # A first-level branch has width**2 triples, each with width suffixes.
    # The Yambda catalogue fits into a single branch, so its first code can
    # always be preserved, even for a completely degenerate audio cluster.
    if len(codes) > width**3:
        raise ValueError('Catalogue too large to guarantee preserving first-level SID assignments')
    codes = codes.astype(np.int64)
    packed = (codes[:, 0] * width + codes[:, 1]) * width + codes[:, 2]
    order = np.argsort(packed, kind='stable')
    keys, starts, counts = np.unique(packed[order], return_index=True, return_counts=True)
    overflow = counts > width
    if not overflow.any():
        return codes

    # Stable catalogue order picks the original members that keep a full triple.
    rows = np.concatenate([order[start+width:start+count]
                           for start, count in zip(starts[overflow], counts[overflow], strict=True)])
    capacity = np.full((width, width, width), width, dtype=np.int32)
    capacity.flat[keys] -= np.minimum(counts, width)
    repaired = codes.copy()
    for first, first_rows in _by_code(rows, codes[:, 0]):
        # Reserve second-level capacity for tracks keeping their original two
        # codes before offering leftover capacity to any displaced tracks.
        remaining = capacity[first].sum(axis=1)
        displaced = []
        for second, second_rows in _by_code(first_rows, codes[:, 1]):
            keep = min(len(second_rows), remaining[second])
            remaining[second] -= keep
            if keep < len(second_rows):
                displaced.append(second_rows[keep:])
        if displaced:
            spill = np.concatenate(displaced)
            residual = _normalized_rows(_normalized_rows(vectors[spill]) - centers[0, first])
            repaired[spill, 1] = _nearest_available(residual, centers[1], remaining)
        for second, second_rows in _by_code(first_rows, repaired[:, 1]):
            residual = _normalized_rows(_normalized_rows(vectors[second_rows]) - centers[0, first])
            residual = _normalized_rows(residual - centers[1, second])
            repaired[second_rows, 2] = _nearest_available(residual, centers[2], capacity[first, second])
    return repaired


def prepare(config, output, grid_source):
    import pyarrow.parquet as pq
    import torch

    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.offline_runtime import resolve_dataset_directory

    spec = config['data']
    root = resolve_dataset_directory(spec['path'])
    listens = root / 'sequential/50m/listens.parquet'
    embeddings = root / 'embeddings.parquet'
    if file_sha256(listens) != spec['listens_sha256']:
        raise ValueError('Yambda source listens changed')
    verify_embedding_footer(embeddings, spec)
    table = pq.read_table(listens, columns=['uid', 'item_id', 'timestamp'])
    users, item_list, time_list = (table[name].combine_chunks() for name in table.column_names)
    users = users.to_numpy()
    offsets = item_list.offsets.to_numpy()
    np.testing.assert_array_equal(offsets, time_list.offsets.to_numpy())
    items, timestamps = item_list.values.to_numpy(), time_list.values.to_numpy()
    if len(items) != spec['events'] or len(np.unique(users)) != len(users):
        raise ValueError('Unexpected Yambda source event count or repeated user')
    _event('yambda_embedding_projection_start', events=len(items), users=len(users))
    with pq.ParquetFile(embeddings) as parquet:
        source_ids, vectors = read_audio_catalogue(parquet, np.unique(items))
    # Source IDs are multiples of 100. Use catalogue order before GRID's modulo
    # hash so that the 2,000 user embeddings can all receive training updates.
    dense_users = np.arange(len(users), dtype=np.int64)
    splits, anchors, lengths = build_cases(dense_users, offsets, items, timestamps, source_ids, spec['history_items'])
    if (len(source_ids), len(anchors), splits['train'][2].size) != (spec['items'], spec['users'], spec['train_examples']):
        raise ValueError('Yambda catalogue or Last-16 split changed')
    cases = output / 'prepared-cases'
    cases.mkdir()
    digests = {name: case_digest(arrays) for name, arrays in splits.items()}
    if digests != config['expected_raw_case_sha256']:
        raise ValueError('Yambda cases differ from the pinned preflight split')
    for name, arrays in splits.items():
        np.savez(cases / f'{name}-cases.npz', **dict(zip(('contexts', 'user_ids', 'targets'), arrays)))
    eval_ids = splits['test'][1]
    np.savez(cases / 'anchors.npz', user_ids=eval_ids, source_user_ids=users[eval_ids],
             prior_train_timestamp=anchors[:, 0], validation_timestamp=anchors[:, 1],
             test_timestamp=anchors[:, 2], test_history_lengths=lengths)
    np.save(cases / 'source-item-ids.npy', source_ids, allow_pickle=False)
    _event('yambda_last16_cases_prepared', items=len(source_ids), users=len(anchors), train_examples=spec['train_examples'])
    codes, centers, report = fit_grid_rkmeans(vectors, config['quantizer'], config['quantizer_seed'],
                                              'cuda:0', grid_source)
    torch.cuda.empty_cache()
    width = config['generator']['codebook_size']
    raw_statistics = semantic_statistics(codes, width)
    raw_prefixes = raw_statistics['prefixes'][-1]
    _write(output / 'yambda-raw-collision-report.json', raw_statistics)
    _event('yambda_raw_sid_collisions', max_collision_group=raw_statistics['max_collision_group'],
           groups_exceeding_suffix_width=raw_prefixes['groups_exceeding_suffix_width'])
    repaired = reassign_overflow(codes, vectors, centers, width)
    moved = np.any(repaired != codes, axis=1)
    resolution = dict(moved_tracks=int(moved.sum()),
                      second_code_changes=int((repaired[:, 1] != codes[:, 1]).sum()),
                      moved_targets={name: int(moved[arrays[2]].sum()) for name, arrays in splits.items()},
                      policy='Preserve first code. Keep up to suffix width members per triple by catalogue row order; '
                             'route excess to nearest available third or second/third centroids.')
    index = SidIndex(repaired, width=width)
    diagnostics = index.diagnostics()
    _event('yambda_sid_index_ready', max_collision_group=diagnostics['max_collision_group'], **resolution)
    indexes = output / 'frozen-indexes'
    indexes.mkdir()
    np.save(indexes / 'items-42.npy', index.sids, allow_pickle=False)
    np.save(indexes / 'centers.npy', centers, allow_pickle=False)
    _write(output / 'yambda-data-report.json', dict(case_digests=digests,
        index_sha256=file_sha256(indexes / 'items-42.npy'), codebooks_sha256=file_sha256(indexes / 'centers.npy'),
        catalogue_items=len(source_ids), users=len(anchors), train_targets=spec['train_examples'],
        user_embedding_bins_used=int(len(np.unique(eval_ids % config['generator']['num_user_bins']))),
        users_with_full_test_window=int((lengths >= spec['history_items']).sum()),
        quantizer=report, collision_resolution=resolution, sid_diagnostics=diagnostics))
