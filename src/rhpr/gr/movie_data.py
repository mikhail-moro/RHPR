"""MovieLens timestamp-group holdouts and strictly-past training contexts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TemporalUser:
    user_id: int
    train: tuple[int, ...]
    train_times: tuple[int, ...]
    validation: int
    validation_time: int
    test: int
    test_time: int
    test_between: tuple[int, ...] = ()


def choose_target(rows, user_id, timestamp, seed):
    """Deterministic choice independent of rating, file order and model outcome."""
    return min(rows, key=lambda r: hashlib.sha256(
        f'ml1m-holdout:{seed}:{user_id}:{timestamp}:{int(r[1])}'.encode()).digest())


def split_ratings(ratings, seed=42):
    ratings = np.asarray(ratings)
    if ratings.ndim != 2 or ratings.shape[1] != 4:
        raise ValueError('expected user,item,rating,timestamp rows')
    if len(np.unique(ratings[:, :2], axis=0)) != len(ratings):
        raise ValueError('duplicate user-movie observations')
    rows, exclusions, embargo = [], [], []
    order = np.lexsort((ratings[:, 1], ratings[:, 3], ratings[:, 0]))
    ordered = ratings[order]
    groups = np.split(ordered, np.flatnonzero(np.diff(ordered[:, 0])) + 1)
    for g in groups:
        uid = int(g[0, 0]); times = np.unique(g[:, 3])
        if len(times) < 3:
            exclusions.append({'user_id': uid, 'events': len(g), 'reason': 'fewer_than_three_timestamps'})
            continue
        vt, tt = map(int, times[-2:])
        vg, tg = g[g[:, 3] == vt], g[g[:, 3] == tt]
        val, test = choose_target(vg, uid, vt, seed), choose_target(tg, uid, tt, seed)
        train = g[g[:, 3] < vt]
        row = {'user_id': uid, 'train': train[:, [1, 3]].tolist(),
               'validation': [int(val[1]), vt], 'test': [int(test[1]), tt]}
        rows.append(row)
        for block, chosen, split in [(vg, val, 'validation'), (tg, test, 'test')]:
            for r in block:
                if r[1] != chosen[1]:
                    embargo.append({'user_id': uid, 'movie_id': int(r[1]), 'timestamp': int(r[3]), 'split': split})
    return rows, exclusions, embargo


def load_temporal_users(path, source_ids):
    item_map = {int(v): i for i, v in enumerate(source_ids)}
    result = []
    for row in json.loads(Path(path).read_text()):
        values, times = zip(*row['train'])
        vt, tt = row['validation'][1], row['test'][1]
        if not values or list(times) != sorted(times) or not max(times) < vt < tt:
            raise ValueError('invalid temporal holdout boundaries')
        between = row.get('test_between', [])
        if any(not vt < t < tt for i,t in between) or [t for i,t in between] != sorted(t for i,t in between):
            raise ValueError('invalid intervening test context')
        all_ids = [*values, row['validation'][0], row['test'][0], *[i for i,t in between]]
        if len(set(all_ids)) != len(all_ids) or any(v not in item_map for v in all_ids):
            raise ValueError('duplicate or unknown user movie ID')
        result.append(TemporalUser(int(row['user_id']), tuple(item_map[v] for v in values),
            tuple(times), item_map[row['validation'][0]], vt, item_map[row['test'][0]], tt, tuple(item_map[i] for i,t in between)))
    if len(set(u.user_id for u in result)) != len(result):
        raise ValueError('duplicate user ID')
    return result


def temporal_examples(users, history_items, split):
    if split not in ('train', 'validation', 'test') or type(history_items) is not int or history_items < 1:
        raise ValueError('invalid split/history length')
    def positions(user):
        first_end = next((i for i, t in enumerate(user.train_times) if t != user.train_times[0]), len(user.train))
        return range(first_end, len(user.train))
    n = sum(len(positions(u)) if split == 'train' else 1 for u in users)
    histories = np.full((n, history_items), -1, dtype=np.int64)
    ids = np.empty(n, dtype=np.int64); targets = np.empty(n, dtype=np.int64)
    at = 0
    for user in users:
        if split == 'train':
            starts = {}
            for i, t in enumerate(user.train_times): starts.setdefault(t, i)
            examples = ((user.train[max(0, starts[user.train_times[i]] - history_items):starts[user.train_times[i]]], user.train[i]) for i in positions(user))
        else:
            h = user.train if split == 'validation' else (*user.train, user.validation, *user.test_between)
            examples = [(h[-history_items:], user.validation if split == 'validation' else user.test)]
        for h, target in examples:
            if not h: raise ValueError('empty temporal training history')
            histories[at, -len(h):] = h; ids[at] = user.user_id; targets[at] = target; at += 1
    assert at == n
    return histories, ids, targets


def case_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()
