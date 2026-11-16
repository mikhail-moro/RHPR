"""Prepare the fixed random Steam cohort; no GPU, sampling on quality, or item pruning.

Sources are the SASRec authors' steam_reviews.json.gz, steam_games.json.gz and
Steam.txt. The optional cached columns contain encounter-order username IDs,
original game IDs and YYYYMMDD dates; their exact bytes are pinned below.
"""
import argparse
import array
import ast
import gzip
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.history_windows import prepare_cases
from rhpr.gr.kindle_cohort import core_mask

SOURCE_HASHES = {
    'steam_reviews.json.gz': '40c6343deedf488689987749c5f8435233da48c163fbaa2a7766c215d584bc61',
    'steam_games.json.gz': '717f4653a26585fec8f463f59311a654127c2ed66b07b9f78b4c0ce64961c740',
    'Steam.txt': '4df1f5feac169a522008956ba5a50fee4e9ce2c0c817cabfd463ac36df05f023',
}
COLUMNS_HASH = '2d9b0523ccadea3b3a2b49594083a02602589f2861a090eee8c4f6f768fda075'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def first_map(values):
    names, first = np.unique(values, return_index=True); names = names[np.argsort(first)]
    mapping = np.zeros(int(values.max())+1, dtype=np.int32)
    mapping[names] = np.arange(1, len(names)+1)
    return mapping[values], names


def load_columns(reviews, cache=None):
    if cache is not None:
        if file_sha256(cache) != COLUMNS_HASH: raise ValueError('unverified raw-column cache')
        with np.load(cache, allow_pickle=False) as z:
            return z['users'], z['items'], z['dates']
    names = {}; users = array.array('I'); items = array.array('I'); dates = array.array('I')
    with gzip.open(reviews, 'rt') as f:
        for line in f:
            r = ast.literal_eval(line)
            # repr(str) is a deterministic injective key, matching the pinned cache.
            name = repr(r['username'])
            if name not in names: names[name] = len(names)
            users.append(names[name]); items.append(int(r['product_id']))
            dates.append(int(r['date'].replace('-', '')))
    return tuple(np.asarray(v, dtype=np.int32) for v in (users, items, dates))


def prepare(raw, author_file, output, columns=None):
    raw, output = Path(raw), Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, path in [('steam_reviews.json.gz', raw/'steam_reviews.json.gz'),
                       ('steam_games.json.gz', raw/'steam_games.json.gz'), ('Steam.txt', Path(author_file))]:
        if file_sha256(path) != SOURCE_HASHES[name]: raise ValueError('source mismatch: '+name)
    u, i, dates = load_columns(raw/'steam_reviews.json.gz', columns)
    if (len(u), len(np.unique(u)), len(np.unique(i))) != (7793069, 2567538, 15474):
        raise ValueError('raw review counts differ')
    # Reconstruct the original one-pass raw count filter and first-encounter IDs.
    mask = (np.bincount(u)[u] >= 5) & (np.bincount(i)[i] >= 5)
    nu, _ = first_map(u[mask]); ni, original_items = first_map(i[mask]); dates = dates[mask]
    width = len(original_items)+1; keys = nu.astype(np.int64)*width+ni
    order = np.argsort(keys, kind='stable'); sk = keys[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1])+1]
    unique_keys = sk[starts]; first_dates = np.minimum.reduceat(dates[order], starts)
    author = np.fromfile(author_file, sep=' ', dtype=np.int64).reshape(-1, 2)
    author_keys = author[:, 0]*width+author[:, 1]
    if not np.array_equal(np.sort(author_keys), unique_keys):
        raise ValueError('original-game mapping fails exact all-user/item incidence replay')
    author_dates = first_dates[np.searchsorted(unique_keys, author_keys)]
    if np.any((author[1:, 0] == author[:-1, 0]) & (author_dates[1:] < author_dates[:-1])):
        raise ValueError('author order not chronological under earliest-review deduplication')
    del u, i, dates, nu, ni, keys, order, sk, mask
    mask, iterations = core_mask(author[:, 0], author[:, 1], min_events=5, min_item_users=5)
    author, author_dates = author[mask], author_dates[mask]
    users, lengths = np.unique(author[:, 0], return_counts=True)
    catalogue_ids = sorted(map(int, np.unique(author[:, 1])))
    if (len(users), len(catalogue_ids), len(author)) != (281210, 11961, 3484694):
        raise ValueError('global iterative5core differs')
    eligible = users[lengths >= 32]; chosen = eligible[np.random.default_rng(4242).permutation(len(eligible))[:2000]]
    metadata = {}
    with gzip.open(raw/'steam_games.json.gz', 'rt') as f:
        for line in f:
            r = ast.literal_eval(line)
            if r.get('id'): metadata[str(r['id'])] = r
    catalog = []; item_map = {v: j for j, v in enumerate(catalogue_ids)}
    for author_id in catalogue_ids:
        original_id = str(original_items[author_id-1]); r = metadata.get(original_id, {})
        title = ' '.join((r.get('title') or r.get('app_name') or '').split())
        if not title: raise ValueError('catalogue title missing; never silently drop events')
        genres = r.get('genres') or []
        catalog.append(dict(item_id=item_map[author_id], source_item_id=original_id, sasrec_item_id=author_id,
            title=title, genres=genres, text='Title: '+title+'. Genres: '+', '.join(genres)))
    ends = np.r_[np.flatnonzero(author[1:, 0] != author[:-1, 0])+1, len(author)]
    bounds = {int(author[a, 0]): (int(a), int(b)) for a, b in zip(np.r_[0, ends[:-1]], ends)}
    sequences = []; audits = []; train_items = set(); val = []; test = []
    for j, source_uid in enumerate(chosen, 1):
        a, b = bounds[int(source_uid)]; s = [item_map[int(v)] for v in author[a:b, 1]]; ts = author_dates[a:b]
        sequences.append(dict(user_id=j, train=s[:-2], validation=s[-2], test=s[-1]))
        audits.append(dict(user_id=j, source_sasrec_user_id=int(source_uid), events=len(s),
            train_last_day=int(ts[-3]), validation_day=int(ts[-2]), test_day=int(ts[-1])))
        train_items.update(s[:-2]); val.append(s[-2]); test.append(s[-1])
    write(output/'catalog.json', catalog); write(output/'sequences.json', sequences); write(output/'user-audit.json', audits)
    counts = dict(users=2000, items=11961, train_events=sum(len(v['train']) for v in sequences),
        train_examples=sum(len(v['train'])-1 for v in sequences))
    if counts['train_examples'] != 127023: raise ValueError('sample differs from candidate audit')
    _, report = prepare_cases(output, counts, output/'cases')
    report.update(events=sum(a['events'] for a in audits), history_items=30, data_seed=4242,
        minimum_full_history=32, global_core_iterations=iterations, within_sample_core=False,
        validation_cold_targets=sum(v not in train_items for v in val), test_cold_targets=sum(v not in train_items for v in test),
        missing_catalogue_titles=0, missing_catalogue_genres=sum(not v['genres'] for v in catalog),
        train_val_same_day=sum(v['train_last_day'] == v['validation_day'] for v in audits),
        val_test_same_day=sum(v['validation_day'] == v['test_day'] for v in audits),
        full_length_quantiles=np.quantile([v['events'] for v in audits], [0, .25, .5, .75, .9, .95, .99, 1]).tolist())
    write(output/'data-report.json', report)
    write(output/'source-provenance.json', dict(source_sha256=SOURCE_HASHES, metadata_mapping='exact all3,686,172 unique author user/item incidences',
        ordering='unchanged author order, verified nondecreasing earliest raw review day; arbitrary provided order within same day',
        raw_duplicate_rows_removed=4213117-3686172, source_core_uses_full_histories=True,
        text_fields=['title', 'genres'], metadata_missing_genres_retained=True,
        selection='global iterative5core;full_length>=32;uniform2000users seed4242;no within-sample item pruning', gpu_used=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--author-file', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--columns-cache', type=Path); a = p.parse_args()
    prepare(a.raw, a.author_file, a.output, a.columns_cache)
