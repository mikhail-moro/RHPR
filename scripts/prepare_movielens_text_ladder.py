"""Prepare cumulative movie texts and audit token lengths, without training."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

FIELDS = ('title', 'genres', 'year', 'runtime', 'keywords', 'cast', 'directors')
LABELS = ('Title', 'Genres', 'Year', 'Runtime', 'Keywords', 'Cast', 'Directors')


def read_csv(root, name):
    plain = root / name
    if plain.exists():
        with plain.open(encoding='utf-8', newline='') as f:
            yield from csv.DictReader(f)
    else:
        with zipfile.ZipFile(str(plain) + '.zip') as z:
            with z.open(name) as b, io.TextIOWrapper(b, encoding='utf-8', newline='') as f:
                yield from csv.DictReader(f)


def present(value):
    return bool(value and value.strip() and value.strip().lower() not in ('nan', 'none', 'null'))


def named(entries):
    """Keep source ranking and identity; do not truncate cast or duplicate people."""
    seen, result = set(), []
    for value in entries:
        key = value['id']
        if key not in seen and value.get('name', '').strip():
            seen.add(key)
            result.append(value['name'].strip())
    return result


def field_texts(fields):
    values = []
    for key in FIELDS:
        value = fields[key]
        values.append(', '.join(value) if isinstance(value, list) else str(value))
    parts = [f'{label}: {value}' for label, value in zip(LABELS, values)]
    return ['; '.join(parts[:i + 1]) + '.' for i in range(len(parts))]


def audit_timestamps(ratings):
    last_ties = same_last_two = extra = insufficient = 0
    duplicate = len(ratings) - len(np.unique(ratings[:, :2], axis=0))
    group_train_events = 0
    first_group_targets = 0
    for uid in np.unique(ratings[:, 0]):
        rows = ratings[ratings[:, 0] == uid]
        _, lengths = np.unique(rows[:, 3], return_counts=True)
        last_ties += int(lengths[-1] > 1)
        same_last_two += int(lengths[-1] > 1)
        insufficient += int(len(lengths) < 3)
        if len(lengths) >= 3:
            extra += int(lengths[-1] + lengths[-2] - 2)
            group_train_events += int(len(rows) - lengths[-1] - lengths[-2])
            first_group_targets += int(lengths[0])
    return {'users': int(len(np.unique(ratings[:, 0]))), 'ratings': len(ratings),
            'duplicate_user_movie_pairs': duplicate,
            'users_multiple_ratings_at_last_timestamp': last_ties,
            'users_ordinary_last_two_share_timestamp': same_last_two,
            'users_fewer_than_three_timestamps': insufficient,
            'extra_heldout_group_ratings_for_users_with_three_groups': extra,
            'group_embargo_train_events_eligible_users': group_train_events,
            'strict_past_only_train_targets_eligible_users': group_train_events - first_group_targets,
            'warning': 'These are rating-submission times, not viewing times. No split applied.'}


def prepare(args):
    import pandas as pd
    from transformers import AutoTokenizer

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    masks = pd.read_parquet(args.mask)
    selected = sorted(masks.loc[masks['filter'].eq('tmdb_rich4_and_zenodo_vit'), 'movie_id'].tolist())
    assert len(selected) == len(set(selected)) == 2875
    with gzip.open(args.catalogue_audit, 'rt') as f:
        audit = {r['movie_id']: r for r in map(json.loads, f)}
    with zipfile.ZipFile(args.movielens) as z:
        official = {}
        for line in z.read('ml-1m/movies.dat').decode('latin-1').splitlines():
            key, title, genres = line.split('::')
            official[int(key)] = (title, genres.split('|'))
        ratings = np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::', ',')), delimiter=',', dtype=np.int64)
    ratings = ratings[np.isin(ratings[:, 1], selected)]
    assert ratings.shape == (929776, 4)
    wanted = {audit[i]['tmdb_id']: i for i in selected}
    assert len(wanted) == len(selected)
    metadata = {}
    for row in read_csv(args.tmdb, 'movies_metadata.csv'):
        if not row['id'].isdigit() or int(row['id']) not in wanted:
            continue
        key = int(row['id'])
        if row['imdb_id'] != audit[wanted[key]]['imdb_id']:
            continue
        rank = lambda x: (sum(present(x.get(k)) for k in ('overview', 'poster_path', 'genres', 'runtime')),
                          len(x.get('overview', '')), json.dumps(x, sort_keys=True))
        if key not in metadata or rank(row) > rank(metadata[key]):
            metadata[key] = row
    keywords, credits = {}, {}
    for row in read_csv(args.tmdb, 'keywords.csv'):
        if row['id'].isdigit() and int(row['id']) in wanted:
            key = int(row['id']); values = ast.literal_eval(row['keywords'])
            if len(values) > len(keywords.get(key, [])):
                keywords[key] = values
    for row in read_csv(args.tmdb, 'credits.csv'):
        if row['id'].isdigit() and int(row['id']) in wanted:
            key = int(row['id']); cast, crew = ast.literal_eval(row['cast']), ast.literal_eval(row['crew'])
            if key not in credits or len(cast) + len(crew) > sum(map(len, credits[key])):
                credits[key] = cast, crew
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    catalogue, lengths, stage_texts = [], [], [[] for _ in FIELDS]
    for iid in selected:
        key = audit[iid]['tmdb_id']; meta = metadata[key]; cast, crew = credits[key]
        title, genres = official[iid]
        match = re.search(r' \((\d{4})\)$', title)
        assert match, title
        runtime = float(meta['runtime']) if present(meta['runtime']) else 0
        fields = dict(title=title, genres=genres, year=int(match[1]),
                      runtime=(f'{runtime:g} minutes' if runtime > 0 else 'unknown'),
                      keywords=named(keywords[key]),
                      cast=named(sorted(cast, key=lambda x: x.get('order', 100000))),
                      directors=named([x for x in crew if x.get('job') == 'Director']))
        assert fields['keywords'] and fields['cast'] and fields['directors']
        texts = field_texts(fields)
        catalogue.append({'item_id': iid, 'tmdb_id': key, 'imdb_id': audit[iid]['imdb_id'], 'fields': fields})
        for stage, text in enumerate(texts):
            n = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            lengths.append({'movie_id': iid, 'stage': stage + 1, 'last_field': FIELDS[stage], 'tokens': n})
            stage_texts[stage].append({'item_id': iid, 'text': text})
    (out / 'catalogue-fields.json').write_text(json.dumps(catalogue, ensure_ascii=False, indent=2))
    stages = []
    for stage, texts in enumerate(stage_texts):
        path = out / f'catalog-stage-{stage+1:02d}.json'
        path.write_text(json.dumps(texts, ensure_ascii=False))
        lens = np.array([r['tokens'] for r in lengths if r['stage'] == stage + 1])
        stages.append({'stage': stage + 1, 'fields': list(FIELDS[:stage + 1]), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'tokens_median': float(np.median(lens)), 'tokens_p95': float(np.quantile(lens, .95)),
                       'tokens_max': int(lens.max()), **{f'over_{cap}': int((lens > cap).sum()) for cap in (256, 512, 1024, 2048)}})
    pd.DataFrame(lengths).to_csv(out / 'token-lengths.csv', index=False)
    report = {'items': len(selected), 'events': len(ratings), 'stages': stages,
              'timestamp_audit': audit_timestamps(ratings),
              'title_year_removed_at_title_stage': False, 'overview_included': False,
              'year_stage_interpretation': 'explicit Year field repeats information already present in the original title',
              'tag_source': 'TMDB keywords, not user tags/reviews', 'people_policy': 'all source people; no topN truncation',
              'split_applied': False, 'training_started': False}
    (out / 'preflight.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tmdb', type=Path, default=Path('/tmp/ml-text-resources'))
    p.add_argument('--movielens', type=Path, default=Path('/tmp/ml-coverage/ml-1m.zip'))
    p.add_argument('--mask', type=Path, default=Path('/workspace/artifacts/movielens-retention/kept-item-masks.parquet'))
    p.add_argument('--catalogue-audit', type=Path, default=Path('/workspace/artifacts/movielens-semantics/catalogue-audit.jsonl.gz'))
    p.add_argument('--tokenizer', type=Path, default=Path('/workspace/scratch/gr-beauty-input/sentence-t5-base'))
    p.add_argument('--output', type=Path, default=Path('/tmp/ml-text-prepared'))
    prepare(p.parse_args())
