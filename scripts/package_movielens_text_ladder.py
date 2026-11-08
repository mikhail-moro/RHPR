"""Package approved temporal split and seven catalogues; no training/dispatch."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import io

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.movie_data import split_ratings, load_temporal_users, temporal_examples, case_digest


def package(prepared, movielens, output):
    output.mkdir(parents=True, exist_ok=True)
    ids = [r['item_id'] for r in json.loads((prepared / 'catalog-stage-01.json').read_text())]
    if len(set(ids)) != 2875:
        raise ValueError('wrong movie subset')
    with zipfile.ZipFile(movielens) as z:
        ratings = np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::', ',')), delimiter=',', dtype=np.int64)
    ratings = ratings[np.isin(ratings[:, 1], ids)]
    rows, excluded, embargo = split_ratings(ratings, seed=42)
    (output / 'sequences.json').write_text(json.dumps(rows, separators=(',', ':')))
    users = load_temporal_users(output / 'sequences.json', ids)
    digests, counts = {}, {}
    for split in ('train', 'validation', 'test'):
        examples = temporal_examples(users, 20, split)
        digests[split] = case_digest(examples); counts[split] = len(examples[2])
    if counts != {'train': 892574, 'validation': 6039, 'test': 6039}:
        raise ValueError(f'temporal count mismatch: {counts}')
    for stage in range(1, 8):
        p = prepared / f'catalog-stage-{stage:02d}.json'
        entries = json.loads(p.read_text())
        assert [r['item_id'] for r in entries] == ids
        shutil.copy2(p, output / p.name)
    for name in ('preflight.json', 'catalogue-fields.json', 'token-lengths.csv'):
        shutil.copy2(prepared / name, output / name)
    report = {'items': len(ids), 'users': len(users), 'retained_events_before_split': len(ratings),
        'train_events': sum(len(u.train) for u in users), 'train_examples': counts['train'],
        'evaluation_examples': counts['validation'], 'case_sha256': digests,
        'excluded_users': excluded, 'embargoed_group_events': len(embargo),
        'seed': 42, 'history_items': 20, 'timestamp_semantics': 'rating submission, not viewing',
        'target_selection': 'SHA256(seed,user_id,timestamp,movie_id), independent of rating/file order',
        'training_context': 'only strictly earlier timestamps; simultaneous targets share preceding history',
        'validation_context': 'strictly earlier training groups',
        'test_context': 'train history plus selected observed validation item; other embargoed items excluded',
        'ratings': 'all values1..5 retained; no positive-rating threshold',
        'candidate_universe': 'same2875staticmovies; no sampled negatives or seen-item mask',
        'title_policy': 'original ML title INCLUDING year; Year stage repeats it explicitly',
        'overview_included': False}
    (output / 'split-report.json').write_text(json.dumps(report, indent=2))
    (output / 'embargoed-events.json').write_text(json.dumps(embargo))
    manifest = {'files': [{'path': p.name, 'size_bytes': p.stat().st_size, 'sha256': file_sha256(p)}
                           for p in sorted(output.iterdir()) if p.is_file() and p.name not in ('resources-manifest.json', 'dataset-metadata.json')]}
    (output / 'resources-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'split': report, 'manifest_sha256': file_sha256(output / 'resources-manifest.json')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=Path('/tmp/ml-text-prepared'))
    p.add_argument('--movielens', type=Path, default=Path('/tmp/ml-coverage/ml-1m.zip'))
    p.add_argument('--output', type=Path, default=Path('/tmp/ml-text-input'))
    a = p.parse_args(); package(a.prepared, a.movielens, a.output)
