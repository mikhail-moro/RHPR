"""Package audited context candidates and the exact previous ALS control."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.resources import verify_resources
from rhpr.gr.sid import SidIndex


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def package(source, destination):
    source, destination = Path(source), Path(destination)
    config = json.loads((source / 'config.json').read_text())
    previous = Path(config['als_input'])
    provenance = json.loads((source / 'provenance.json').read_text())
    old = verify_resources(previous, provenance['als_manifest_sha256'])
    for record in json.loads((source / 'files.json').read_text()):
        p = source / record['path']
        if (p.stat().st_size != record['size_bytes'] or file_sha256(p) != record['sha256']):
            raise ValueError('CPU context artifact identity mismatch')
    if config['neighbor_ranking'] != 'cosine' or config['windows'] != [20, 5]:
        raise ValueError('this screen requires the audited cosine-neighbor windows20/5')
    destination.mkdir(parents=True, exist_ok=False)
    # Hardlinks save local space, but uploaded files are independent regular bytes.
    for name, origin in [('catalogue.json', previous / 'catalogue.json'),
                         ('original.npy', previous / 't5_rqkmeans-original.npy'),
                         ('als-sids.npy', previous / 't5_rqkmeans_uniform-sids.npy'),
                         ('als-owner.npy', previous / 't5_rqkmeans_uniform-owner.npy')]:
        os.link(origin, destination / name)
    original = np.load(destination / 'original.npy')
    SidIndex.from_full_sids(original)
    indices = {'item_als': dict(family='t5_rqkmeans', role='swapped', variant='als_uniform',
                               semantic_lambda=.3, sids='als-sids.npy', owner='als-owner.npy',
                               original_sids='original.npy', source_manifest_sha256=provenance['als_manifest_sha256'],
                               source_asset='t5_rqkmeans_uniform',
                               selected_snapshot=old['indices']['t5_rqkmeans_uniform']['selected_snapshot'])}
    for window in [20, 5]:
        root = source / f'window-{window}'
        report = json.loads((root / 't5_rqkmeans-search.json').read_text())
        if report['accepted'] != 256:
            raise ValueError('context screen must use the fixed256accepted-move budget')
        name = f'context_{window}'
        with np.load(root / 't5_rqkmeans-candidate.npz') as f:
            for key in ['sids', 'owner', 'moves']:
                np.save(destination / f'{name}-{key}.npy', f[key], allow_pickle=False)
        indices[name] = dict(family='t5_rqkmeans', role='swapped', variant=f'context_als_window{window}',
                             semantic_lambda=.3, sids=f'{name}-sids.npy', owner=f'{name}-owner.npy',
                             original_sids='original.npy', context_window=window,
                             neighbor_ranking='cosine', selected_snapshot=report)
    for candidate in indices.values():
        paths = np.load(destination / candidate['sids'])
        owner = np.load(destination / candidate['owner'])
        np.testing.assert_array_equal(np.sort(owner), np.arange(len(original)))
        np.testing.assert_array_equal(paths, original[owner])
        SidIndex.from_full_sids(paths)
    for name in ['config.json', 'provenance.json', 'curves.json', 'signals.json', 'reliability-audit.json']:
        os.link(source / name, destination / ('context-' + name))
    files = [dict(path=p.name, sha256=file_sha256(p), size_bytes=p.stat().st_size)
             for p in sorted(destination.iterdir())]
    manifest = dict(kind='frozen_sid_graph_placement', catalogue_items=len(original),
                     data_sha256=old['data_sha256'], files=files, indices=indices,
                     full_paths_preserved=True, quantizer_retrained=False,
                     graph_optimization='frozen TRAIN-only context-ALS candidates;no new search in GPU session',
                     source_cpu_files_sha256=file_sha256(source / 'files.json'))
    write(destination / 'resources-manifest.json', manifest)
    write(destination / 'dataset-metadata.json', dict(
        title='RHPR context ALS SID screen inputs', id='carasik/rhpr-sid-context-screen',
        licenses=[dict(name='other')],
        description='Private frozen numerical SID assignments derived from public Amazon Beauty TRAIN data;'
                    ' original and exact previous ALS control, context windows20/5. No raw reviews or credentials.'))
    print(json.dumps(dict(path=str(destination), manifest_sha256=file_sha256(destination / 'resources-manifest.json'),
                          files=len(files), bytes=sum(p.stat().st_size for p in destination.iterdir()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package(args.source, args.output)
