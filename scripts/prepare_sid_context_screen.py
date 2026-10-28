"""CPU-only context-ALS SID candidates; reuses pinned factors, no GPU launch."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from rhpr.gr.data import file_sha256, load_beauty
from rhpr.gr.index_proxies import IndexProxies
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_context_signal import context_neighbors, context_prototypes, prototype_reliability
from rhpr.gr.sid_graph_search import budgeted_search
from rhpr.gr.sid_graph_swap import NeighborGraph, heldout_gain


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def graph(path, n):
    with np.load(path) as f:
        return NeighborGraph(f['source'], f['target'], f['weight'], n)


def run(config):
    started = time.monotonic()
    if (not config['windows'] or config['primary_window'] not in config['windows']
            or len(set(config['windows'])) != len(config['windows'])):
        raise ValueError('primary window must occur exactly once in the window list')
    prior = json.loads(Path(config['preparation_config']).read_text())
    assets = Path(config['als_input'])
    # Check both the original upstream assets and the prior exported factor files.
    manifest = json.loads((assets / 'resources-manifest.json').read_text())
    for record in manifest['files']:
        if file_sha256(assets / record['path']) != record['sha256']:
            raise ValueError('prior ALS resource hash mismatch')
    for record in prior['sources'].values():
        if file_sha256(Path(record['path'])) != record['sha256']:
            raise ValueError('upstream source hash mismatch')
    data = Path(prior['data'])
    for name, digest in manifest['data_sha256'].items():
        if file_sha256(data / name) != digest:
            raise ValueError('data identity mismatch')
    _, users, ids, counts = load_beauty(data)
    n = len(ids)
    if n != manifest['catalogue_items']:
        raise ValueError('catalogue size mismatch')
    with np.load(assets / 'user-split.npz') as f:
        fit_rows, held_rows = f['fit_rows'], f['heldout_rows']
    np.testing.assert_array_equal(np.sort(np.r_[fit_rows, held_rows]), np.arange(len(users)))
    fit_users, held_users = [users[i] for i in fit_rows], [users[i] for i in held_rows]
    factors = np.load(assets / 'als-item-factors.npy')
    frequency = np.load(assets / 'cf-fit-user-frequency.npy')
    # Assert that the factor fit used exactly these TRAIN users/events and item IDs.
    rr, cc = [], []
    for u, user in enumerate(fit_users):
        unique = np.unique(user.train)
        rr.extend([u] * len(unique)); cc.extend(unique)
    matrix = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(fit_users), n))
    digest = hashlib.sha256()
    for a in [matrix.indptr, matrix.indices, matrix.data]:
        digest.update(a.tobytes())
    als = json.loads((assets / 'als.json').read_text())
    if digest.hexdigest() != als['input_csr_sha256']:
        raise ValueError('ALS TRAIN user matrix identity mismatch')
    np.testing.assert_array_equal(np.asarray((matrix > 0).sum(0)).ravel(), frequency)
    if hashlib.sha256(factors.tobytes()).hexdigest() != als['item_factor_sha256']:
        raise ValueError('ALS factor matrix identity mismatch')
    semantic = graph(assets / 'semantic-graph.npz', n)
    item_cf = graph(assets / 'als-graph.npz', n)
    embedding = np.load(prior['sources']['embeddings']['path'], mmap_mode='r')
    with np.load(prior['sources']['fields']['path']) as f:
        price, observed = f['price'][:, 0], f['price_observed']
    proxies = IndexProxies(users, price, observed)
    fit_proxy, held_proxy = IndexProxies(fit_users, price, observed), IndexProxies(held_users, price, observed)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', config)
    provenance = dict(als_manifest_sha256=file_sha256(assets / 'resources-manifest.json'),
                      preparation_config_sha256=file_sha256(Path(config['preparation_config'])),
                      sources=prior['sources'], als_input=str(assets), data_sha256=manifest['data_sha256'],
                      counts=counts, als=als, factors_refit=False, generator_fitted=False,
                      validation_and_test_used=False,
                      selection='primary window fixed before analysis; sensitivities not selected by ranking outcomes')
    write(out / 'provenance.json', provenance)
    print(json.dumps(dict(phase='context_inputs_verified', fit_users=len(fit_users), heldout_users=len(held_users))), flush=True)
    all_curves, reports = [], []
    for window in config['windows']:
        event_eligible = frequency >= config['minimum_event_factor_users']
        fit = context_prototypes(fit_users, factors, window, event_eligible)
        held = context_prototypes(held_users, factors, window, event_eligible)
        ranking = config['neighbor_ranking']
        cf = context_neighbors(fit, config['neighbors'], config['minimum_context_users'], config['support_shrinkage'], ranking=ranking)
        held_cf = context_neighbors(held, config['neighbors'], config['minimum_context_users'], config['support_shrinkage'], ranking=ranking)
        directory = out / f'window-{window}'
        directory.mkdir()
        # Keep both original item vectors (via pinned reference) and context vectors.
        # Only primary vectors/graph are persisted; sensitivity aggregates are small.
        if window == config['primary_window']:
            if config.get('reuse_primary_vectors') and Path(config['reuse_primary_vectors']).is_file():
                source = Path(config['reuse_primary_vectors'])
                np.testing.assert_array_equal(np.load(source), fit.vectors)
                os.link(source, directory / 'context-vectors.npy')
            else:
                np.save(directory / 'context-vectors.npy', fit.vectors)
            np.savez_compressed(directory / 'context-graph.npz', source=cf.source, target=cf.target, weight=cf.weight)
        np.savez_compressed(directory / 'support.npz', contexts=fit.contexts, users=fit.users,
                            length_sum=fit.length_sum, embedded_sum=fit.embedded_sum,
                            heldout_contexts=held.contexts, heldout_users=held.users)
        report = dict(window=window, fit=fit.report, heldout=held.report,
                      neighbor_ranking=ranking,
                      prototype_reliability=prototype_reliability(fit, held),
                      fit_edges=len(cf.source), heldout_edges=len(held_cf.source),
                      fit_graph_items=int(np.unique(np.r_[cf.source, cf.target]).size))
        for key, g in [('fit', cf), ('heldout', held_cf)]:
            mass = np.bincount(g.target, weights=g.weight, minlength=n)
            report[key + '_target_concentration'] = dict(
                top100_weight_fraction=float(np.sort(mass)[-100:].sum() / mass.sum()),
                effective_targets=float(mass.sum()**2 / (mass @ mass)))
        reports.append(report)
        write(directory / 'signal.json', report)
        for family in config['families']:
            initial = np.load(assets / f'{family}-original.npy')

            def probe(variant, codes, accepted=0, proposals=0):
                row = dict(window=window, family=family, variant=variant, accepted=accepted, proposals=proposals,
                           changed_items=int(np.any(codes != initial, axis=1).sum()),
                           context_fit_score=cf.score(codes), context_heldout_score=held_cf.score(codes),
                           item_als_score=item_cf.score(codes), semantic_score=semantic.score(codes),
                           **proxies.first(codes[:, 0]), **heldout_gain(codes[:, 0], fit_proxy, held_proxy))
                all_curves.append(row)
                return row

            probe('original', initial)
            probe('existing_item_als_uniform', np.load(assets / f'{family}_uniform-sids.npy'),
                  prior['search']['accepted_cap'])
            codes, owner, moves, snapshots, search = budgeted_search(
                initial, cf, semantic, embedding, fit.users, prior['search'], 'uniform')
            SidIndex.from_full_sids(codes)
            np.testing.assert_array_equal(codes, initial[owner])
            for depth in range(1, initial.shape[1] + 1):
                a, ac = np.unique(initial[:, :depth], axis=0, return_counts=True)
                b, bc = np.unique(codes[:, :depth], axis=0, return_counts=True)
                np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(ac, bc)
            for position, accepted, paths, _ in snapshots[1:]:
                probe('context_als_uniform', paths, accepted, position)
            np.savez_compressed(directory / f'{family}-candidate.npz', sids=codes, owner=owner, moves=moves)
            write(directory / f'{family}-search.json', search)
            print(json.dumps(dict(phase='context_candidate_ready', window=window, family=family, **search)), flush=True)
    write(out / 'curves.json', all_curves)
    write(out / 'signals.json', reports)
    write(out / 'files.json', [dict(path=p.relative_to(out).as_posix(), sha256=file_sha256(p), size_bytes=p.stat().st_size)
                               for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(phase='context_preparation_complete', seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(json.loads(Path(args.config).read_text()))
