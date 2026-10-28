"""Reproducible local CPU experiment, fixed graphs, held-out TRAIN users.

Run with PYTHONPATH=src python scripts/probe_sid_graph_swap.py --config PATH.
Large source arrays are referenced, not copied. Export hashes allow a later
generator fit to use exact saved item->SID assignments without recomputation.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from rhpr.gr.data import file_sha256, load_beauty
from rhpr.gr.index_proxies import IndexProxies
from rhpr.gr.sid_graph_swap import (PairTerm, ContrastiveTerm, SwapSearch,
    cosine_neighbors, couser_neighbors, heldout_gain)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def entropy(counts):
    p = counts[counts > 0]/counts.sum()
    return float(-np.sum(p*np.log2(p)))


def dimension99(z):
    x = np.array(z, dtype=np.float64, copy=True); x -= x.mean(0)
    variance = np.maximum(np.linalg.eigvalsh(x.T@x)[::-1], 0)
    return int(np.searchsorted(np.cumsum(variance), .99*variance.sum())+1)


def explained_variance(labels, values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1: x = x[:, None]
    x = x-x.mean(0)
    count = np.bincount(labels); total = float(np.square(x).sum())
    matrix = sparse.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))),
                               shape=(len(count), len(labels)))
    sums = matrix@x
    return float(np.sum(np.square(sums[count > 0])/count[count > 0, None])/total)


def code_statistics(codes):
    result = {}
    for depth in range(1, 4):
        _, labels, count = np.unique(codes[:, :depth], axis=0, return_inverse=True, return_counts=True)
        result.update({f'prefixes_l{depth}':len(count), f'catalogue_prefix_entropy_l{depth}':entropy(count),
                       f'max_prefix_group_l{depth}':int(count.max())})
    result['semantic_unique_ids'] = result['prefixes_l3']
    result['max_collision_group'] = result['max_prefix_group_l3']
    for level in range(3):
        count = np.bincount(codes[:, level], minlength=256)
        result[f'codes_used_l{level+1}'] = int(np.count_nonzero(count))
        result[f'code_perplexity_l{level+1}'] = 2**entropy(count)
    return result


def legacy_nll(codes, proxy):
    """Same three-fold count model and smoothing as the earlier RQ-VAE plots."""
    totals = {'heldout_unigram_nll_bits':0., 'heldout_conditional_nll_bits':0.}
    count = 0
    for fold in range(3):
        fit, held = proxy.fold != fold, proxy.fold == fold
        a = SimpleNamespace(src=proxy.src[fit],dst=proxy.dst[fit])
        b = SimpleNamespace(src=proxy.src[held],dst=proxy.dst[held])
        score = heldout_gain(codes,a,b)
        for key in totals: totals[key] += score[key]*int(held.sum())
        count += int(held.sum())
    unigram = totals['heldout_unigram_nll_bits']/count
    conditional = totals['heldout_conditional_nll_bits']/count
    return dict(unigram_nll_bits=unigram, conditional_nll_bits=conditional,
                relative_nll_reduction=1-conditional/unigram)


def quantization_metrics(codes, index, z, books):
    if index['kind'] == 'rqvae':
        reconstructed = sum(books[d][codes[:, d]] for d in range(3))
        error = float(np.mean(np.square(z-reconstructed)))
        baseline = float(np.mean(np.square(z-z.mean(0))))
        return dict(fixed_book_quantization_mse=error, relative_quantization_mse=error/baseline)
    residual = np.asarray(z, dtype=np.float64).copy(); result = {}
    for d in range(3):
        residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-30)
        residual -= books[d][codes[:, d]]
        result[f'normalized_residual_mse_l{d+1}'] = float(np.square(residual).mean())
    return result


def diagnose(search, index, z, books, embedding, title, proxies, graphs, panels, state):
    codes = search.codes
    all_proxy, fit_proxy, held_proxy = proxies
    initial = state.pop('_initial')
    row = dict(state, objective=search.score, accepted=search.accepted,
               moved_items=int(np.count_nonzero(search.owner != np.arange(len(codes)))),
               moved_fraction=float(np.mean(search.owner != np.arange(len(codes)))),
               semantic_prefix_changed_fraction=float(np.mean(np.any(codes[:, :3] != initial[:, :3], axis=1))))
    row.update(all_proxy.first(codes[:, 0]))
    row.update(legacy_nll(codes[:, 0],all_proxy))
    row['heldout_prefix_coherence'] = held_proxy.first(codes[:, 0])['train_prefix_coherence']
    for depth in range(1, 4):
        _, labels = np.unique(codes[:, :depth], axis=0, return_inverse=True)
        mass = np.bincount(labels[all_proxy.dst],minlength=labels.max()+1)
        row[f'train_prefix_entropy_l{depth}'] = entropy(mass)
        row[f'train_next_token_entropy_l{depth}'] = row[f'train_prefix_entropy_l{depth}']-(row[f'train_prefix_entropy_l{depth-1}'] if depth>1 else 0)
        for key, value in heldout_gain(labels, fit_proxy, held_proxy).items():
            row[f'{key}_l{depth}'] = value
    row.update(code_statistics(codes), **quantization_metrics(codes, index, z, books))
    _, labels = np.unique(codes[:, 0], return_inverse=True)
    row['first_fulltext_variance_explained'] = explained_variance(labels, embedding)
    row['first_title_variance_explained'] = explained_variance(labels, title)
    for name, graph in graphs.items():
        row[f'{name}_score'] = graph.score(codes)
        row[f'{name}_macro'] = graph.macro_score(codes)
        for d, value in enumerate(graph.prefix_scores(codes), 1): row[f'{name}_prefix_l{d}'] = value
    for name, panel in panels.items(): row[f'{name}_infonce_loss'] = panel.loss(codes)
    # Report cold and popularity cohorts on unseen TRAIN-user transitions.
    thresholds = np.quantile(fit_proxy.frequency, [.5, .9])
    groups = {'zero':fit_proxy.frequency == 0,
              'rare':(fit_proxy.frequency > 0) & (fit_proxy.frequency <= thresholds[0]),
              'middle':(fit_proxy.frequency > thresholds[0]) & (fit_proxy.frequency <= thresholds[1]),
              'popular':fit_proxy.frequency > thresholds[1]}
    for name, group in groups.items():
        mask = group[held_proxy.dst]
        row[f'heldout_targets_{name}'] = int(mask.sum())
        if mask.any():
            tiny = SimpleNamespace(src=held_proxy.src[mask], dst=held_proxy.dst[mask])
            row[f'heldout_gain_{name}'] = heldout_gain(codes[:, 0], fit_proxy, tiny)['heldout_gain_bits']
    return row


def build(config, out):
    begin = time.monotonic()
    data = Path(config['data']); embedding_path = Path(config['embeddings'])
    if file_sha256(embedding_path) != config['embeddings_sha256']:
        raise ValueError('pinned Sentence-T5 embedding mismatch')
    _, users, ids, counts = load_beauty(data)
    embedding = np.load(embedding_path)
    if embedding.shape != (len(ids), 768): raise ValueError('catalogue shape mismatch')
    with np.load(config['fields']) as f:
        price, observed, title = f['price'][:, 0], f['price_observed'], f['title']
    split = config['split']; order = np.random.default_rng(split['seed']).permutation(len(users))
    number = round(len(users)*split['heldout_fraction'])
    held_rows, fit_rows = np.sort(order[:number]), np.sort(order[number:])
    fit_users, held_users = [users[i] for i in fit_rows], [users[i] for i in held_rows]
    np.savez_compressed(out/'user-split.npz', fit_rows=fit_rows, heldout_rows=held_rows,
                        fit_user_ids=np.array([u.user_id for u in fit_users]),
                        heldout_user_ids=np.array([u.user_id for u in held_users]), source_item_ids=np.array(ids))
    proxies = [IndexProxies(u, price, observed) for u in [users, fit_users, held_users]]
    graph_config = config['graph']; kmax = graph_config['maximum_k']
    sem = cosine_neighbors(embedding, kmax, graph_config['block'])
    log(phase='semantic_graph_complete', seconds=time.monotonic()-begin, edges=len(sem.source))
    fit, ff = couser_neighbors(fit_users, len(ids), kmax, graph_config['minimum_cf_users'])
    held, hf = couser_neighbors(held_users, len(ids), kmax, graph_config['minimum_cf_users'])
    for name, graph in [('semantic',sem),('fit_cf',fit),('heldout_cf',held)]:
        np.savez_compressed(out/f'{name}-graph.npz', source=graph.source, target=graph.target, weight=graph.weight)
    graphs = {'semantic20':sem.top(20), 'fit_cf20':fit.top(20), 'heldout_cf20':held.top(20)}
    panels = {}
    for offset, (name, graph) in enumerate([('fit',graphs['fit_cf20']),('heldout',graphs['heldout_cf20'])]):
        spec = config['contrastive']
        panel = ContrastiveTerm(graph, spec['seed']+offset, spec['negatives'], spec['positives_per_anchor'], spec['temperature'])
        panels[name] = panel
        np.savez_compressed(out/f'{name}-infonce-panel.npz', anchor=panel.anchor, candidates=panel.candidates, mass=panel.mass)
    proposals = np.random.default_rng(config['search']['seed']).integers(len(ids), size=(config['search']['proposals'],2))
    # Exclude self proposals without changing their number, one deterministic draw.
    equal = proposals[:,0] == proposals[:,1]; proposals[equal,1] = (proposals[equal,1]+1)%len(ids)
    np.save(out/'proposals.npy', proposals)
    log(phase='graphs_complete', seconds=time.monotonic()-begin, fit_users=len(fit_users), heldout_users=len(held_users),
        fit_cf_edges=len(fit.source), heldout_cf_edges=len(held.source))
    inputs = [data/'catalog.json', data/'sequences.json', embedding_path, Path(config['fields'])]
    for index in config['indices']:
        inputs.extend(Path(index[key]) for key in ['codes','books','source'])
        if 'latents' in index: inputs.append(Path(index['latents']))
    script = Path(__file__).resolve(); module = script.parents[1]/'src/rhpr/gr/sid_graph_swap.py'
    inputs.extend([script,module,script.parents[1]/'src/rhpr/gr/index_proxies.py'])
    save_json(out/'manifest.json', dict(config=config, counts=counts,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        executed_source_sha256={str(p):file_sha256(p) for p in inputs},
        fit_users=len(fit_users), heldout_users=len(held_users),
        evaluation='TRAIN events of disjoint users; recommender validation/test targets never read by graph/loss/proxies',
        warning='Legacy all-TRAIN proxies are adaptive/descriptive after graph optimization; use heldout-user metrics for transfer.',
        graph_seconds=time.monotonic()-begin, generator_trained=False, kaggle_submitted=False))
    return embedding, title, proxies, sem, fit, graphs, panels, proposals


def run(config):
    out = Path(config['output']); out.mkdir(parents=True,exist_ok=False)
    save_json(out/'config.json',config)
    start = time.monotonic(); rows = []
    embedding,title,proxies,sem,fit,graphs,panels,proposals = build(config,out)
    for index in config['indices']:
        initial = np.load(index['codes']); books = np.load(index['books']).astype(np.float64)
        z = np.load(index['latents']).astype(np.float64) if 'latents' in index else embedding.astype(np.float64)
        if initial.shape != (len(embedding),4): raise ValueError('complete four-token SID required')
        invariant = code_statistics(initial); pca = dimension99(z)
        reference_acceptance = {}; index_start = time.monotonic()
        for trial in config['trials']:
            trial_start = time.monotonic(); directory = out/index['id']/trial['id']; directory.mkdir(parents=True)
            cf_term = panels['fit'] if trial['kind']=='infonce' else PairTerm(fit.top(trial['k']))
            terms = [cf_term,PairTerm(sem.top(trial['k']))]
            coefficients = [0 if trial['kind']=='semantic' else 1,trial['lambda']]
            search = SwapSearch(initial,terms,coefficients)
            previous = 0; random_cursor = 0; move_log = []
            for step in config['search']['snapshots']:
                if trial['kind']=='random':
                    target = reference_acceptance[step]
                    while search.accepted < target:
                        a,b = proposals[random_cursor]; random_cursor += 1
                        accepted,delta = search.try_swap(a,b,force=True)
                        move_log.append([random_cursor,int(a),int(b),delta])
                else:
                    for position in range(previous,step):
                        a,b = proposals[position]; accepted,delta = search.try_swap(a,b)
                        if accepted: move_log.append([position+1,int(a),int(b),delta])
                previous = step
                search.audit(initial)
                state=dict(index=index['id'],trial=trial['id'],kind=trial['kind'],k=trial['k'],
                    semantic_lambda=trial['lambda'],proposals=step,pca99=pca,
                    _initial=initial,search_and_diagnostics_seconds=time.monotonic()-trial_start)
                row=diagnose(search,index,z,books,embedding,title,proxies,graphs,panels,state)
                for key,value in invariant.items():
                    if abs(row[key]-value)>1e-10: raise AssertionError(f'index structure changed: {key}')
                row['actual_random_proposals'] = random_cursor if trial['kind']=='random' else 0
                rows.append(row)
                np.savez_compressed(directory/f'step-{step:06d}.npz',sids=search.codes,owner=search.owner)
                if trial['id']=='lambda_1': reference_acceptance[step]=search.accepted
                save_json(out/'curves.json',rows)
                log(phase='snapshot',index=index['id'],trial=trial['id'],proposals=step,
                    accepted=search.accepted,moved=row['moved_fraction'],gain=row['heldout_gain_bits_l1'],
                    semantic=row['semantic20_score'],heldout_cf=row['heldout_cf20_score'],seconds=time.monotonic()-trial_start)
            np.save(directory/'sids.npy',search.codes);np.save(directory/'owner.npy',search.owner)
            np.save(directory/'accepted-swaps.npy',np.asarray(move_log).reshape(-1,4))
            save_json(directory/'result.json',dict(final=row,sids_sha256=file_sha256(directory/'sids.npy'),
                owner_sha256=file_sha256(directory/'owner.npy'),verified_exact_local_deltas=True,
                verified_prefix_occupancies_unchanged=True,elapsed_seconds=time.monotonic()-trial_start))
        log(phase='index_complete',index=index['id'],seconds=time.monotonic()-index_start)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    save_json(out/'metrics.json',dict(trials=len(config['indices'])*len(config['trials']),snapshots=len(rows),
        seconds=time.monotonic()-start,generator_trained=False,kaggle_submitted=False))
    log(phase='complete',seconds=time.monotonic()-start,snapshots=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=config['cpu_threads']): run(config)
