"""CPU-only, descriptive audit of where the saved random32 checkpoint wins.

No fitting, inference, checkpoint selection, or data filtering. All exploratory
groups and sensitivity calculations are retained, including negative findings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_kindle_alignment import compare
from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_pairs import collect_keys

MODELS = ('random32', 'pair32', 'mixed32', 'recent32', 'recent20')
CUTOFF = 2.300820444821813  # Previously defined VAL covariate median.


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def credit(rank):
    return np.where(rank <= 10, 1 / np.log2(rank + 1), 0.)


def table(path, rows):
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def concentration(delta, target):
    """Fixed denominator sensitivities, NOT alternative evaluation results."""
    positive = np.sort(delta[delta > 0])[::-1]
    negative = np.sort(delta[delta < 0])
    ids, inverse = np.unique(target, return_inverse=True)
    totals = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(4242)
    draws = rng.integers(0, len(ids), size=(5000, len(ids)))
    bootstrap = totals[draws].sum(1) / counts[draws].sum(1)
    return dict(net_credit=float(delta.sum()), gross_positive_credit=float(positive.sum()),
        gross_negative_credit=float(negative.sum()), distinct_evaluation_targets=len(ids),
        improved_target_items=int((totals > 0).sum()), worsened_target_items=int((totals < 0).sum()),
        conditional_target_cluster_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
        largest_positive_target_credit=float(totals.max()),
        largest_negative_target_credit=float(totals.min()),
        sensitivities=[dict(k=k, positive_credit=float(positive[:k].sum()),
            negative_credit=float(negative[:k].sum()),
            delta_without_k_largest_positive_contributions=float((delta.sum()-positive[:k].sum())/len(delta)),
            delta_without_k_largest_negative_contributions=float((delta.sum()-negative[:k].sum())/len(delta)))
            for k in (1, 2, 3, 5)],
        delta_without_largest_positive_target_contribution=float((delta.sum()-max(0., totals.max()))/len(delta)))


def diagnose(root, parent, mixed, data, vectors, output):
    root, parent, mixed, data, output = map(Path, (root, parent, mixed, data, output))
    vectors = Path(vectors)
    output.mkdir(parents=True, exist_ok=True)
    previous = read(root/'analysis.json')
    config = read(root/'experiment.json')
    for name, sha in config['movie_input']['sha256'].items():
        assert file_sha256(data/name) == sha
    assert previous['all_sampled_epoch_inputs_hashes_replayed'] == 60
    raw = {}
    for split in ('train', 'validation', 'test'):
        with np.load(data/f'{split}-cases.npz') as z:
            raw[split] = tuple(z[k] for k in ('contexts', 'user_ids', 'targets'))
    rank_file = root/'user-ranks.npz'
    with np.load(rank_file) as z:
        saved = {k: z[k] for k in z.files}
    catalog = read(data/'catalog.json'); nitems = len(catalog)
    sid = np.load(root/'frozen-indexes/items-42.npy')
    emb = np.asarray(np.load(vectors), dtype=np.float64)
    assert emb.shape == (nitems, 768) and np.isfinite(emb).all()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    freq = np.bincount(raw['train'][2], minlength=nitems)
    user_freq = np.bincount(raw['train'][1])
    train_user_items = raw['train'][1].astype(np.int64)*nitems+raw['train'][2]
    leakage = {}
    for split in ('validation', 'test'):
        h, u, target = raw[split]
        leakage[split] = dict(target_in_same_user_train_targets=int(np.isin(u.astype(np.int64)*nitems+target,
            train_user_items).sum()), target_in_input_context=int((h == target[:, None]).any(1).sum()))
        assert not any(leakage[split].values())
    fixed_pairs = collect_keys(raw['train'][0], 32, nitems)
    # Potential old-half training candidates, not evidence of actual exposure.
    h = raw['train'][0][:, -32:]; length = (h >= 0).sum(1)
    compact = np.take_along_axis(h, np.minimum((32-length)[:, None]+np.arange(32), 31), axis=1)
    valid = np.arange(15)[None, :]+1 < (length//2)[:, None]
    possible_pairs = np.unique((compact[:, :15].astype(np.int64)*nitems+compact[:, 1:16])[valid])
    np.testing.assert_array_equal(fixed_pairs, possible_pairs)
    folders = dict(random32=root/'trials/random32_wide_42', mixed32=mixed/'trials/mixed32_wide_42',
        pair32=parent/'trials/pair32_wide_42', recent32=parent/'baselines/recent32_wide_42',
        recent20=parent/'baselines/recent20_wide_42')
    ranks = {}; predictions = {}; curves = {}; files = []; features = {}
    for model, folder in folders.items():
        curve = read(folder/'training-curves.json'); curves[model] = curve
        assert read(folder/'trial-result.json')['selection']['best_epoch'] == 55
        for split in ('validation', 'test'):
            test_dir = parent/'control-tests'/folder.name if model.startswith('recent') else folder
            path = folder/'validation-epoch-055.npz' if split == 'validation' else test_dir/'test-best-predictions.npz'
            with np.load(path) as z:
                pred = z['semantic_ids']
                np.testing.assert_array_equal(z['user_ids'], raw[split][1])
                np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
            hit = (pred == sid[raw[split][2], None, :]).all(-1)
            rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
            np.testing.assert_array_equal(rank, saved[f'{split}_{model}_rank'])
            ranks[split, model] = rank; predictions[split, model] = pred
            files.append(dict(path=str(path), sha256=file_sha256(path)))
    np.testing.assert_array_equal(raw['validation'][1], raw['test'][1])
    for split in ('validation', 'test'):
        h, u, target = raw[split]; h = h[:, -32:]; assert (h >= 0).all()
        target_similarity = np.einsum('nd,nkd->nk', emb[target], emb[h])
        pair_similarity = np.einsum('nkd,nkd->nk', emb[h[:, :16:2]], emb[h[:, 1:16:2]]).mean(1)
        pairs = h[:, :16:2].astype(np.int64)*nitems+h[:, 1:16:2]
        features[split] = dict(train_examples=user_freq[u], target_frequency=freq[target],
            recent_popularity=np.log1p(freq[h[:, -16:]]).mean(1), pair_similarity=pair_similarity,
            older_target_similarity=target_similarity[:, :16].max(1),
            recent_target_similarity=target_similarity[:, 16:].max(1),
            older_semantic_advantage=target_similarity[:, :16].max(1)-target_similarity[:, 16:].max(1),
            novel_fixed_pair=(~np.isin(pairs, fixed_pairs)).any(1),
            novel_random_candidate=(~np.isin(pairs, possible_pairs)).any(1))
    thresholds = {f: float(np.median(features['validation'][f])) for f in
        ('train_examples', 'pair_similarity', 'older_semantic_advantage')}
    # Each family is an exhaustive partition; none is optimized on quality.
    groups = []; metrics = []; comparisons = []; examples = []; cross = []; users = []
    for split in ('validation', 'test'):
        h, u, target = raw[split]; f = features[split]; n = len(u)
        freq_target = f['target_frequency']
        masks = dict(
            target_support=dict(cold=freq_target == 0, support1_5=(freq_target >= 1)&(freq_target <= 5),
                support6_20=(freq_target >= 6)&(freq_target <= 20), support_over20=freq_target > 20),
            recent_popularity=dict(low=f['recent_popularity'] <= CUTOFF, high=f['recent_popularity'] > CUTOFF),
            history=dict(shorter=f['train_examples'] <= thresholds['train_examples'], longer=f['train_examples'] > thresholds['train_examples']),
            pair_similarity=dict(lower=f['pair_similarity'] <= thresholds['pair_similarity'], higher=f['pair_similarity'] > thresholds['pair_similarity']),
            older_semantic_advantage=dict(lower=f['older_semantic_advantage'] <= thresholds['older_semantic_advantage'],
                higher=f['older_semantic_advantage'] > thresholds['older_semantic_advantage']),
            novel_fixed_pair=dict(absent=~f['novel_fixed_pair'], present=f['novel_fixed_pair']),
            novel_random_candidate=dict(absent=~f['novel_random_candidate'], present=f['novel_random_candidate']),
            val_fixed_user_popularity=dict(low=features['validation']['recent_popularity'] <= CUTOFF,
                high=features['validation']['recent_popularity'] > CUTOFF))
        for family, partition in masks.items():
            assert np.all(np.stack(list(partition.values())).sum(0) == 1)
        a = ranks[split, 'random32']; ca = credit(a)
        for model in MODELS:
            b = ranks[split, model]; cb = credit(b)
            metrics.append(dict(split=split, model=model, ndcg10=float(cb.mean()),
                **{f'hits{k}': int((b <= k).sum()) for k in (1, 3, 5, 10, 20)},
                distinct_hit_targets=len(np.unique(target[b <= 10])),
                rank1_fraction_of_ndcg=float((b == 1).sum()/cb.sum())))
            if model == 'random32': continue
            delta = ca-cb
            accounting = compare(ca, cb, a, b)
            assert abs(accounting['gain_from_new_hits']+accounting['loss_from_lost_hits']+
                accounting['shared_rank_change']-delta.mean()) < 1e-15
            rank1_delta = float(((a == 1).sum()-(b == 1).sum())/n)
            comparisons.append(dict(split=split, comparator=model, **accounting, **concentration(delta, target),
                rank1_delta_contribution=rank1_delta,
                ranks2_10_delta_contribution=float(delta.mean()-rank1_delta)))
            for family, partition in masks.items():
                for label, mask in partition.items():
                    if not mask.any(): continue
                    groups.append(dict(split=split, comparator=model, family=family, group=label, users=int(mask.sum()),
                        random_ndcg10=float(ca[mask].mean()), comparator_ndcg10=float(cb[mask].mean()),
                        delta_ndcg10=float(delta[mask].mean()), total_delta_contribution=float(delta[mask].sum()/n),
                        random_hits=int((a[mask] <= 10).sum()), comparator_hits=int((b[mask] <= 10).sum()),
                        gained_hits=int(((a <= 10)&(b > 10)&mask).sum()), lost_hits=int(((b <= 10)&(a > 10)&mask).sum())))
                rows = [v for v in groups if v['split'] == split and v['comparator'] == model and v['family'] == family]
                assert abs(sum(v['total_delta_contribution'] for v in rows)-delta.mean()) < 1e-15
            for direction, order in [('gain', np.argsort(-delta)), ('loss', np.argsort(delta))]:
                for i in order[:5]:
                    if delta[i]*(1 if direction == 'gain' else -1) <= 0: continue
                    examples.append(dict(split=split, comparator=model, direction=direction, user=int(u[i]),
                        item=int(target[i]), **catalog[target[i]], target_train_support=int(freq_target[i]),
                        random_rank=int(a[i]), comparator_rank=int(b[i]), discounted_credit_delta=float(delta[i]),
                        **{k: float(v[i]) for k, v in f.items()}))
        for i in range(n):
            users.append(dict(split=split, user=int(u[i]), target=int(target[i]),
                **{f'{m}_rank': int(ranks[split, m][i]) for m in MODELS},
                **{k: float(v[i]) for k, v in f.items()}))
    for model in MODELS[1:]:
        v = credit(ranks['validation', 'random32'])-credit(ranks['validation', model])
        t = credit(ranks['test', 'random32'])-credit(ranks['test', model])
        cross.append(dict(comparator=model, improved_both=int(((v > 0)&(t > 0)).sum()),
            worsened_both=int(((v < 0)&(t < 0)).sum()), val_gain_test_loss=int(((v > 0)&(t < 0)).sum()),
            val_loss_test_gain=int(((v < 0)&(t > 0)).sum()),
            nonzero_either=int(((v != 0)|(t != 0)).sum()),
            zero_both=int(((v == 0)&(t == 0)).sum()),
            delta_correlation=float(np.corrcoef(v, t)[0, 1])))
    curves_result = []
    for model in MODELS[1:]:
        late = []
        for a, b in zip(curves['random32'], curves[model]):
            assert a['epoch'] == b['epoch']
            if 'validation' not in a: continue
            late.append(dict(epoch=a['epoch'], delta=a['validation']['ndcg@10']-b['validation']['ndcg@10']))
        curves_result.append(dict(comparator=model, all_checks=late, wins_all=sum(v['delta'] > 0 for v in late),
            wins_last5=sum(v['delta'] > 0 for v in late if v['epoch'] >= 40),
            mean_delta_last5=float(np.mean([v['delta'] for v in late if v['epoch'] >= 40]))))
    result = dict(source=previous['source'], predictions_verified=files, data_hashes=config['movie_input']['sha256'],
        embeddings_sha256=file_sha256(vectors), saved_rank_sha256=file_sha256(rank_file),
        unchanged_selected_epoch=55, target_leakage_checks=leakage,
        groups_exploratory=True, validation_covariate_thresholds=thresholds,
        previously_fixed_popularity_cutoff=CUTOFF, metrics=metrics, comparisons=comparisons,
        groups=groups, cross_split_same_users=cross, validation_curves=curves_result,
        possible_fixed_pair_count=len(fixed_pairs), possible_random_pair_count=len(possible_pairs),
        limits=['One generator/index/tokenization seed; TEST already inspected. No hypothesis confirmation or multiplicity correction.',
            'Conditional user and target-cluster intervals omit training/index/selection uncertainty; different dependence assumptions.',
            'Covariate groups include future target support and similarity: diagnostic only, not serving gates.',
            'Potential training pair membership is not actual optimizer exposure or unseen individual SID-code tokens.',
            'Top-contribution subtraction retains original denominator and is a sensitivity calculation, never a revised score.',
            'Only final beam20 paths are saved; absence does not distinguish generator-logit versus beam pruning errors.'])
    write(output/'analysis.json', result); write(output/'extreme-examples.json', examples)
    table(output/'user-features.csv', users); table(output/'groups.csv', groups); table(output/'metrics.csv', metrics)
    plot(output, result)
    return result


def plot(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = dict(validation='#287d8e', test='#b24c46')
    for split, color in colors.items():
        comp = [v for v in result['comparisons'] if v['split'] == split]
        x = np.arange(len(comp)) + (-.12 if split == 'validation' else .12)
        means = np.array([v['delta_ndcg10'] for v in comp])
        ci = np.array([v['conditional_user_ci95'] for v in comp])
        axes[0].errorbar(x, means, yerr=np.stack([means-ci[:, 0], ci[:, 1]-means]),
            fmt='o', capsize=3, label=split, color=color)
    axes[0].set_xticks(range(4), MODELS[1:]); axes[0].axhline(0, color='grey', lw=1)
    axes[0].set_ylabel('Random minus comparator NDCG@10'); axes[0].legend()
    axes[0].set_title('Selected checkpoints; conditional user 95% intervals')
    for j, model in enumerate(MODELS):
        row = next(v for v in result['metrics'] if v['split'] == 'test' and v['model'] == model)
        values = np.array([row[f'hits{k}'] for k in (1, 3, 5, 10)])
        previous = 0
        for k, value, color in zip(('1', '2–3', '4–5', '6–10'), np.diff(np.r_[0, values]),
            ('#a24139', '#da976d', '#80aaa0', '#3c7284')):
            axes[1].bar(j, value, bottom=previous, color=color, label=f'Rank {k}' if j == 0 else None)
            previous += value
    axes[1].set_xticks(range(5), MODELS, rotation=25); axes[1].set_ylabel('TEST targets in top 10 (958 total)')
    axes[1].legend(); axes[1].set_title('The apparent gain depends on a few high ranks')
    fig.savefig(output/'diagnosis.png', dpi=180); fig.savefig(output/'diagnosis.pdf'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'parent', 'mixed', 'data', 'vectors', 'output'): p.add_argument('--'+name, required=True)
    args = vars(p.parse_args()); result = diagnose(**args)
    print(json.dumps({k: result[k] for k in ('metrics', 'cross_split_same_users', 'validation_curves')}, indent=2))
