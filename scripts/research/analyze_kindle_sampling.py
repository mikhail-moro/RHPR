"""Replay completed probability/curriculum fits without GPU inference."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_kindle_alignment import compare
from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_sampling import TRIALS, POLICY, ScheduledTokenization, probability_at, validate
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids


def read(path): return json.loads(Path(path).read_text())


def analyze(root, parent, cache_parent, data):
    root, parent, cache_parent, data = map(Path, (root, parent, cache_parent, data))
    c = read(root/'experiment.json'); validate(c)
    manifest = read(root/'run-manifest.json'); suite = read(root/'suite-results.json'); study = read(root/'study-summary.json')
    assert manifest['git_sha'] == 'e239144016df6d98c8f6a7759dc0315e92bcaf0b' and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures'] and study['new_generator_fits'] == 3
    metrics = read(root/'metrics.json'); assert all(not isinstance(v, float) or math.isfinite(v) for v in metrics.values())
    for label, source in [('parent', parent), ('cache_parent', cache_parent)]:
        assert read(source/'run-manifest.json')['git_sha'] == c[label]['git_sha']
        for rel, pin in c[label]['files'].items():
            path = source/rel; assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], rel
    for rel, sha in c['movie_input']['sha256'].items(): assert file_sha256(data/rel) == sha
    for rel in ('items-42.npy', 'pair-keys.npy', 'pairs-42.npy'):
        assert file_sha256(root/'frozen-indexes'/rel) == file_sha256(cache_parent/'frozen-indexes'/rel)
    sid = np.load(root/'frozen-indexes/items-42.npy'); keys = np.load(root/'frozen-indexes/pair-keys.npy'); pairs = np.load(root/'frozen-indexes/pairs-42.npy')
    idx = SidIndex.from_full_sids(sid, width=256); raw = {s: cases(data, s) for s in ('train', 'validation', 'test')}
    records = []; curves = {}; credits = {}; ranks_by_case = {}; files = 0; replayed_epochs = 0
    def replay(path, split, expected):
        nonlocal files
        with np.load(path) as z:
            np.testing.assert_array_equal(z['user_ids'], raw[split][1]); np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
            pred = z['semantic_ids']; scores = z['scores']
        assert pred.shape == (958, 20, 4) and np.isfinite(scores).all() and np.all(np.diff(scores, axis=1) <= 1e-6)
        actual = evaluate_ranked_sids(idx, pred, raw[split][2], cutoffs=(5, 10))
        for key, value in actual.items(): assert abs(value-expected[key]) < 1e-12
        hit = (pred == sid[raw[split][2], None, :]).all(-1); ranks = np.where(hit.any(1), hit.argmax(1)+1, 21)
        files += 1
        return np.where(ranks <= 10, 1/np.log2(ranks+1), 0.), int((ranks <= 10).sum()), ranks
    cells = [(t['id'], root/'trials'/t['id'], t) for t in TRIALS]
    cells += [('p50', cache_parent/'trials/mixed20_wide_42', None), ('recent10', cache_parent/'trials/recent10_wide_42', None),
              ('recent20', parent/'baselines/recent20_wide_42', None)]
    initial = set()
    for name, folder, trial in cells:
        row = read(folder/'trial-result.json'); selection = row['selection']; curve = read(folder/'training-curves.json'); curves[name] = curve
        assert row['raw_case_sha256'] == {s: case_digest(raw[s]) for s in ('train', 'validation')}
        initial.add(row['initial_state_sha256']); assert row['sid_file_sha256'] == file_sha256(root/'frozen-indexes/items-42.npy')
        checks = [r for r in curve if 'validation' in r]; best = max(checks, key=lambda r: r['validation']['ndcg@10'])
        assert best['epoch'] == selection['best_epoch'] and best['validation']['ndcg@10'] == selection['validation_ndcg10']
        assert [r['epoch'] for r in curve] == list(range(1, selection['training_epochs']+1))
        if trial:
            views = ScheduledTokenization(raw['train'][0], sid, keys, pairs, sampling=trial['sampling'])
            for r in curve:
                epoch = r['epoch']; order = np.random.default_rng(42+epoch).permutation(len(raw['train'][2]))
                choices = views.schedule(epoch); histogram = np.zeros(6, dtype=np.int64); actual = 0
                for batch, begin in enumerate(range(0, len(order), 512)):
                    part = order[begin:begin+512]; count = int(choices[batch]); histogram[count] += len(part)
                    slots = np.r_[0, views.merge_slots(count, epoch, batch).cumsum()]
                    actual += int(slots[views.lengths[part]//4].sum())
                import hashlib
                logged = r['training_views']
                assert logged['merge_count_examples'] == histogram.tolist()
                assert logged['pair_summary_occurrences'] == actual
                assert logged['schedule_sha256'] == hashlib.sha256(choices.tobytes()).hexdigest()
                assert logged['merge_fraction'] == probability_at(trial['sampling'], epoch)
                if epoch == 1:
                    for begin in range(0, len(order), 512): views(epoch, begin//512, order[begin:begin+512])
                    assert views.epoch_summary(epoch) == logged, (name, epoch)
                replayed_epochs += 1
            assert row['random_tokenization'] == {**POLICY, 'probability_schedule': trial['sampling']}
            assert selection['stopping_reason'] == 'epoch_limit' and len(curve) == 60
        for r in checks: replay(folder/f"validation-epoch-{r['epoch']:03d}.npz", 'validation', r['validation'])
        credits[name, 'validation'], vhits, ranks_by_case[name, 'validation'] = replay(folder/f"validation-epoch-{best['epoch']:03d}.npz", 'validation', best['validation'])
        if name == 'recent20':
            testdir = parent/'control-tests/recent20_wide_42'; test = read(testdir/'control-result.json')['test']
        else: testdir = folder; test = row['test']
        assert test['epoch'] == best['epoch']
        credits[name, 'test'], thits, ranks_by_case[name, 'test'] = replay(testdir/'test-best-predictions.npz', 'test', test['metrics'])
        records.append(dict(model=name, best_epoch=best['epoch'], epochs=len(curve), validation_ndcg10=selection['validation_ndcg10'],
            test_ndcg10=test['metrics']['ndcg@10'], validation_hits10=vhits, test_hits10=thits,
            forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9,
            warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'],
            final_probability=probability_at(trial['sampling'], len(curve)) if trial else None))
    assert len(initial) == 1
    comparisons = {name: {split: compare(credits[name, split], credits['p50', split], ranks_by_case[name, split], ranks_by_case['p50', split]) for split in ('validation', 'test')} for name, _, trial in cells if trial}
    result = dict(records=records, comparisons_to_p50=comparisons, prediction_files_replayed=files,
        actual_training_count_exposure_epochs_replayed=replayed_epochs,
        full_training_tensor_hash_epochs_replayed=3, data_and_parent_hashes_verified=True,
        duration_seconds=manifest['duration_seconds'],
        limitations=['One seed;previously inspected TEST;conditional user bootstrap excludes seed/item dependence;cap60 not plateau.'])
    (root/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, curve in curves.items():
        checks = [r for r in curve if 'validation' in r]
        axes[0].plot([r['epoch'] for r in checks], [r['validation']['ndcg@10'] for r in checks], label=name)
        axes[1].plot([r['epoch'] for r in checks], [r['validation_loss']['total_ce'] for r in checks], label=name)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].set_ylabel('VAL NDCG@10'); axes[1].set_ylabel('VAL teacher-forced CE'); axes[0].legend(fontsize=7)
    fig.savefig(root/'sampling-curves.png', dpi=180); fig.savefig(root/'sampling-curves.pdf'); plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('results', 'parent', 'cache-parent', 'data'): parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); r = analyze(a.results, a.parent, a.cache_parent, a.data)
    print(json.dumps(r, indent=2))
