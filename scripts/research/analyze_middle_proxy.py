"""Replay the bounded intermediate proxy screen; never train or read TEST."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.research.analyze_mini_transfer import decode, load, sha
from rhpr.gr.mini_tiger import MIDDLE_ARCHITECTURES
from rhpr.gr.movie_data import case_digest
from rhpr.gr.proxy_architectures import summarize_middle, validate_middle
from rhpr.gr.proxy_target import _stats


def replay(root, recovered, plateau, pilot, transfer, data, output):
    output.mkdir(parents=True, exist_ok=True)
    c = load(root / 'experiment.json'); validate_middle(c)
    manifest = load(root / 'run-manifest.json'); suite = load(root / 'suite-results.json')
    assert manifest['git_sha'] == '2fd232815bc0435f9c54b0f76f5c7964116f5561'
    assert manifest['config'] == load(root / 'plan.json')['config'] == c
    assert not suite['failures'] and len(suite['results']) == 24
    workers = [load(p) for p in sorted((root / 'workers').glob('worker-*.json'))]
    assert len(workers) == 2 and all(len(w['results']) == 12 and not w['failures'] for w in workers)
    assert {k: v for w in workers for k, v in w['results'].items()} == suite['results']
    saved = load(root / 'correlations.json')
    with tempfile.TemporaryDirectory() as d:
        assert summarize_middle(c, suite['results'], Path(d)) == load(root / 'metrics.json')
        assert load(Path(d) / 'correlations.json') == saved
    for split in ['train', 'validation']:
        with np.load(data / (split + '-cases.npz'), allow_pickle=False) as z:
            cases = tuple(z[k] for k in ['contexts', 'user_ids', 'targets'])
        assert case_digest(cases) == c['data']['case_sha256'][split]
    ids = c['index_ids']; checks = []; table = []; times = []; full = []; vectors = {}
    for i in ids:
        for name, digest in c['parents'][i.split('_seed')[1]]['artifacts'][i].items():
            assert sha(root / 'frozen-indexes' / (i + '-' + name)) == digest
        sid = np.load(root / 'frozen-indexes' / (i + '-semantic-ids.npy'), allow_pickle=False)
        ref = c['plateau_reference']['indexes'][i]
        src = (plateau if i.endswith('202') else recovered) / 'trials' / f'full_{i}_gen161803'
        for name, digest in ref['evidence'].items(): assert sha(src / name) == digest
        curve = load(src / 'training-curves.json')
        expected = next(x['validation'] for x in curve if x['epoch'] == ref['best_epoch'])
        p = src / f"validation-epoch-{ref['best_epoch']:03d}.npz"
        v, check = decode(p, sid, cases, expected); checks.append(dict(path=str(p), **check)); full.append(v)
        np.testing.assert_allclose(v.mean(), ref['selected_ndcg10'], atol=1e-12, rtol=0)
        for e, score in ref['early_selected'].items():
            np.testing.assert_allclose(max(x['validation']['ndcg@10'] for x in curve if 'validation' in x and x['epoch'] <= int(e)), score, atol=1e-12, rtol=0)
        row = dict(index_id=i, full_selected=float(v.mean()), best_epoch=ref['best_epoch'])
        for a in ['narrow', *MIDDLE_ARCHITECTURES]:
            parent = (transfer if i.endswith('202') else pilot) if a == 'narrow' else root
            for s in [31415, 27182]:
                tid = f'{a}_{i}_gen{s}'; p = parent / 'trials' / tid
                g = load(p / 'generator.json'); sel = load(p / 'training-selection.json'); curve = load(p / 'training-curves.json')
                assert g['sids_sha256'] == ref['sids_sha256'] and g['test_evaluated'] is False
                if a == 'narrow':
                    rr = c['proxy_reference']['indexes'][i]
                    for name, digest in rr['evidence'][str(s)].items(): assert sha(p / name) == digest
                    epochs = [10]
                else:
                    r = suite['results'][tid]
                    assert load(p / 'trial-result.json') == r and all(sel[k] == v for k, v in r['selection'].items())
                    assert g['config'] == {**c['generator'], **MIDDLE_ARCHITECTURES[a]}
                    assert g['initial_state_sha256'] == r['initial_state_sha256'] == c['initialization_pins'][a][str(s)]
                    assert r['initialization_group'] == f'{a}_{s}' and g['seed'] == s
                    assert g['parameters'] == {'middle_width': 2897952, 'half_depth': 2748416}[a]
                    assert sel['training_epochs'] == 10 and sel['validation_checks'] == 2
                    assert sel['optimizer_updates'] + sel['amp_skipped_updates'] == 2960
                    assert sel['training_precision'] == 'fp16' and sel['evaluation_precision'] == 'fp32'
                    assert sel['training_examples_per_epoch'] == 151166 and len(curve) == 10
                    assert sel['stopping_reason'] == 'epoch_limit'
                    assert [x['epoch'] for x in curve] == list(range(1, 11))
                    assert all(x['updates'] + x['amp_skipped_updates'] == 296*x['epoch'] and x['train_examples'] == 151166 and np.isfinite(x['train_loss_sum_over_codes']) for x in curve)
                    assert [x['epoch'] for x in curve if 'validation' in x] == [5, 10]
                    assert all(np.isfinite(x['validation_loss']['total_ce']) for x in curve if 'validation' in x)
                    epochs = [5, 10]
                for e in epochs:
                    x = next(x for x in curve if x['epoch'] == e)
                    pred = p / f'validation-epoch-{e:03d}.npz'
                    v, check = decode(pred, sid, cases, x['validation']); checks.append(dict(path=str(pred), **check))
                    expected = c['proxy_reference']['indexes'][i]['scores_by_seed'][str(s)] if a == 'narrow' else r['epochs'][str(e)]
                    np.testing.assert_allclose(v.mean(), expected, atol=1e-12, rtol=0)
                    vectors[a, i, s, e] = v; row[f'{a}_{s}_epoch{e}'] = float(v.mean())
                times.append(dict(architecture=a,index_id=i,seed=s,parameters=g['parameters'],
                    mean_epoch_training_seconds=float(np.mean([x['train_seconds'] for x in curve if x['epoch'] <= 10])),
                    ten_epoch_training_ranking_seconds=sum(x['train_seconds'] + x.get('validation', {}).get('seconds', 0) for x in curve if x['epoch'] <= 10),
                    amp_skips_first10=curve[9]['amp_skipped_updates']))
        table.append(row)
    full = np.asarray(full); target = full.mean(1); summaries = {}; arrs = {}
    for a in ['narrow', *MIDDLE_ARCHITECTURES]:
        xx = np.asarray([[vectors[a, i, s, 10] for i in ids] for s in [31415, 27182]])
        mean = xx.mean(0); arrs[a] = mean
        stats = _stats(mean.mean(1).tolist(), target.tolist(), ids)
        expected = saved['groups']['all']['controls']['narrow_mean10'] if a == 'narrow' else saved['groups']['all']['mini'][a + '_epoch10']['mean']
        assert stats.keys() == expected.keys()
        for k, value in stats.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                np.testing.assert_allclose(value, expected[k], atol=1e-12, rtol=0)
            else:
                assert value == expected[k]
        tt = [x for x in times if x['architecture'] == a]
        summaries[a] = dict(primary=stats, own_mean_ndcg10=float(mean.mean()),
            seed_to_seed_spearman=float(spearmanr(xx[0].mean(1), xx[1].mean(1)).statistic),
            seed_to_target_spearman={str(s):float(spearmanr(xx[k].mean(1),target).statistic) for k,s in enumerate([31415,27182])},
            parameters=tt[0]['parameters'], mean_epoch_training_seconds=float(np.mean([x['mean_epoch_training_seconds'] for x in tt])),
            mean_two_seed_training_ranking_minutes=float(sum(x['ten_epoch_training_ranking_seconds'] for x in tt)/6/60),
            amp_skips_first10=sum(x['amp_skips_first10'] for x in tt))
    # Conditional uncertainty only; each draw resamples the SAME users for all models.
    rng = np.random.default_rng(20261107); boot = {a:[] for a in arrs}
    for _ in range(50):
        draws = rng.integers(0, full.shape[1], size=(100, full.shape[1]))
        rt = rankdata(full[:,draws].mean(-1), axis=0); rt -= rt.mean(0)
        for a, u in arrs.items():
            rp = rankdata(u[:,draws].mean(-1), axis=0); rp -= rp.mean(0)
            den = np.sqrt((rp*rp).sum(0)*(rt*rt).sum(0))
            boot[a].extend(np.divide((rp*rt).sum(0),den,out=np.full(100,np.nan),where=den>0))
    result = dict(complete=True,source_code_sha=manifest['git_sha'],finished_at=manifest['finished_at'],
        session_minutes=manifest['duration_seconds']/60,completed_fits=24,failures=0,test_evaluated=False,
        verified_prediction_files=len(checks),prediction_file_counts=dict(new=48,narrow=12,selected_full=6),
        primary_epoch=10,architectures=summaries,comparisons=saved,table=table,timing=times,
        conditional_user_bootstrap_rho_ci95={a:np.nanquantile(v,[.025,.975]).tolist() for a,v in boot.items()},
        bootstrap_scope='5000 shared-user draws of fixed checkpoints; excludes training/tokenizer randomness and checkpoint/index selection correction; not independent confirmation.',
        timing_scope='Mean per-index SUM of both seed fits: 10 epochs training plus recorded ranking-validation only. Excludes loss validation, initialization, I/O, setup. Older narrow timings from different sessions; not matched hardware benchmarking.',
        limitations=saved['limitations'],conclusion='Neither new architecture improves the fixed epoch10 proxy on these six indexes. Higher standalone quality does not guarantee index-ranking transfer. No portable proxy or architecture recipe established.')
    (output / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output / 'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (output / 'indexes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    plot(result, output)
    return result


def plot(r, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(12,8)); names=list(r['architectures'])
    colors=['#4c78a8','#f58518','#e45756']; stats=r['architectures']; xx=np.arange(3)
    ax[0,0].bar(xx,[stats[a]['primary']['spearman'] for a in names],color=colors)
    ax[0,0].axhline(0,color='black',linewidth=.5);ax[0,0].set(xticks=xx,xticklabels=names,ylim=(-1,1),ylabel='Spearman vs selected full',title='Primary: fixed epoch10, two-seed mean')
    for a,col in zip(names,colors):
        y=[np.mean([row[f'{a}_{s}_epoch10'] for s in [31415,27182]]) for row in r['table']]
        ax[0,1].scatter(y,[row['full_selected'] for row in r['table']],color=col,label=a)
    ax[0,1].set(xlabel='Proxy VAL NDCG@10',ylabel='Selected full VAL NDCG@10',title='Better own predictions do not imply better rankings');ax[0,1].legend(fontsize=8)
    ax[1,0].bar(xx,[stats[a]['mean_two_seed_training_ranking_minutes'] for a in names],color=colors)
    ax[1,0].set(xticks=xx,xticklabels=names,ylabel='Minutes / index (sum of two seeds)',title='Training + recorded ranking validation; cross-session')
    for j,a in enumerate(names[1:]):
        v=r['comparisons']['groups']['all']['mini']
        for k,s in enumerate(['31415','27182']):
            ax[1,1].plot([5,10],[v[f'{a}_epoch{e}']['seeds'][s]['spearman'] for e in [5,10]],marker='o',color=colors[j+1],linestyle=['-','--'][k],label=f'{a} / {s}')
    ax[1,1].axhline(0,color='black',linewidth=.5);ax[1,1].set(xticks=[5,10],ylim=(-1,1),xlabel='Proxy epoch',ylabel='Spearman vs selected full',title='Secondary: individual seeds are unstable');ax[1,1].legend(fontsize=7)
    fig.suptitle('Intermediate mini-TIGER architectures — six examined indexes, VAL only');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/('middle-proxy.'+ext),dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['root','recovered','plateau','pilot','transfer','data','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=replay(a.root,a.recovered,a.plateau,a.pilot,a.transfer,a.data,a.output)
    print(json.dumps({k:r[k] for k in ['complete','session_minutes','verified_prediction_files','architectures','conditional_user_bootstrap_rho_ci95']},indent=2))
