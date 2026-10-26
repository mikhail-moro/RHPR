"""Enrich/replot a finished swap probe; never runs search or fits a model."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_sid_graph_swap import save_json, legacy_nll, entropy
from rhpr.gr.data import load_beauty, file_sha256
from rhpr.gr.index_proxies import IndexProxies
from rhpr.gr.sid_graph_swap import heldout_gain


def transition_gain(codes, fit, held):
    _,labels=np.unique(codes,return_inverse=True);k=labels.max()+1
    yfit=labels[fit.dst];y=labels[held.dst]
    prior=(np.bincount(yfit,minlength=k)+.5)/(len(yfit)+.5*k)
    keys,mass=np.unique(fit.src*k+yfit,return_counts=True)
    query=held.src*k+y;pos=np.searchsorted(keys,query);valid=pos<len(keys)
    valid[valid]=keys[pos[valid]]==query[valid]
    match=np.zeros(len(query));match[valid]=mass[pos[valid]]
    total=np.bincount(fit.src,minlength=len(codes));p=(match+10*prior[y])/(total[held.src]+10)
    return np.log2(p/prior[y])


def bootstrap_deltas(out,config,users,price,observed,rows):
    with np.load(out/'user-split.npz') as f: fit_rows=f['fit_rows'];held_rows=f['heldout_rows']
    fit=IndexProxies([users[i] for i in fit_rows],price,observed)
    held=IndexProxies([users[i] for i in held_rows],price,observed)
    user=np.repeat(np.arange(len(held_rows)),[len(users[i].train)-1 for i in held_rows])
    count=np.bincount(user,minlength=len(held_rows)); records=[]
    for index in config['indices']:
        initial=np.load(index['codes']);base=transition_gain(initial[:,0],fit,held)
        full_reference=heldout_gain(np.unique(initial,axis=0,return_inverse=True)[1],fit,held)['heldout_gain_bits']
        for r in [r for r in rows if r['index']==index['id'] and r['proposals']==config['search']['proposals']]:
            codes=np.load(out/r['index']/r['trial']/'sids.npy')
            full=heldout_gain(np.unique(codes,axis=0,return_inverse=True)[1],fit,held)['heldout_gain_bits']
            assert abs(full-full_reference)<1e-12
            diff=transition_gain(codes[:,0],fit,held)-base
            assert abs(diff.mean()-(r['heldout_gain_bits_l1']-base.mean()))<1e-12
            sums=np.bincount(user,weights=diff,minlength=len(held_rows));rng=np.random.default_rng(2704)
            samples=[]
            for _ in range(20):
                draw=rng.integers(len(held_rows),size=(100,len(held_rows)))
                samples.extend(sums[draw].sum(1)/count[draw].sum(1))
            low,high=np.quantile(samples,[.025,.975])
            records.append(dict(index=r['index'],trial=r['trial'],gain_delta_bits=float(diff.mean()),
                ci95_low=float(low),ci95_high=float(high),repeats=2000,seed=2704,
                bootstrap_unit='heldout TRAIN user, paired; ratio of transition sums',
                full_sid_count_gain_invariant=True,
                caveat='Conditional on this fit-user split/index/proposals. Exploratory multiple comparisons, not independent training repeats.'))
    save_json(out/'analysis/paired-user-bootstrap.json',records)


COLORS = dict(cf_only='#c62828',lambda_0p3='#ef8b19',lambda_1='#1b8d49',lambda_3='#087eab',
    semantic_only='#6d51a8',infonce_lambda_1='#bb4190',k5_lambda_1='#748b28',k50_lambda_1='#886440',random_matched='#777777')


def report(out):
    config=json.loads((out/'config.json').read_text()); rows=json.loads((out/'curves.json').read_text())
    _,users,_,_=load_beauty(Path(config['data']))
    with np.load(config['fields']) as f: price,observed=f['price'][:,0],f['price_observed']
    proxy=IndexProxies(users,price,observed)
    # Offline enrichment: earlier snapshots are always reused, never re-optimized.
    for r in rows:
        p=out/r['index']/r['trial']/f'step-{r["proposals"]:06d}.npz'
        with np.load(p) as f: codes=f['sids']
        r.update(legacy_nll(codes[:,0],proxy))
        assert abs(r['train_proxy_gain_bits']-(r['unigram_nll_bits']-r['conditional_nll_bits']))<1e-12
        for d in range(1,4):
            _,labels=np.unique(codes[:,:d],axis=0,return_inverse=True)
            r[f'train_prefix_entropy_l{d}']=entropy(np.bincount(labels[proxy.dst],minlength=labels.max()+1))
            r[f'train_next_token_entropy_l{d}']=r[f'train_prefix_entropy_l{d}']-(r[f'train_prefix_entropy_l{d-1}'] if d>1 else 0)
    save_json(out/'curves.json',rows)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    save_json(analysis/'enrichment.json',dict(script_sha256=file_sha256(Path(__file__)),
        snapshot_count=len(rows),new_search=False,new_training=False,
        fields=['unigram_nll_bits','conditional_nll_bits','relative_nll_reduction',
                'train_prefix_entropy_l1..3','train_next_token_entropy_l1..3']))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    indices=[x['id'] for x in config['indices']];trial_ids=[t['id'] for t in config['trials']]
    for figure_id, panels in [
        ('transfer', [('semantic20_score','Semantic graph: weighted LCP'),('fit_cf20_score','CF graph: optimization users'),
          ('heldout_cf20_score','CF graph: unseen TRAIN users'),('heldout_gain_bits_l1','Next-token gain: unseen users, bits')]),
        ('contrastive', [('fit_infonce_loss','Fixed fit InfoNCE loss (lower better)'),('heldout_infonce_loss','Fixed held-out InfoNCE loss'),
          ('heldout_gain_bits_l2','Next-prefix gain: two levels, bits'),('heldout_gain_bits_l3','Next-prefix gain: three levels, bits')]),
        ('levels', [('semantic20_prefix_l1','Semantic: same first code'),('semantic20_prefix_l2','Semantic: same first two codes'),
          ('heldout_cf20_prefix_l1','Held-out CF: same first code'),('heldout_cf20_prefix_l2','Held-out CF: same first two codes')])]:
        fig,axes=plt.subplots(len(indices),len(panels),figsize=(17,7),squeeze=False)
        for i,index in enumerate(indices):
            for ax,(key,label) in zip(axes[i],panels):
                for trial in trial_ids:
                    rr=[r for r in rows if r['index']==index and r['trial']==trial]
                    ax.plot([r['proposals'] for r in rr],[r[key] for r in rr],label=trial,color=COLORS[trial],
                            linestyle='--' if trial=='random_matched' else '-',marker='.',markersize=4)
                ax.set_title(label);ax.set_xlabel('Proposals (random control matched by accepted swaps)')
                if ax is axes[i,0]:ax.set_ylabel(index)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,-.01))
        fig.suptitle('Fixed paths; frozen embeddings; disjoint TRAIN-user graph evaluation',fontsize=13)
        fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(analysis/f'{figure_id}.png',dpi=160);plt.close(fig)
    # Legacy RQ-VAE comparison metrics remain explicit and on their original units.
    legacy=[('train_proxy_gain_bits','Legacy gain: all TRAIN (adaptive)'),('train_prefix_coherence','Legacy co-user: all TRAIN (adaptive)'),
        ('price_r2_trim99','First-code price R2 (trim99)'),('first_fulltext_variance_explained','First-code T5 variance explained'),
        ('relative_nll_reduction','Legacy relative NLL reduction'),('code_perplexity_l1','Catalogue first-code perplexity (fixed)'),
        ('semantic_unique_ids','Distinct semantic triples (fixed)'),('max_collision_group','Largest collision group (fixed)')]
    control_path=Path('/workspace/artifacts/ae-ablation-probes/analysis/endpoints.json')
    controls=json.loads(control_path.read_text()) if control_path.exists() else []
    for index in indices:
        fig,axes=plt.subplots(2,4,figsize=(17,8))
        for ax,(key,label) in zip(axes.flat,legacy):
            for trial in trial_ids:
                rr=[r for r in rows if r['index']==index and r['trial']==trial]
                ax.plot([r['proposals'] for r in rr],[r[key] for r in rr],color=COLORS[trial],label=trial,
                        linestyle='--' if trial=='random_matched' else '-',marker='.')
            for variant,style,color in [('gradnorm',':','#222222'),('title_only','-.','#555555')]:
                values=[r[key] for r in controls if r['variant']==variant and key in r]
                if values and key in ['train_proxy_gain_bits','train_prefix_coherence','price_r2_trim99']:
                    ax.axhline(np.mean(values),ls=style,color=color,alpha=.65,label=f'Prior Field {variant} (seed mean)')
            ax.set_title(label);ax.set_xlabel('Proposals')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.01))
        fig.suptitle(f'{index}: earlier RQ-VAE diagnostics; Field lines are contextual references')
        fig.tight_layout(rect=(0,.12,1,.95));fig.savefig(analysis/f'legacy-{index}.png',dpi=160);plt.close(fig)
    # Show fixed-representation dimensionality and decoding cost of reassignment.
    fig,axes=plt.subplots(2,4,figsize=(17,7))
    for i,index in enumerate(indices):
        error='relative_quantization_mse' if index=='t5_rqvae' else 'normalized_residual_mse_l3'
        for ax,key,label in zip(axes[i],['moved_fraction','pca99',error,'train_prefix_entropy_l1'],
            ['Fraction of items moved','Fixed embedding PCA99 dimension',
             'RQ-VAE relative latent MSE' if index=='t5_rqvae' else 'RQ-KMeans normalized residual MSE, level3',
             'TRAIN-weighted first-prefix entropy (can change)']):
            for trial in trial_ids:
                rr=[r for r in rows if r['index']==index and r['trial']==trial]
                ax.plot([r['proposals'] for r in rr],[r[key] for r in rr],label=trial,color=COLORS[trial],marker='.')
            ax.set_title(label);ax.set_xlabel('Proposals')
            if ax is axes[i,0]:ax.set_ylabel(index)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(analysis/'geometry.png',dpi=160);plt.close(fig)
    endpoint=[r for r in rows if r['proposals']==config['search']['proposals']]
    save_json(analysis/'endpoints.json',endpoint)
    delta=[]
    for r in endpoint:
        base=next(a for a in rows if a['index']==r['index'] and a['trial']==r['trial'] and a['proposals']==0)
        delta.append(dict(index=r['index'],trial=r['trial'],accepted=r['accepted'],moved_fraction=r['moved_fraction'],
            **{key+'_delta':r[key]-base[key] for key in ['semantic20_score','fit_cf20_score','heldout_cf20_score',
                'heldout_gain_bits_l1','heldout_gain_bits_l2','heldout_gain_bits_l3','heldout_infonce_loss','train_proxy_gain_bits']}))
    save_json(analysis/'endpoint-deltas.json',delta)
    # Zoom retained-semantic variants without the destructive CF-only/random arms.
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for i,index in enumerate(indices):
        for ax,(key,label) in zip(axes[i],[('semantic20_score','Semantic LCP change'),
                ('heldout_cf20_score','Held-out CF LCP change'),('heldout_gain_bits_l1','Held-out gain change, bits')]):
            for trial in trial_ids:
                if trial in ['cf_only','random_matched']:continue
                rr=[r for r in rows if r['index']==index and r['trial']==trial];base=rr[0][key]
                ax.plot([r['proposals'] for r in rr],[r[key]-base for r in rr],label=trial,color=COLORS[trial],marker='.')
            ax.axhline(0,color='#333333',lw=.7);ax.set_title(label);ax.set_xlabel('Proposals')
            if ax is axes[i,0]:ax.set_ylabel(index)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4)
    fig.suptitle('Changes from each original index; CF-only and random controls omitted for scale')
    fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(analysis/'transfer-zoom.png',dpi=170);plt.close(fig)
    bootstrap_deltas(out,config,users,price,observed,rows)
    for r in delta: print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='/workspace/artifacts/sid-graph-swap');a=p.parse_args()
    with threadpool_limits(limits=2): report(Path(a.output))
