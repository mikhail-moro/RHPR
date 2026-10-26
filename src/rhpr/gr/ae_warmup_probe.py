"""Fixed AE stopping probes: CPU RQ-KMeans diagnostics, no generator or test."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.study_parallel import atomic_json


def trial_settings(config, trial):
    overrides = trial.get('quantizer_overrides', {})
    if set(overrides) - {'latent_dim', 'neural_learning_rate', 'freeze_batchnorm_running_stats'}:
        raise ValueError('unsupported AE ablation override')
    spec = {**config['quantizer'], **overrides, 'loss_policy': trial['loss_policy']}
    kmeans = {**config['kmeans'], 'latent_dim': spec['latent_dim']}
    return spec, kmeans


def validate_probe(config):
    from rhpr.gr.ae_suite import assignments
    steps = config['quantizer']['snapshot_steps']
    maximum = config['quantizer']['max_steps']
    if (not isinstance(steps, list) or not steps
            or any(type(s) is not int or not 0 <= s <= maximum for s in steps)
            or steps != sorted(set(steps)) or steps[-1] != maximum):
        raise ValueError('snapshot_steps must be ordered unique nonnegative steps ending at max_steps')
    if ('warmup_stopping' in config['quantizer'] or 'warmup' in config
            or config['quantizer']['kind'] != 'continuous_field_autoencoder'
            or config['kmeans']['kind'] != 'ae_residual_kmeans'
            or any(k in config for k in ['generator', 'evaluation'])):
        raise ValueError('fixed AE probes have no adaptive stopping, generator or retrieval evaluation')
    seeds = [t['seed'] for t in config['trials']]
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('probe needs exactly two independent AE pipeline seeds')
    if config.get('ablation_suite'):
        variants = {t['variant'] for t in config['trials']}
        if len(set(seeds)) != 2 or not config.get('compare_initialization_by_shape'):
            raise ValueError('ablation needs two shared seeds and shape-aware initialization checks')
        for variant in variants:
            pair = [t for t in config['trials'] if t['variant'] == variant]
            if len(pair) != 2 or len({t['seed'] for t in pair}) != 2:
                raise ValueError('each ablation needs the same two seeds')
            a, b = (trial_settings(config, t) for t in pair)
            if a != b:
                raise ValueError('paired ablation settings must match across seeds')
    else:
        if len(config['trials']) != 2 or len(set(seeds)) != 2:
            raise ValueError('probe needs exactly two independent AE pipeline seeds')
        if any(t['loss_policy'] != config['quantizer']['loss_policy'] for t in config['trials']):
            raise ValueError('probe objectives must be identical across seeds')
    for k in ['num_hierarchies', 'codebook_size', 'latent_dim']:
        if config['quantizer'][k] != config['kmeans'][k]:
            raise ValueError('AE and KMeans dimensions differ')
    assignments(config, 2 if config['training']['device'] == 'cuda' else 0)


def run_worker(plan_path, slot):
    import numpy as np
    import torch
    from rhpr.gr.data import load_beauty
    from rhpr.gr.field_autoencoder import fit_field_autoencoder
    from rhpr.gr.ae_rqkmeans import fit_residual_kmeans
    from rhpr.gr.codebook_diagnostics import diagnose_codebooks
    from rhpr.gr.index_proxies import IndexProxies

    plan = json.loads(Path(plan_path).read_text());config = plan['config']
    assignment = plan['assignments'][slot];root = Path(plan['output'])
    torch.set_num_threads(config['execution']['cpu_threads_per_worker'])
    device = 'cpu'
    if assignment['device'] != 'cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('probe worker must see exactly its assigned GPU')
        device = 'cuda:0';torch.cuda.set_device(0)
    status = dict(slot=slot,status='running',results={},failures=[],device=assignment['device'])
    status_path = root/'workers'/f'worker-{slot}.json';atomic_json(status_path,status)
    texts,users,_,_=load_beauty(Path(plan['resources'])/'data',config['data'])
    with np.load(plan['parent_fields'],allow_pickle=False) as f:
        titles,observed=f['title'],f['title_observed']
    for trial in assignment['trials']:
        name=trial['id'];directory=root/'trials'/name;directory.mkdir(parents=True)
        _event('warmup_probe_trial_start',trial=name,seed=trial['seed'])
        try:
            spec,kmeans=trial_settings(config,trial)
            if trial.get('expected_initial_state_sha256'):
                spec['expected_initial_state_sha256']=trial['expected_initial_state_sha256']
            _,report,_=fit_field_autoencoder(texts,config['representation'],spec,trial['seed'],device,
                plan['source'],plan['recipe'],directory,title_embeddings=titles,title_observed=observed)
            if report['steps']!=spec['max_steps']:raise RuntimeError('AE ended before fixed budget')
            with np.load(directory/'catalogue-fields.npz',allow_pickle=False) as f:
                proxies=IndexProxies(users,f['price'][:,0],f['price_observed'])
            rows=[]
            for step in spec['snapshot_steps']:
                snapshot=directory/'checkpoints'/f'step-{step:05d}'
                ae=json.loads((snapshot/'autoencoder.json').read_text())
                z=np.load(snapshot/'latent-embeddings.npy',allow_pickle=False)
                if ae['steps']!=step or ae['seed']!=trial['seed'] or ae['quantization_enabled'] is not False:
                    raise ValueError('AE snapshot identity mismatch')
                if file_sha256(snapshot/'autoencoder.pt')!=ae['checkpoint_sha256']:
                    raise ValueError('AE snapshot checkpoint digest mismatch')
                codes,books,fit=fit_residual_kmeans(z,kmeans,trial['seed'])
                np.save(snapshot/'semantic-codes.npy',codes,allow_pickle=False)
                np.save(snapshot/'codebooks.npy',books,allow_pickle=False)
                diagnostics=diagnose_codebooks(codes,z,books)
                _write(snapshot/'kmeans.json',{**fit,'diagnostics':diagnostics})
                row=dict(seed=trial['seed'],step=step,
                    latent_dimension99=ae['latent_spectrum']['effective_dimension_99pct'],
                    active_modes_24bits=ae['latent_spectrum']['aggregate_nominal_rate']['active_modes'],
                    reconstruction=ae['field_reconstruction'],
                    checkpoint_sha256=ae['checkpoint_sha256'],latents_sha256=file_sha256(snapshot/'latent-embeddings.npy'),
                    codes_sha256=file_sha256(snapshot/'semantic-codes.npy'),
                    books_sha256=file_sha256(snapshot/'codebooks.npy'),
                    **proxies.first(codes[:,0]))
                if config.get('ablation_suite'):
                    row.update(trial=name,variant=trial['variant'],latent_dim=spec['latent_dim'],
                        loss_policy=trial['loss_policy'],parameters=report['parameters'])
                row.update(semantic_unique_ids=diagnostics['semantic_unique_ids'],
                    max_collision_group=diagnostics['max_collision_group'],
                    relative_quantization_mse=diagnostics['quantization_mse_over_mean_predictor_mse'])
                rows.append(row);_write(directory/'probe-curves.json',rows)
                _event('warmup_probe_index_complete',trial=name,
                    **{k:v for k,v in row.items() if k not in {'reconstruction','trial'}})
            result=dict(seed=trial['seed'],steps=report['steps'],initial_state_sha256=report['initial_state_sha256'],
                probes=len(rows),elapsed_seconds=report['elapsed_seconds'],rows=rows)
            if config.get('compare_initialization_by_shape'):
                result['initialization_group']=f"{trial['seed']}:latent{spec['latent_dim']}"
                result['resolved_quantizer']=spec
            status['results'][name]=result
            _event('warmup_probe_trial_complete',trial=name,probes=len(rows))
        except Exception as error:
            (directory/'failure.log').write_text(traceback.format_exc())
            failure=dict(trial=name,exception_type=type(error).__name__,message=str(error))
            status['failures'].append(failure);_event('warmup_probe_trial_failed',**failure)
        atomic_json(status_path,status);gc.collect()
        if device!='cpu':torch.cuda.empty_cache()
    status['status']='complete';atomic_json(status_path,status)


def plot_curves(rows, output, controls=None):
    import csv
    columns=['seed','step','latent_dimension99','active_modes_24bits','price_r2_trim99',
        'train_proxy_gain_bits','train_prefix_coherence','first_codes_used',
        'trial','variant','latent_dim','parameters','semantic_unique_ids','max_collision_group','relative_quantization_mse']
    with (output/'probe-curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        # Plotting is not in the pinned neural wheel set. Preserve completed
        # numerical evidence if the host image has no plotting stack; the same
        # function renders it locally after collection without another fit.
        _write(output/'plot-status.json',dict(rendered=False,reason=str(error)))
        return
    if any('variant' in r for r in rows):
        plot_ablation_curves(rows, output, controls, plt)
        return
    metrics=[('latent_dimension99','Latent PCA dimension (99%)'),('price_r2_trim99','First-code price R² (trim top1%)'),
        ('train_proxy_gain_bits','TRAIN previous-item gain (bits vs own unigram)'),
        ('train_prefix_coherence','TRAIN co-user cosine within first-code groups')]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(key,title) in zip(axes.flat,metrics):
        for seed in sorted({r['seed'] for r in rows}):
            values=[r for r in rows if r['seed']==seed and r[key] is not None]
            ax.plot([r['step'] for r in values],[r[key] for r in values],marker='o',ms=4,label=f'seed{seed}')
        ax.set(title=title,xlabel='AE updates before fixed RQ-KMeans');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed AE stopping probes: same raw-latent RQ-KMeans3×256 at each checkpoint\nCatalogue/TRAIN proxies; no generator, validation or test metrics',fontsize=12)
    fig.tight_layout();fig.savefig(output/'warmup-proxies.png',dpi=170);plt.close(fig)
    _write(output/'plot-status.json',dict(rendered=True))


def plot_ablation_curves(rows, output, controls, plt):
    """One panel per intervention; keep fixed-AE/T5 references on every panel."""
    from math import ceil
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    metrics = [('latent_dimension99','Latent PCA dimension (99%)'),
        ('price_r2_trim99','First-code trimmed price R²'),
        ('train_proxy_gain_bits','TRAIN previous-item gain (bits)'),
        ('train_prefix_coherence','TRAIN co-user coherence'),
        ('semantic_unique_ids','Distinct semantic triples'),
        ('relative_quantization_mse','Relative quantization MSE')]
    for key,title in metrics:
        fig,axes=plt.subplots(ceil(len(variants)/3),3,figsize=(15,4*ceil(len(variants)/3)),squeeze=False)
        for ax,variant in zip(axes.flat,variants):
            for seed in sorted({r['seed'] for r in rows}):
                values=[r for r in rows if r['variant']==variant and r['seed']==seed and r[key] is not None]
                ax.plot([r['step'] for r in values],[r[key] for r in values],marker='o',ms=3,label=f'seed{seed}')
            if controls:
                old=controls['ae_curves'];steps=sorted({r['step'] for r in old})
                low=[min(r[key] for r in old if r['step']==s) for s in steps]
                high=[max(r[key] for r in old if r['step']==s) for s in steps]
                ax.fill_between(steps,low,high,color='gray',alpha=.22,label='Fixed-normalized AE seeds42/43')
                ax.plot(steps,[(a+b)/2 for a,b in zip(low,high)],color='gray',ls=':',lw=1)
                ref=controls['embedding_reference']['metrics'].get(key)
                if ref is not None:ax.axhline(ref,color='#8751a5',ls='--',label='T5 → RQ-VAE final')
            ax.set(title=variant,xlabel='AE updates');ax.set_xscale('symlog',linthresh=100);ax.set_xlim(-10,6000)
            if key=='relative_quantization_mse':ax.set_yscale('log')
            ax.grid(alpha=.2);ax.legend(fontsize=6)
        for ax in list(axes.flat)[len(variants):]:ax.set_visible(False)
        fig.suptitle(title+'\nOne intervention per panel; fixed stopping probes. No generator or retrieval scoring.',fontsize=12)
        fig.tight_layout();fig.savefig(output/f'ablation-{key}.png',dpi=160);plt.close(fig)
    _write(output/'plot-status.json',dict(rendered=True,ablation_panels=len(variants)))


def run_warmup_probe(config, output):
    import numpy as np
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import load_beauty
    from rhpr.gr.resources import verify_resources,verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.offline_runtime import resolve_dataset_directory

    validate_probe(config);started=time.monotonic();output=Path(output).resolve()
    jobs=assignments(config,torch.cuda.device_count())
    resources=resolve_dataset_directory(config['resources']['path'])
    manifest=verify_resources(resources,config['resources']['manifest_sha256'])
    parent=verify_parent_artifacts(config['parent'])
    controls=None
    if config.get('controls_parent'):
        from rhpr.gr.generator_suite import verify_file
        control_root=verify_parent_artifacts(config['controls_parent'])
        control_manifest=json.loads((control_root/'run-manifest.json').read_text())
        if control_manifest['config']['data']['sha256']!=config['data']['sha256']:
            raise ValueError('AE control data differs from the current catalogue')
        controls=dict(ae_curves=json.loads((control_root/'probe-curves.json').read_text()),
            embedding_reference=config['embedding_reference'])
        for row in controls['ae_curves']:
            key=f"{row['seed']}:{row['step']}"
            path=verify_file(control_root,config['control_diagnostics'][key])
            report=json.loads(path.read_text())['diagnostics']
            row.update(semantic_unique_ids=report['semantic_unique_ids'],
                relative_quantization_mse=report['quantization_mse_over_mean_predictor_mse'])
        _write(output/'comparison-controls.json',controls)
    texts,_,ids,counts=load_beauty(resources/'data',config['data'])
    with np.load(parent/'catalogue-fields.npz',allow_pickle=False) as f:
        if f['title'].shape!=(len(ids),config['representation'].get('title',{}).get('dimension',768)) or not np.isfinite(f['title']).all():
            raise ValueError('pinned title targets do not match catalogue')
    scratch=Path(config['resources']['scratch_path'])
    original=unpack_grid(resources/'grid-source.bin',scratch/'original')
    source=prepare_grid_source(original,scratch/'runtime')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    (output/'workers').mkdir(parents=True)
    _write(output/'input-provenance.json',dict(manifest=manifest,counts=counts,original_item_ids=ids,
        parent=config['parent'],signals='catalogue metadata and train histories only; no validation/test targets',
        checkpoint_selection='fixed steps, no PCA stopping or quality selection',generator_trained=False))
    plan=dict(config=config,assignments=jobs,output=str(output),resources=str(resources),source=str(source),
        recipe=str(original/'configs/experiment/rqvae_train_flat.yaml'),parent_fields=str(parent/'catalogue-fields.npz'))
    path=output/'workers/plan.json';atomic_json(path,plan)
    summary=launch_workers(path,worker_module='rhpr.gr.ae_warmup_probe',event_prefix='warmup_probe')
    rows=[row for r in summary['results'].values() for row in r['rows']]
    _write(output/'probe-curves.json',rows);plot_curves(rows,output,controls)
    return dict(trials=len(summary['results']),parallel_workers=2,generator_trained=False,test_evaluations=0,
        validation_evaluations=0,ae_max_steps=config['quantizer']['max_steps'],index_probes=len(rows),
        catalogue_items=len(texts),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--slot',type=int,required=True)
    a=parser.parse_args();run_worker(a.plan,a.slot)
