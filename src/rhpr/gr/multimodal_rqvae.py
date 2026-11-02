"""Fixed catalogue normalization and summed per-channel reconstruction losses."""
from pathlib import Path
import time
import numpy as np
from rhpr.gr.experiment import _write, _event
from rhpr.gr.data import file_sha256


def normalize_modalities(values, observed, columns=None):
    result, statistics = {}, {}
    count = None
    for name, raw in values.items():
        x = np.asarray(raw, dtype=np.float32)
        mask = np.asarray(observed[name], dtype=bool)
        if (x.ndim != 2 or mask.shape != (len(x),) or not mask.any()
                or not np.isfinite(x).all() or (count is not None and len(x) != count)):
            raise ValueError('invalid modality values or observation mask')
        count = len(x)
        if columns and name in columns:
            selected = columns[name]
            if (not selected or len(set(selected)) != len(selected)
                    or any(type(c) is not int or c < 0 or c >= x.shape[1] for c in selected)):
                raise ValueError('invalid normalization columns')
            reduced, stats = normalize_modalities({name:x[:, selected]}, {name:mask})
            y = np.zeros_like(x); y[:, selected] = reduced[name]
            result[name] = y
            statistics[name] = dict(stats[name], selected_columns=list(selected),
                original_dimension=x.shape[1], inactive_input_zero=True)
            continue
        mean = x[mask].mean(0, dtype=np.float64)
        scale = float(np.sqrt(np.square(x[mask].astype(np.float64) - mean).mean()))
        if not np.isfinite(scale) or scale < 1e-8:
            raise ValueError('modality has no reconstruction variance')
        y = ((x - mean) / scale).astype(np.float32)
        y[~mask] = 0
        result[name] = y
        statistics[name] = {'mean': mean.tolist(), 'scale': scale, 'observed_items': int(mask.sum()),
            'zero_predictor_mse': float(np.square(y[mask]).mean())}
    return result, statistics


def fit_multimodal_rqvae(values, masks, spec, seed, device, source, recipe, output, active):
    import torch
    from lightning.pytorch import Callback, Trainer
    from lightning.pytorch.trainer.states import TrainerFn
    from torch.utils.data import DataLoader
    from rhpr.gr.rqvae import build_rqvae
    from rhpr.gr.codebook_diagnostics import semantic_statistics
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    arrays, normalization = normalize_modalities(values, masks, spec.get("reconstruction_columns"))
    _write(output / 'modality-normalization.json', normalization)
    torch.manual_seed(seed)
    schema = {k: v.shape[1] for k, v in arrays.items()}
    model, resolved = build_rqvae(source, recipe, len(schema) * spec['head_dim'], spec,
        modality_schema=schema, active_modalities=active)
    from rhpr.gr.sid_study import state_digest
    initial = state_digest(dict(model.named_parameters()))
    expected=spec.get('expected_initial_state_sha256',initial)
    if 'expected_initial_state_sha256_by_seed' in spec:
        if str(seed) not in spec['expected_initial_state_sha256_by_seed']:
            raise ValueError('quantizer initialization not pinned for the trial seed')
        expected=spec['expected_initial_state_sha256_by_seed'][str(seed)]
    if expected!=initial:
        raise ValueError('initial quantizer weights differ from the declared direct-training control')
    from src.data.loading.components.interfaces import ItemData
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    tensors.update({k + '_observed': torch.as_tensor(v, dtype=torch.bool) for k, v in masks.items()})
    warmup=None
    if spec.get('ae_warmup'):
        from rhpr.gr.multimodal_warmup import warmup_autoencoder
        tensors['_item_indices']=torch.arange(len(next(iter(arrays.values()))))
        devices=[torch.device(device).index or 0] if device.startswith('cuda') else []
        # Keep native initializer randomness independent of AE duration/probes.
        with torch.random.fork_rng(devices=devices):
            warmup=warmup_autoencoder(model,tensors,spec,seed,device,output/'ae-warmup',normalization)
    def collate(indices):
        return ItemData(transformed_features={k: v[indices] for k, v in tensors.items()})
    n = len(next(iter(arrays.values())))
    if n < max(spec['codebook_size'], spec['init_buffer_size']) or n % spec['batch_size'] == 1:
        raise ValueError('catalogue incompatible with native quantizer initialization/BatchNorm')
    loader = DataLoader(list(range(n)), batch_size=spec['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate, num_workers=0)
    def export():
        modes = {m: m.training for m in model.modules()};fn = model.trainer.state.fn
        try:
            model.to(device).eval();model.trainer.state.fn = TrainerFn.PREDICTING
            codes=[];sums={k:0. for k in active};counts={k:0 for k in active}
            with torch.inference_mode():
                for start in range(0,n,spec['batch_size']):
                    f={k:v[start:start+spec['batch_size']].to(device) for k,v in tensors.items()}
                    z=model.encode_modalities(f);c,_,q,_=model(z);_,loss=model.reconstruction(q,f)
                    if not torch.isfinite(z).all() or any(not torch.isfinite(v) for v in loss.values()):
                        raise FloatingPointError('nonfinite multimodal export')
                    codes.append(c.cpu().numpy())
                    for k,v in loss.items():
                        m=int(f[k+'_observed'].sum());sums[k]+=float(v)*m;counts[k]+=m
            return np.concatenate(codes),{k:sums[k]/counts[k] for k in active}
        finally:
            model.trainer.state.fn=fn
            for m,mode in modes.items():m.training=mode
    class Progress(Callback):
        def __init__(self):self.last_loss=None;self.rows=[];self.last_step=-1;self.probe_seconds=0.
        def on_train_start(self,trainer,module):
            if warmup is None:return
            from rhpr.gr.multimodal_warmup import neural_state
            if state_digest(neural_state(module))!=warmup['neural_state_sha256']:
                raise RuntimeError('AE parameters or buffers changed before RQ training')
            optimizer=trainer.optimizers[0]
            if not isinstance(optimizer,torch.optim.Adagrad):
                raise RuntimeError('native RQ optimizer must remain Adagrad')
            for state in optimizer.state.values():
                if float(state['step'])!=0 or torch.count_nonzero(state['sum']):
                    raise RuntimeError('RQ must start with fresh optimizer accumulators')
            warmup['fresh_rq_optimizer_verified']=True
            warmup['neural_transfer_verified']=True
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss=outputs['loss'] if isinstance(outputs,dict) else outputs
            if loss is None or not torch.isfinite(loss).all():raise FloatingPointError('nonfinite multimodal RQ loss')
            self.last_loss=float(loss.detach());step=trainer.global_step
            if step and step!=self.last_step and step%spec.get('diagnostics_interval_steps',500)==0:
                before=time.monotonic();self.last_step=step
                codes,losses=export();stats=semantic_statistics(codes,spec['codebook_size'])
                row={'step':step,'training_total':self.last_loss,'reconstruction_mse_by_head':losses,
                    'weighted_reconstruction_mse':sum(model.reconstruction_weights[k]*v for k,v in losses.items()),
                    'training_quantization_loss':module.last_quantization_loss,
                    'training_reconstruction_loss':module.last_reconstruction_loss,
                    'semantic_unique_ids':stats['semantic_unique_ids'],'max_collision_group':stats['max_collision_group'],
                    'used_codes_per_level':[len(np.unique(codes[:,i])) for i in range(codes.shape[1])]}
                self.rows.append(row);_write(output/'codebook-learning-curves.json',self.rows)
                _event('multimodal_rqvae_progress',**row);self.probe_seconds+=time.monotonic()-before
    progress=Progress();model.log_dict=lambda *a,**kw:None
    trainer=Trainer(accelerator='gpu' if device.startswith('cuda') else 'cpu',
        devices=[torch.device(device).index or 0] if device.startswith('cuda') else 1,
        precision='32-true',max_steps=spec['max_steps'],max_epochs=-1,logger=False,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
        num_sanity_val_steps=0,limit_val_batches=0,callbacks=[progress],default_root_dir=str(output))
    started=time.monotonic();trainer.fit(model,train_dataloaders=[loader])
    if trainer.global_step!=spec['max_steps'] or not all(l.is_initialized for l in model.quantization_layer_list):
        raise RuntimeError('multimodal RQ did not complete initialized fixed-budget training')
    torch.save({'model':model.state_dict(),'optimizer':trainer.optimizers[0].state_dict(),
        'seed':seed,'spec':spec,'active_modalities':active,'schema':schema,'normalization':normalization,
        'resolved_model':resolved,'steps':trainer.global_step,'warmup':warmup},output/'quantizer.pt')
    codes,losses=export();stats=semantic_statistics(codes,spec['codebook_size'])
    books=np.stack([layer.centroids.detach().cpu().numpy() for layer in model.quantization_layer_list])
    np.savez_compressed(output/'codebooks-final.npz',codebooks=books,
        counts=np.stack([np.bincount(codes[:,i],minlength=spec['codebook_size']) for i in range(codes.shape[1])]))
    np.save(output/'semantic-codes.npy',codes,allow_pickle=False);_write(output/'semantic-code-diagnostics.json',stats)
    report={'kind':'grid_multimodal_rqvae','seed':seed,'steps':trainer.global_step,'config':spec,
        'quantizer_initial_state_sha256':initial,
        'warmup':warmup,'initialization_latent_audit':model.initial_latent_audit,
        'reconstruction_batches':model.reconstruction_updates,
        'batches_without_reconstruction':model.initialization_batches,
        'active_modalities':active,'modality_dimensions':schema,'head_reconstruction_mse':losses,
        'reconstruction_weights':model.reconstruction_weights,
        'precision':'fp32','quantization_loss_weight':model.quantization_loss_weight,
        'reconstruction_loss_weight':model.reconstruction_loss_weight,
        'reconstruction_loss':'weighted sum of observed-item, dimension-mean MSEs; fixed catalogue normalization; no UW',
        'semantic_code_statistics':stats,'parameters':sum(p.numel() for p in model.parameters()),
        'fit_seconds':time.monotonic()-started,'diagnostic_seconds':progress.probe_seconds,
        'checkpoint_sha256':file_sha256(output/'quantizer.pt'),'resolved_model':resolved}
    _write(output/'quantizer.json',report)
    return codes,report
