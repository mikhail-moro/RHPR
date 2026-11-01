"""Continuous modality AE, followed by unchanged native residual KMeans/RQ."""
from pathlib import Path
import time
import numpy as np
from rhpr.gr.experiment import _write,_event
from rhpr.gr.data import file_sha256

NEURAL_MODULES=('encoder_heads','normalization_layer','encoder','decoder','decoder_heads')


def neural_state(model):
    return {k:v for k,v in model.state_dict().items() if k.split('.')[0] in NEURAL_MODULES}


def warmup_autoencoder(model, tensors, spec, seed, device, output, normalization):
    import torch
    from torch.utils.data import DataLoader
    from rhpr.gr.ae_warmup import SpectralPlateau,latent_spectrum
    from rhpr.gr.sid_study import state_digest
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    warm=spec['ae_warmup'];model.to(device)
    parameters=[p for name in NEURAL_MODULES for p in getattr(model,name).parameters()]
    optimizer=model.optimizer(params=parameters)
    initial=state_digest(neural_state(model))
    books_before=state_digest({f'{i}.{k}':v for i,book in enumerate(model.quantization_layer_list) for k,v in book.state_dict().items()})
    if any(book.is_initialized for book in model.quantization_layer_list):raise ValueError('AE warm-up requires untouched codebooks')
    n=len(tensors[next(iter(model.modality_schema))]);rng=torch.Generator().manual_seed(seed)
    loader=DataLoader(range(n),batch_size=spec['batch_size'],shuffle=True,generator=rng,num_workers=0)
    monitor=SpectralPlateau(warm['stopping'],warm['max_steps']);rows=[];updates=0;exposures=0;stopped=False;started=time.monotonic()
    def evaluate(step):
        modes={m:m.training for m in model.modules()};zs=[];sums={k:0. for k in model.active_modalities};counts={k:0 for k in model.active_modalities}
        try:
            model.eval()
            with torch.inference_mode():
                for begin in range(0,n,spec['batch_size']):
                    fields={k:v[begin:begin+spec['batch_size']].to(device) for k,v in tensors.items()}
                    z=model.encode_modalities(fields);_,losses=model.reconstruction(z,fields)
                    if not torch.isfinite(z).all() or any(not torch.isfinite(v) for v in losses.values()):raise FloatingPointError('nonfinite continuous modality AE probe')
                    zs.append(z.cpu().numpy())
                    for name,loss in losses.items():
                        count=int(fields[name+'_observed'].sum());sums[name]+=float(loss)*count;counts[name]+=count
            latents=np.concatenate(zs);spectrum=latent_spectrum(latents,spec['codebook_size'],spec['num_hierarchies'])
            stop=monitor.observe(step,spectrum)
            row={'step':step,'head_mse':{k:sums[k]/counts[k] for k in sums},'latent_spectrum':spectrum,'spectral_plateau':stop}
            rows.append(row);_write(output/'learning-curves.json',rows)
            _event('multimodal_ae_probe',step=step,head_mse=row['head_mse'],active_modes=spectrum['aggregate_nominal_rate']['active_modes'],spectral_plateau=stop)
            return stop,latents
        finally:
            for m,mode in modes.items():m.training=mode
    evaluate(0)
    while updates<warm['max_steps'] and not stopped:
        for indices in loader:
            model.train();fields={k:v[indices].to(device) for k,v in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            # Intentionally bypass the entire residual-quantization forward.
            z=model.encode_modalities(fields);loss,_=model.reconstruction(z,fields)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite continuous modality AE loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,float('inf'))
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite continuous modality AE gradient')
            optimizer.step();updates+=1;exposures+=len(indices)
            if updates%warm['probe_interval']==0 or updates==warm['max_steps']:
                stopped,_=evaluate(updates)
            if stopped or updates==warm['max_steps']:break
    _,latents=evaluate(updates) if rows[-1]['step']!=updates else (False,None)
    if latents is None:
        modes={m:m.training for m in model.modules()}
        model.eval()
        with torch.inference_mode():
            latents=np.concatenate([model.encode_modalities({k:v[b:b+spec['batch_size']].to(device) for k,v in tensors.items()}).cpu().numpy() for b in range(0,n,spec['batch_size'])])
        for m,mode in modes.items():m.training=mode
    books_after=state_digest({f'{i}.{k}':v for i,book in enumerate(model.quantization_layer_list) for k,v in book.state_dict().items()})
    if books_before!=books_after or any(book.is_initialized for book in model.quantization_layer_list):raise RuntimeError('continuous AE changed codebooks')
    stop={'reason':'spectral_plateau' if stopped else 'max_steps','actual_steps':updates,'max_steps':warm['max_steps'],'criterion':warm['stopping']}
    state=neural_state(model);after=state_digest(state)
    torch.save({'model':state,'optimizer':optimizer.state_dict(),'seed':seed,'steps':updates,'spec':spec,
        'normalization':normalization,'quantization_enabled':False,'stopping':stop,
        'data_rng':rng.get_state(),'rng_cpu':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []},output/'autoencoder.pt')
    np.save(output/'latent-embeddings.npy',latents,allow_pickle=False)
    report={'seed':seed,'steps':updates,'item_exposures':exposures,'stopping':stop,'quantization_enabled':False,
        'initial_neural_state_sha256':initial,'neural_state_sha256':after,'codebooks_unchanged':True,
        'final_head_mse':rows[-1]['head_mse'],'seconds':time.monotonic()-started,
        'checkpoint_sha256':file_sha256(output/'autoencoder.pt'),
        'transfer':'same neural modules/weights/BatchNorm buffers retained; fresh native RQ optimizer; native residual KMeans initialization in eval mode',
        'spectral_criterion_note':'same Field-Aware heuristic; nominal sum24bit reverse-water-filling is not a proven recommendation-quality criterion'}
    _write(output/'autoencoder.json',report)
    model.warmup_transferred=True;model.ae_anchor_latents=torch.as_tensor(latents,device=device)
    return report
