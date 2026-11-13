"""Verified epoch-boundary resume; no fallback to weights-only warm starts."""
import hashlib,json,random,shutil
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_autocast,_index_model_batch,_loss


def array_digest(arrays):
    from rhpr.gr.movie_data import case_digest
    return case_digest(arrays)


def restore_training(model,optimizer,scaler,index,config,train,validation,source,output,device):
    import torch
    source=Path(source);output=Path(output)
    state=torch.load(source/'generator-last.pt',map_location='cpu',weights_only=False)
    previous=state['config'];epoch=state['epoch']
    if previous['training'].get('scheduler') is not None and previous['training']['scheduler']!=config['training'].get('scheduler'):
        raise ValueError('resume changes an existing scheduler policy')
    for key in ['seed','data','generator','sasrec']:
        if previous.get(key)!=config.get(key):raise ValueError(f'resume changes {key}')
    for key in ['learning_rate','weight_decay','batch_size','physical_batch_size','precision','evaluation_precision','grad_clip_norm','sampling','loss_log_key']:
        if previous['training'].get(key)!=config['training'].get(key):raise ValueError(f'resume changes training.{key}')
    if epoch>=config['training']['max_epochs']:raise ValueError('resume budget must extend the saved epoch')
    if state['sid_matrix_sha256']!=hashlib.sha256(index.sids.tobytes()).hexdigest():raise ValueError('resume SID/item map changed')
    for name,examples in [('train',train),('validation',validation)]:
        if state.get(name+'_case_sha256',array_digest(examples))!=array_digest(examples):raise ValueError('resume case schedule changed')
    curves=json.loads((source/'training-curves.json').read_text())
    if ([x['epoch'] for x in curves]!=list(range(1,epoch+1)) or curves[-1]['updates']!=state['updates']
            or any(x['train_examples']!=len(train[2]) for x in curves)):
        raise ValueError('resume curves/updates/case count mismatch')
    best=torch.load(source/'generator-best.pt',map_location='cpu',weights_only=False)
    if best['epoch']!=state['best_epoch'] or best['validation']['ndcg@10']!=state['best_metric']:
        raise ValueError('resume best checkpoint differs from last-checkpoint selection state')
    if max(x['validation']['ndcg@10'] for x in curves if 'validation' in x)!=state['best_metric']:
        raise ValueError('resume best metric differs from validation history')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    if device.startswith('cuda') and len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('resume CUDA RNG device count mismatch')
    selection=json.loads((source/'training-selection.json').read_text()) if (source/'training-selection.json').exists() else {}
    training_seconds=state.get('optimizer_training_seconds',sum(x['train_seconds'] for x in curves))
    validation_seconds=state.get('validation_seconds',selection.get('validation_seconds',sum(x.get('validation',{}).get('seconds',0.) for x in curves)))
    if (source/'generator-best.pt').resolve()!=(output/'generator-best.pt').resolve():shutil.copy2(source/'generator-best.pt',output/'generator-best.pt')
    for path in source.glob('validation-epoch-*.npz'):
        if path.resolve()!=(output/path.name).resolve():shutil.copy2(path,output/path.name)
    _write(output/'training-curves.json',curves)
    from rhpr.gr.sid_study import state_digest
    if state_digest(model.state_dict())!=state_digest(state['model']):raise RuntimeError('resume model transfer mismatch')
    report=dict(source_epoch=epoch,source_updates=state['updates'],best_epoch=state['best_epoch'],best_metric=state['best_metric'],
        model_state_sha256=state_digest(model.state_dict()),optimizer_restored=True,scaler_restored=True,
        cpu_cuda_python_numpy_rng_to_restore=True,sid_matrix_sha256=state['sid_matrix_sha256'],
        train_case_sha256=array_digest(train),validation_case_sha256=array_digest(validation),
        rng_note='RNG restored immediately before first resumed epoch; this does not enforce bitwise deterministic CUDA kernels')
    _write(output/'resume-verification.json',report)
    return dict(epoch=epoch,updates=state['updates'],skipped=state['amp_skipped_updates'],curves=curves,
        best_epoch=state['best_epoch'],best_metric=state['best_metric'],stale=state['stale_validation_checks'],
        training_seconds=training_seconds,validation_seconds=validation_seconds,
        previous_wall_seconds=selection.get('training_seconds',training_seconds+validation_seconds),state=state)


def restore_rng(state,device):
    import torch
    torch.set_rng_state(state['torch_rng'].cpu())
    if device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    if not torch.equal(torch.get_rng_state(),state['torch_rng'].cpu()) or random.getstate()!=state['python_rng']:
        raise RuntimeError('CPU/Python RNG restoration mismatch')
    actual=np.random.get_state();expected=state['numpy_rng']
    if actual[0]!=expected[0] or not np.array_equal(actual[1],expected[1]) or actual[2:]!=expected[2:]:
        raise RuntimeError('NumPy RNG restoration mismatch')
    if device.startswith('cuda') and any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng'])):
        raise RuntimeError('CUDA RNG restoration mismatch')


def build_scheduler(optimizer,spec,state=None,best_metric=-1.,epoch=0,steps_per_epoch=None):
    if not spec:return None
    import torch
    if spec['kind']=='warmup_cosine':
        import math
        warm=spec['warmup_epochs'];total=spec['total_epochs'];minimum=spec['min_lr']
        if (type(steps_per_epoch)is not int or steps_per_epoch<1 or type(warm)is not int
                or type(total)is not int or not 0<warm<total
                or not math.isfinite(minimum)or minimum<=0
                or any(minimum>=g.get('initial_lr',g['lr'])for g in optimizer.param_groups)):
            raise ValueError('invalid warmup/cosine schedule')
        warm_steps=warm*steps_per_epoch;total_steps=total*steps_per_epoch
        def multiplier(step,base):
            if step<warm_steps:return (step+1)/warm_steps
            progress=min(1.,(step-warm_steps+1)/(total_steps-warm_steps))
            floor=minimum/base
            return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
        functions=[lambda step,base=g.get('initial_lr',g['lr']):multiplier(step,base)
                   for g in optimizer.param_groups]
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,functions)
        if state is not None:
            scheduler.load_state_dict(state)
            # LambdaLR construction changes optimizer LR; load_state_dict does
            # not undo that. Restore the actual next-update LR on exact resume.
            for group,lr in zip(optimizer.param_groups,state['_last_lr']):group['lr']=lr
        elif epoch:raise ValueError('step scheduler resume requires its saved state')
        return scheduler
    if spec['kind']!='reduce_on_plateau':raise ValueError('unsupported generator scheduler')
    # User-facing patience5 means reduce on the fifth failed check; Torch uses >patience.
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=spec['factor'],
        patience=spec['patience_epochs']-1,min_lr=spec['min_lr'],threshold=spec['threshold'],threshold_mode='abs')
    if state is not None:scheduler.load_state_dict(state)
    elif epoch:
        scheduler.best=best_metric;scheduler.last_epoch=epoch
    return scheduler


def validation_losses(model,examples,index,batch_size,device,precision):
    import torch
    total=None;count=0;before=model.training
    devices=[torch.device(device).index or 0] if device.startswith('cuda') else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices),torch.inference_mode(),_autocast(precision,device):
            for start in range(0,len(examples[2]),batch_size):
                batch=_index_model_batch(*(v[start:start+batch_size] for v in examples),index,device)
                if hasattr(model,'rhpr_loss'):losses=[_loss(model,batch)]
                else:
                    output=model(**batch)[:,:-1]
                    losses=[torch.nn.functional.cross_entropy(head(output[:,level]),batch['future_ids'][:,level]) for level,head in enumerate(model.decoder.decoder_mlp)]
                values=np.array([float(x) for x in losses],dtype=np.float64)
                if not np.isfinite(values).all():raise FloatingPointError('nonfinite teacher-forced validation loss')
                n=len(examples[2][start:start+batch_size]);total=values*n if total is None else total+values*n;count+=n
        return dict(examples=count,ce_by_level=(total/count).tolist(),total_ce=float(total.sum()/count),
                    objective='full_catalogue_item_ce' if hasattr(model,'rhpr_loss') else 'teacher_forced_sum_of_SID_token_CE',precision=precision)
    finally:model.train(before)
