"""Existing ID-only SASRec on the same strict-past movie next-item cases."""
import time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_autocast
from rhpr.sasrec.model import SASRecModel


class ItemIDIndex:
    prediction_array_name='item_ids'
    def __init__(self,items):
        self.items=items
        # Generic checkpoint bookkeeping only, not a semantic tokenizer.
        self.sids=np.arange(items,dtype=np.int64)[:,None]

    def model_batch(self,contexts,users,targets,device):
        import torch
        x=np.asarray(contexts);valid=x>=0;lengths=valid.sum(1)
        if x.ndim!=2 or np.any(lengths<1) or np.any(x[valid]>=self.items):
            raise ValueError('SASRec needs nonempty valid catalogue histories')
        offsets=np.arange(x.shape[1])[None,:]
        if not np.array_equal(valid,offsets>=x.shape[1]-lengths[:,None]):
            raise ValueError('temporal cases must have contiguous left padding')
        # Existing SASRec expects right padding and ID0 reserved for PAD.
        mask=offsets<lengths[:,None]
        positions=np.minimum(x.shape[1]-lengths[:,None]+offsets,x.shape[1]-1)
        packed=np.where(mask,np.take_along_axis(x,positions,axis=1)+1,0)
        return dict(item_ids=torch.as_tensor(packed,dtype=torch.long,device=device),
                    valid_mask=torch.as_tensor(mask,dtype=torch.bool,device=device),
                    target_item_ids=torch.as_tensor(targets,dtype=torch.long,device=device))


class MovieSASRec(SASRecModel):
    def catalogue_logits(self,batch):
        state=self.last_state(batch['item_ids'],batch['valid_mask'])
        return state @ self.item_embedding.weight[1:].T

    def rhpr_loss(self,batch):
        import torch
        return torch.nn.functional.cross_entropy(self.catalogue_logits(batch),batch['target_item_ids'])

    def recommend(self,batch,k=20):
        return self.catalogue_logits(batch).topk(min(k,self.num_items),dim=-1)

    def rhpr_evaluate(self,examples,index,batch_size,device,precision):
        import torch
        before=time.monotonic();self.eval();predictions=[];scores=[]
        with torch.inference_mode(),_autocast(precision,device):
            for start in range(0,len(examples[2]),batch_size):
                b=index.model_batch(*(v[start:start+batch_size] for v in examples),device)
                result=self.recommend(b)
                if not torch.isfinite(result.values).all():raise FloatingPointError('nonfinite SASRec catalogue scores')
                predictions.append(result.indices.cpu().numpy());scores.append(result.values.float().cpu().numpy())
        p=np.concatenate(predictions);s=np.concatenate(scores);hit=p==examples[2][:,None]
        rank=np.where(hit.any(1),hit.argmax(1)+1,0)
        values=dict(evaluation_examples=len(p),prediction_count=int(p.size),invalid_predictions=0,
                    invalid_prediction_fraction=0.,duplicate_predictions=0,duplicate_prediction_fraction=0.)
        for k in [5,10]:
            valid=(rank>0)&(rank<=k)
            values[f'recall@{k}']=float(valid.mean())
            values[f'ndcg@{k}']=float(np.where(valid,1/np.log2(np.maximum(rank,1)+1),0).mean())
        values['seconds']=time.monotonic()-before
        return values,p,s


def build_movie_sasrec(config,seed,device):
    import torch
    from rhpr.gr.sid_study import state_digest
    torch.manual_seed(seed)
    model=MovieSASRec(num_items=config['data']['items'],max_seq_len=config['data']['history_items'],
        **config['sasrec']).to(device)
    pins=config.get('expected_sasrec_initial_state_sha256_by_seed')
    if pins is not None and pins.get(str(seed))!=state_digest(dict(model.named_parameters())):
        raise ValueError('SASRec initialization differs from pinned control')
    return model


def parameter_counts(model):
    """Deduplicate tied weights; exclude every nn.Embedding on both sides."""
    import torch
    embedding_ids={id(p) for m in model.modules() if isinstance(m,torch.nn.Embedding) for p in m.parameters(recurse=False)}
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
        embedding_parameters=sum(p.numel() for p in model.parameters() if id(p) in embedding_ids),
        active_nonembedding_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None and id(p) not in embedding_ids),
        nonembedding_parameters=sum(p.numel() for p in model.parameters() if id(p) not in embedding_ids),
        inactive_nonembedding_names=[n for n,p in model.named_parameters() if p.grad is None and id(p) not in embedding_ids])


def benchmark_calls(call_factory,examples,device,spec):
    import torch
    results=[]
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize(device)
    with torch.inference_mode():
        for size in spec['batch_sizes']:
            # Allocate/prepare once; exclude CPU input conversion and transfer.
            call=call_factory(size)
            for _ in range(spec['warmup']):call()
            sync();samples=[]
            for _ in range(spec['iterations']):
                sync();before=time.perf_counter();call();sync();samples.append((time.perf_counter()-before)*1000)
            results.append(dict(batch_size=size,median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),
                users_per_second=float(1000*size/np.median(samples))))
    from rhpr.gr.movie_data import case_digest
    return dict(device=device,precision='fp32',warmup=spec['warmup'],iterations=spec['iterations'],measurements=results,
        case_sha256=case_digest(examples),history_width=int(examples[0].shape[1]),
        valid_history_lengths=(examples[0]>=0).sum(1).tolist(),
        scope='model resident; full recommendation: SASRec all-catalogue top20 or TIGER free beam20; excludes input H2D, CPU decoding, offline embeddings/AE/RQ/index construction',
        note='Latency differs even when active parameter counts match; repeated samples are one hardware session.')


def benchmark_examples(users,history):
    from rhpr.gr.movie_data import temporal_examples
    examples=temporal_examples(users,history,'train')
    indices=np.flatnonzero((examples[0]>=0).sum(1)==history)[:32]
    # Only tiny CPU contract fixtures may lack full-length histories.
    if not len(indices):indices=np.arange(min(32,len(examples[2])))
    return tuple(v[indices] for v in examples)


def run_sasrec_trial(trial,config,plan,users,device):
    import torch,random
    from rhpr.gr.movie_data import temporal_examples
    from rhpr.gr.experiment import _train_generator,_evaluate
    from rhpr.gr.sid_study import state_digest
    seed=trial['seed'];config={**config,'seed':seed,'training':{**config['training'],'loss_log_key':'train_loss_item_ce'}}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    model=build_movie_sasrec(config,seed,device);index=ItemIDIndex(config['data']['items'])
    initial=state_digest(dict(model.named_parameters()))
    _write(root/'generator.json',dict(kind='sasrec_id_only',config=config['sasrec'],initial_state_sha256=initial,
        parameters=sum(p.numel() for p in model.parameters()),loss='full-catalogue next-item CE, tied item matrix; no PAD logit, no sampled negatives',user_features=False))
    selection=_train_generator(model,users,index,config,root,device,example_builder=temporal_examples)
    if selection['training_epochs']!=20 or selection['validation_checks']!=4:raise RuntimeError('incomplete SASRec control')
    examples=temporal_examples(users,config['data']['history_items'],'test')
    metrics,predictions,scores=_evaluate(model,examples,index,config['training']['evaluation_batch_size'],device,'fp32')
    np.savez_compressed(root/'test-predictions.npz',item_ids=predictions,scores=scores,user_ids=examples[1],target_item_ids=examples[2])
    bench=benchmark_examples(users,config['data']['history_items'])
    def factory(size):
        ix=np.arange(size)%len(bench[2]);batch=index.model_batch(*(v[ix] for v in bench),device)
        return lambda:model.recommend(batch)
    model.eval();_write(root/'inference-benchmark.json',benchmark_calls(factory,bench,device,config['benchmark']))
    result=dict(trial=trial['id'],stage=trial['stage'],seed=seed,initial_state_sha256=initial,
        initialization_group=f'sasrec_seed{seed}',selection=selection,test=metrics,predictor='sasrec_id_only')
    _write(root/'trial-result.json',result)
    return result
