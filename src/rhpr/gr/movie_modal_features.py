"""Prepare shared full-plot embeddings once before the independent fit workers."""
import gc,json,shutil,time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_event
from rhpr.gr.data import file_sha256


def prepare_shared_modalities(config, source, resources, output, device):
    import torch
    from sentence_transformers import SentenceTransformer
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    before=time.monotonic()
    texts=json.loads((source/'plot-texts.json').read_text())
    specification=config.get('plot_encoder')
    model_path=(source/specification['path'] if specification else Path(resources)/'sentence-t5-base')
    limit=specification['max_length'] if specification else 2048
    model=SentenceTransformer(str(model_path),device=device,local_files_only=True)
    model.eval();model.max_seq_length=limit
    tokenizer=model.tokenizer;capacity=limit-tokenizer.num_special_tokens_to_add(pair=False)
    chunks=[];lengths=[]
    for owner,txt in enumerate(texts):
        tokens=tokenizer.encode(txt,add_special_tokens=False,verbose=False) if txt else []
        lengths.append(len(tokens))
        for start in range(0,len(tokens),capacity):
            piece=tokens[start:start+capacity]
            chunks.append((owner,tokenizer.build_inputs_with_special_tokens(piece),len(piece)))
    values=np.zeros((len(texts),768),dtype=np.float64);weight=np.zeros(len(texts))
    with torch.inference_mode():
        for start in range(0,len(chunks),config['representation']['batch_size']):
            rows=chunks[start:start+config['representation']['batch_size']]
            batch=tokenizer.pad({'input_ids':[r[1] for r in rows]},padding=True,return_tensors='pt')
            batch={k:v.to(device) for k,v in batch.items()}
            if batch['input_ids'].shape[1]>limit:raise ValueError('plot chunk exceeds token budget')
            embeddings=model(batch)['sentence_embedding'].float().cpu().numpy()
            for (owner,_,tokens),embedding in zip(rows,embeddings):
                values[owner]+=tokens*embedding;weight[owner]+=tokens
            if start%1024==0:_event('plot_embedding_progress',chunks=min(start+len(rows),len(chunks)),total=len(chunks))
    observed=weight>0;values[observed]/=weight[observed,None]
    norms=np.linalg.norm(values,axis=1,keepdims=True);values[observed]/=norms[observed]
    if not np.isfinite(values).all():raise FloatingPointError('nonfinite plot embeddings')
    del model;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    with np.load(source/'base-features.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays.update(plot=values.astype(np.float32),plot_observed=observed)
    np.savez_compressed(output/'features.npz',**arrays)
    shutil.copy2(source/'source-ids.json',output/'source-ids.json')
    provenance=json.loads((source/'provenance.json').read_text())
    provenance['plot_encoder']={'model':specification or config['representation'],'pooling':f'token-count-weighted mean of nonoverlapping{limit}-token chunks, then L2','truncated_items':0,
        'items':len(texts),'observed':int(observed.sum()),'chunks':len(chunks),'max_document_tokens':max(lengths,default=0),
        'seconds':time.monotonic()-before,'features_sha256':file_sha256(output/'features.npz')}
    _write(output/'provenance.json',provenance)
    return output
