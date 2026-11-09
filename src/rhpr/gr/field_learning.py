"""ML1M feature-index learning curves, with fixed held-out cases and paired RNG."""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest

FIELDS = ('title', 'genres', 'duration', 'shuffled_duration')
TOKENIZER_SEEDS = (42, 101, 202)
GENERATOR_SEEDS = (31415, 27182, 161803)
_METADATA_PATTERN = re.compile(r'Title: (.*); Genres: (.*); Year: (.*); Runtime: (.*); Keywords: (.*)[.]')


def make_texts(metadata, shuffle_seed):
    """Extract pinned fields and permute only observed durations across items."""
    parsed = []
    for row in metadata:
        match = _METADATA_PATTERN.fullmatch(row)
        if not match:
            raise ValueError('metadata text does not match the pinned source template')
        title, genres, year, duration, keywords = match.groups()
        if not title or not genres or not year or not keywords:
            raise ValueError('missing expected metadata field')
        if duration != 'unknown' and not re.fullmatch(r'[0-9]+ minutes', duration):
            raise ValueError('unrecognized duration')
        parsed.append((title, genres, duration))
    if not parsed:
        raise ValueError('empty catalogue')
    durations = [row[2] for row in parsed]
    present = np.flatnonzero(np.asarray(durations) != 'unknown')
    owner = np.arange(len(parsed))
    owner[present] = np.random.default_rng(shuffle_seed).permutation(present)
    shuffled = [durations[i] for i in owner]
    if collections.Counter(shuffled) != collections.Counter(durations):
        raise AssertionError('duration shuffle changed the value distribution')
    texts = {k: [] for k in FIELDS}
    for (title, genres, duration), swapped in zip(parsed, shuffled):
        prefix = f'Title: {title}'
        texts['title'].append(prefix + '.')
        texts['genres'].append(prefix + f'; Genres: {genres}.')
        texts['duration'].append(prefix + f'; Runtime: {duration}.')
        texts['shuffled_duration'].append(prefix + f'; Runtime: {swapped}.')
    return texts, owner, dict(items=len(parsed), unknown=durations.count('unknown'),
        mapped_to_other_item=int(np.sum(owner != np.arange(len(parsed)))),
        changed_value=sum(a != b for a, b in zip(durations, shuffled)))


def expected_trials():
    rows = [dict(id=f'{field}_q{q}_g31415', field=field, tokenizer_seed=q, seed=31415,
                 index_id=f'{field}_q{q}', architecture='narrow')
            for q in TOKENIZER_SEEDS for field in FIELDS]
    rows += [dict(id=f'{field}_q42_g{g}', field=field, tokenizer_seed=42, seed=g,
                  index_id=f'{field}_q42', architecture='narrow')
             for g in GENERATOR_SEEDS[1:] for field in ('title', 'genres')]
    return rows


def validate(c):
    g, t, q, d, e = (c[k] for k in ('generator','training','quantizer','data','evaluation'))
    required_training = dict(device='cuda',batch_size=512,physical_batch_size=512,
        learning_rate=.001,weight_decay=.0001,precision='fp16',evaluation_precision='fp32',
        min_epochs=30,max_epochs=100,patience_epochs=30,validation_interval=10,
        validate_first_epoch=False,evaluation_batch_size=32,sampling='precomputed_raw_clean_windows')
    required_generator = dict(encoder_layers=4,decoder_layers=4,d_model=64,d_ff=512,d_kv=32,
        num_heads=6,attn_implementation='sdpa',beam_width=20,should_check_prefix=False,
        num_user_bins=2000,codebook_size=256)
    required_quantizer = dict(kind='grid_rqvae',num_hierarchies=3,codebook_size=256,
        max_steps=3000,batch_size=2048,init_buffer_size=2874,initialization_steps=1000,
        distance_backend='torch_cdist')
    if (c.get('kind') != 'gr_ml1m_field_learning' or c.get('tokenizer_seeds') != list(TOKENIZER_SEEDS)
            or c.get('generator_seeds') != list(GENERATOR_SEEDS) or c.get('shuffle_seed') != 20261109
            or c.get('trials') != expected_trials() or c.get('compare_initialization_by_seed') is not True
            or any(t.get(k) != v for k,v in required_training.items())
            or any(g.get(k) != v for k,v in required_generator.items())
            or any(q.get(k) != v for k,v in required_quantizer.items())
            or d.get('items') != 2874 or d.get('users') != 2786 or d.get('train_examples') != 151166
            or d.get('history_items') != 20 or d.get('split') != 'raw_contiguous_20_val_test'
            or e.get('test_after_selection') is not False or e.get('save_validation_predictions') is not True
            or e.get('save_validation_losses') is not True or t.get('max_seconds',0) < 5000
            or c.get('suite_deadline_seconds',0) < 35000 or 'scheduler' in t
            or c['execution']['worker_devices'] != ['cuda:0','cuda:1']
            or c['execution']['cpu_threads_per_worker'] != 1):
        raise ValueError('field-learning experiment differs from the registered protocol')
    if set(c['initialization_pins']['narrow']) != {str(s) for s in GENERATOR_SEEDS}:
        raise ValueError('missing generator initial state pins')


def prepare_texts(c, data, output):
    if file_sha256(Path(data)/'metadata-texts.json') != c['metadata_texts_sha256']:
        raise ValueError('metadata source hash mismatch')
    metadata = json.loads((Path(data)/'metadata-texts.json').read_text())
    ids = json.loads((Path(data)/'source-ids.json').read_text())
    if len(ids) != c['data']['items'] or len(metadata) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('catalogue alignment mismatch')
    texts, owner, report = make_texts(metadata, c['shuffle_seed'])
    if report['unknown'] != 5 or report['changed_value'] < 2400:
        raise ValueError('unexpected missing or unchanged duration permutation')
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    np.save(output/'duration-owner.npy',owner,allow_pickle=False)
    for name, rows in texts.items():
        _write(output/f'{name}-texts.json',rows)
    report.update(shuffle_seed=c['shuffle_seed'],source_metadata_sha256=c['metadata_texts_sha256'],
        text_sha256={name:file_sha256(output/f'{name}-texts.json') for name in FIELDS},
        owner_sha256=file_sha256(output/'duration-owner.npy'))
    _write(output/'text-provenance.json',report)
    return ids, texts, report


def prepare_indices(c, texts, resources, source, original, output, device):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.rqvae import fit_grid_rqvae
    from rhpr.gr.sid import SidIndex
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    encoder=SentenceTransformer(str(Path(resources)/'sentence-t5-base'),device=device,local_files_only=True)
    encoder.max_seq_length=2048;encoder.eval();refs={}
    for name in FIELDS:
        started=time.monotonic(); rows=texts[name]
        lengths=[len(encoder.tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in rows]
        if max(lengths)>2048:raise ValueError('unexpected truncated metadata')
        vectors=np.asarray(encoder.encode(rows,batch_size=8,convert_to_numpy=True,show_progress_bar=False),np.float32)
        if vectors.shape!=(len(rows),768) or not np.isfinite(vectors).all():raise ValueError('invalid text vectors')
        np.save(root/f'{name}-embeddings.npy',vectors,allow_pickle=False)
        _write(root/f'{name}-encoder.json',dict(items=len(rows),max_tokens=max(lengths),
            embedding_sha256=file_sha256(root/f'{name}-embeddings.npy'),seconds=time.monotonic()-started))
        for seed in TOKENIZER_SEEDS:
            idx=f'{name}_q{seed}';work=root/idx;work.mkdir()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            _event('field_index_start',index=idx)
            codes,report=fit_grid_rqvae(vectors,c['quantizer'],seed,device,source,
                Path(original)/'configs/experiment/rqvae_train_flat.yaml',work)
            index=SidIndex(codes,width=256);np.save(root/f'{idx}-semantic-ids.npy',index.sids,allow_pickle=False)
            refs[idx]=dict(sids_sha256=file_sha256(root/f'{idx}-semantic-ids.npy'),
                embedding_sha256=file_sha256(root/f'{name}-embeddings.npy'),field=name,tokenizer_seed=seed,
                items=len(rows),quantizer_steps=report['steps'],diagnostics=index.diagnostics())
            _write(work/'index-provenance.json',refs[idx]);_event('field_index_complete',index=idx,**refs[idx])
            gc.collect();torch.cuda.empty_cache()
    del encoder;gc.collect();torch.cuda.empty_cache();_write(root/'references.json',refs)
    return refs


def validate_completion(selection, curves, c, root):
    epoch=selection['training_epochs'];best=selection['best_epoch'];stop=selection['stopping_reason']
    if (epoch<30 or epoch>100 or epoch%10 or len(curves)!=epoch
            or best not in range(10,epoch+1,10) or stop not in ('early_stopping','epoch_limit')
            or (stop=='epoch_limit' and epoch!=100)
            or (stop=='early_stopping' and epoch-best<30)
            or selection['optimizer_updates']+selection['amp_skipped_updates']!=296*epoch
            or selection['training_examples_per_epoch']!=151166):
        raise RuntimeError('incomplete training or invalid stopping state')
    checks=[r for r in curves if 'validation' in r]
    if [r['epoch'] for r in checks]!=list(range(10,epoch+1,10)):
        raise RuntimeError('validation schedule incomplete')
    for row in checks:
        if ('validation_loss' not in row or len(row['validation_loss']['ce_by_level'])!=4
                or not np.isfinite(row['validation']['ndcg@10'])
                or not (Path(root)/f"validation-epoch-{row['epoch']:03d}.npz").is_file()):
            raise RuntimeError('missing validation metrics or predictions')
    if max(checks,key=lambda x:x['validation']['ndcg@10'])['epoch']!=best:
        raise RuntimeError('selected checkpoint does not match validation')
    if not all((Path(root)/n).is_file() for n in ('generator-best.pt','generator-last.pt','generator.json')):
        raise RuntimeError('missing checkpoint')
    return dict(plateau_reached=stop=='early_stopping',budget_censored=stop=='epoch_limit')


def run_trial(trial,c,plan,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    seed=trial['seed'];c=copy.deepcopy(c);c['seed']=seed
    root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    sid=Path(plan['index_root'])/f"{trial['index_id']}-semantic-ids.npy"
    if file_sha256(sid)!=plan['references'][trial['index_id']]['sids_sha256']:
        raise ValueError('frozen SID hash mismatch')
    index=SidIndex.from_full_sids(np.load(sid,allow_pickle=False),width=256)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build_grid_generator(index.sids,c['generator'],seed,device,plan['source'])
    initial=state_digest(dict(model.named_parameters()))
    if initial!=c['initialization_pins']['narrow'][str(seed)]:raise ValueError('generator initialization mismatch')
    _write(root/'generator.json',dict(config=c['generator'],parameters=sum(p.numel() for p in model.parameters()),
        initial_state_sha256=initial,index_id=trial['index_id'],sids_sha256=file_sha256(sid),seed=seed))
    selection=_train_generator(model,plan['data_root'],index,c,root,device,example_builder=case_examples)
    curves=json.loads((root/'training-curves.json').read_text());flags=validate_completion(selection,curves,c,root)
    row=dict(trial=trial['id'],field=trial['field'],index_id=trial['index_id'],tokenizer_seed=trial['tokenizer_seed'],
        seed=seed,architecture='narrow',initial_state_sha256=initial,selection=selection,
        epochs={str(r['epoch']):r['validation']['ndcg@10'] for r in curves if 'validation' in r},**flags)
    _write(root/'trial-result.json',row);return row


def run_worker(plan_path,slot):
    import torch,transformers
    plan=json.loads(Path(plan_path).read_text());c=plan['config'];validate(c)
    if transformers.__version__!='5.22.1' or torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5):
        raise RuntimeError('worker requires exactly one T4 and HF 5.22.1')
    torch.set_num_threads(1);status=dict(results={},failures=[]);path=Path(plan['output'])/'workers'/f'worker-{slot}.json'
    for trial in plan['assignments'][slot]['trials']:
        _event('field_learning_trial_start',trial=trial['id'],worker=slot)
        try:
            if time.time()>plan['deadline']:raise TimeoutError('declared suite deadline reached')
            status['results'][trial['id']]=run_trial(trial,c,plan,'cuda:0')
            _event('field_learning_trial_complete',**status['results'][trial['id']])
        except Exception as e:
            root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
            (root/'failure.log').write_text(traceback.format_exc());status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)))
            _write(path,status);return
        _write(path,status);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):
        raise RuntimeError('requires two T4 GPUs without fallback')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('input case hash mismatch')
    ids,texts,text_report=prepare_texts(c,roots['movie_input'],output/'text-inputs')
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    original=unpack_grid(roots['resources']/'grid-source.bin',scratch/'original');source=prepare_grid_source(original,scratch/'grid')
    refs=prepare_indices(c,texts,roots['resources'],source,original,output/'frozen-indexes','cuda:0')
    if set(refs)!={f'{f}_q{s}' for f in FIELDS for s in TOKENIZER_SEEDS}:raise RuntimeError('index matrix incomplete')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
        '--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),
        '-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),
        index_root=str(output/'frozen-indexes'),references=refs,source_ids_sha256=file_sha256(roots['movie_input']/'source-ids.json'),
        deadline=time.time()+c['suite_deadline_seconds'],assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    _event('field_learning_suite_start',indexes=len(refs),trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.field_learning',event_prefix='field_learning')
    records=summary['results'];_write(output/'learning-summary.json',dict(results=records,test_evaluated=False,
        note='Operational early stopping is not proof of convergence; token and generator seeds are partially crossed.'))
    if set(records)!={t['id'] for t in c['trials']}:raise RuntimeError('missing generator results')
    return dict(experiment=c['name'],models=len(records),indexes=len(refs),test_evaluated=False,
        operational_stops=sum(r['plateau_reached'] for r in records.values()),
        budget_censored=sum(r['budget_censored'] for r in records.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--slot',type=int,required=True)
    args=parser.parse_args();run_worker(args.plan,args.slot)
