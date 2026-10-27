"""Random whole-path SID reassignment with exact catalogue trie preservation."""
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256


def validate_corruption(config):
    spec = config['sid_corruption']
    fractions = [t['corruption_fraction'] for t in config['trials']]
    if (type(spec['seed']) is not int or not 0 <= spec['seed'] < 2**32
            or any(type(p) not in [int,float] or not np.isfinite(p) or not 0 <= p <= 1 for p in fractions)
            or fractions != sorted(set(fractions)) or fractions[0] != 0 or fractions[-1] != 1
            or config['quantizer']['kind'] != 'frozen_full_sids' or config.get('index_gate')
            or any('generator_seed' in t for t in config['trials'])
            or config.get('evaluation',{}).get('test_after_selection',True)):
        raise ValueError('invalid frozen SID corruption protocol')


def corruption_owner(items, fraction, seed):
    if (type(items) is not int or items < 2 or type(seed) is not int or not 0 <= seed < 2**32
            or type(fraction) not in [int,float] or not np.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError('invalid corruption size, fraction or seed')
    # One fixed random priority yields nested selected sets. Each dose has its
    # own independent RNG, so adding/removing another dose cannot alter it.
    chosen = np.random.default_rng(seed).permutation(items)[:round(items*fraction)]
    owner = np.arange(items,dtype=np.int64)
    if len(chosen) == 1:
        raise ValueError('one selected item cannot receive a different existing path')
    if len(chosen):
        rng = np.random.default_rng(np.random.SeedSequence([seed,len(chosen)]))
        for _ in range(10000):
            assigned = rng.permutation(chosen)
            if np.all(assigned != chosen):
                owner[chosen] = assigned
                break
        else:raise RuntimeError('failed to sample a uniform derangement')
    return owner,chosen


def corruption_diagnostics(original, owner, frequency, graph):
    original, owner = np.asarray(original),np.asarray(owner)
    np.testing.assert_array_equal(np.sort(owner),np.arange(len(original)))
    paths = original[owner]
    levels=[]
    for depth in range(1,original.shape[1]+1):
        keys,labels,counts=np.unique(original[:,:depth],axis=0,return_inverse=True,return_counts=True)
        other,assigned,newcounts=np.unique(paths[:,:depth],axis=0,return_inverse=True,return_counts=True)
        np.testing.assert_array_equal(keys,other);np.testing.assert_array_equal(counts,newcounts)
        before=np.bincount(labels,weights=frequency,minlength=len(keys))
        after=np.bincount(assigned,weights=frequency,minlength=len(keys))
        p,q=before/before.sum(),after/after.sum()
        def entropy(x):
            x=x[x>0];return float(-np.sum(x*np.log2(x)))
        levels.append(dict(depth=depth,catalogue_prefixes=len(keys),catalogue_occupancies_exact=True,
            changed_item_prefixes=int(np.any(original[:,:depth]!=paths[:,:depth],axis=1).sum()),
            train_target_prefix_total_variation=float(.5*np.abs(p-q).sum()),
            train_target_prefix_entropy_bits=entropy(q),original_train_target_prefix_entropy_bits=entropy(p),
            train_target_max_prefix_share=float(q.max())))
    semantic=graph.prefix_scores(paths,depth=original.shape[1]-1)
    return dict(changed_items=int((owner!=np.arange(len(owner))).sum()),
        changed_item_train_target_mass=float(frequency[owner!=np.arange(len(owner))].sum()/frequency.sum()),
        levels=levels,semantic_neighbor_shared_prefix_fractions=semantic,
        semantic_neighbor_normalized_lcp=float(np.mean(semantic)),
        interpretation='Exact catalogue occupancy preservation;TRAIN-weighted token frequencies are allowed to change. '
                       'Semantic scores measure frozen Sentence-T5 cosine neighbors,not ground-truth semantics.')


def prepare_corrupted_indices(config, source_ids, users, output):
    from rhpr.gr.full_sid_index import prepare_full_sid_indices
    from rhpr.gr.resources import verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_graph_swap import NeighborGraph
    from rhpr.offline_runtime import resolve_dataset_directory
    validate_corruption(config)
    output=Path(output);spec=config['sid_corruption']
    # Load the identical byte-pinned original path table through the existing
    # catalogue/manifest checks before deriving any randomized assignments.
    base={**config,'trials':[{**t,'role':'original','semantic_lambda':0,
        'use_original_paths':True,'expected_variant':'original'} for t in config['trials']]}
    records=prepare_full_sid_indices(base,source_ids,output)
    first=output/'indices'/config['trials'][0]['id']/'semantic-ids.npy'
    if file_sha256(first)!=spec['original_sids_sha256']:
        raise ValueError('corruption base differs from the declared original mapping')
    original=np.load(first,allow_pickle=False)
    semantic_spec=spec['semantic_graph']
    semantic_root=resolve_dataset_directory(semantic_spec['path'])
    manifest=verify_resources(semantic_root,semantic_spec['manifest_sha256'])
    if (manifest['data_sha256']!=config['data']['sha256']
            or json.loads((semantic_root/'catalogue.json').read_text())['original_item_ids']!=source_ids):
        raise ValueError('semantic graph catalogue/data differs from SID catalogue')
    if file_sha256(semantic_root/'semantic-graph.npz')!=semantic_spec['file_sha256']:
        raise ValueError('semantic graph file digest mismatch')
    with np.load(semantic_root/'semantic-graph.npz',allow_pickle=False) as f:
        graph=NeighborGraph(f['source'],f['target'],f['weight'],len(original))
    frequency=np.bincount(np.concatenate([u.train[1:] for u in users]),minlength=len(original))
    diagnostics=[]
    for trial in config['trials']:
        name=trial['id'];d=output/'indices'/name
        owner,selected=corruption_owner(len(original),trial['corruption_fraction'],spec['seed'])
        paths=original[owner]
        index=SidIndex.from_full_sids(paths,width=config['quantizer']['codebook_size'])
        if trial['corruption_fraction']:
            np.save(d/'semantic-ids.npy',paths,allow_pickle=False)
        np.save(d/'owner.npy',owner,allow_pickle=False)
        np.save(d/'selected-items.npy',selected,allow_pickle=False)
        for key,file in [('expected_sid_sha256','semantic-ids.npy'),('expected_owner_sha256','owner.npy')]:
            if key in trial and file_sha256(d/file)!=trial[key]:
                raise ValueError('corrupted path/owner bytes differ from pinned CPU preflight')
        np.save(d/'semantic-codes.npy',index.semantic_codes,allow_pickle=False)
        (d/'sid-diagnostics.json').write_text(json.dumps(index.diagnostics(),indent=2)+'\n')
        report=dict(trial=name,fraction=trial['corruption_fraction'],permutation_seed=spec['seed'],
                    **corruption_diagnostics(original,owner,frequency,graph))
        diagnostics.append(report)
        record=records[name]
        record['original_source']=record['source']
        record['source']=dict(family=trial['family'],role=trial['role'],variant=trial['expected_variant'],
                              corruption_fraction=trial['corruption_fraction'])
        record.update(semantic_ids_sha256=file_sha256(d/'semantic-ids.npy'),
                      diagnostics=index.diagnostics(),corruption=report)
        (d/'index-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'sid-catalogue.json').write_text(json.dumps(dict(original_item_ids=source_ids,
        row_order='internal_item_id_0_based',indices=records),indent=2)+'\n')
    (output/'corruption-diagnostics.json').write_text(json.dumps(dict(
        config=spec,train_target_positions=int(frequency.sum()),trials=diagnostics,
        validation_and_test_used=False),indent=2,allow_nan=False)+'\n')
    np.save(output/'train-target-item-frequency.npy',frequency,allow_pickle=False)
    return records


def teacher_forced_diagnostics(model, examples, index, batch_size, device, precision):
    import torch
    from rhpr.gr.experiment import _autocast,_index_model_batch
    contexts,users,targets=examples
    was_training=model.training
    nll,ranks=[],[]
    model.eval()
    try:
        with torch.inference_mode(),_autocast(precision,device):
            for begin in range(0,len(targets),batch_size):
                end=begin+batch_size
                batch=_index_model_batch(contexts[begin:end],users[begin:end],targets[begin:end],index,device)
                hidden=model(**batch)[:,:-1]
                losses,positions=[],[]
                for level,head in enumerate(model.decoder.decoder_mlp):
                    logits=head(hidden[:,level]).float();truth=batch['future_ids'][:,level]
                    loss=torch.nn.functional.cross_entropy(logits,truth,reduction='none')
                    if not torch.isfinite(loss).all():raise FloatingPointError('nonfinite teacher-forced NLL')
                    losses.append(loss.cpu().numpy())
                    positions.append((1+(logits>logits.gather(1,truth[:,None])).sum(1)).cpu().numpy())
                nll.append(np.stack(losses,axis=1));ranks.append(np.stack(positions,axis=1))
    finally:model.train(was_training)
    nll=np.concatenate(nll);ranks=np.concatenate(ranks).astype(np.int16)
    report=dict(examples=len(targets),nll_per_level=nll.mean(0,dtype=np.float64).tolist(),
                nll_sum_over_codes=float(nll.sum(1,dtype=np.float64).mean()),
                true_token_top1_per_level=(ranks==1).mean(0).tolist(),
                conditioning='Teacher-forced true preceding SIDtokens;full head vocabulary incl invalid continuations.',
                rank_ties='Optimistic rank:one plus number of strictly greater logits.')
    return report,dict(user_ids=users,target_item_ids=targets,nll_by_level=nll,true_token_rank_by_level=ranks)
