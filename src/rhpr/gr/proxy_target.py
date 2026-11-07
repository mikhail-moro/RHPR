"""Small independent full-generator-seed check of a frozen mini-TIGER proxy."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from rhpr.gr.mini_tiger import selection_stats


INDEX_IDS = [f'{m}_seed{s}' for s in [42, 101, 202] for m in ['sum', 'gradnorm']]
TARGET_SEED = 161803


def reference_digest(reference):
    return hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_target(c):
    """Reject changes outside the six-fit, fixed-index, VAL-only authorization."""
    t, g = c['training'], c['generator']
    trials = [dict(id=f'full_{i}_gen{TARGET_SEED}', architecture='full', index_id=i,
                   seed=TARGET_SEED) for i in INDEX_IDS]
    generator = dict(kind='grid_encoder_decoder', d_model=128, d_ff=1024, d_kv=64,
                     num_heads=6, encoder_layers=4, decoder_layers=4, dropout=.1,
                     mlp_layers=None, num_user_bins=2000, separator_token=False,
                     codebook_size=256, beam_width=20, should_check_prefix=False,
                     attn_implementation='sdpa')
    training = dict(device='cuda', cpu_threads=1, batch_size=512, physical_batch_size=512,
                    learning_rate=.001, weight_decay=.0001, amp=True, grad_clip_norm=1.,
                    max_epochs=30, min_epochs=30, patience=5, validation_interval=5,
                    evaluation_batch_size=32, max_seconds=2300,
                    sampling='precomputed_raw_clean_windows', precision='fp16',
                    evaluation_precision='fp32', validate_first_epoch=False, patience_epochs=30)
    if (c['kind'] != 'gr_mini_tiger' or c['calibration_phase'] != 'full_seed_replication'
            or c['index_ids'] != INDEX_IDS or c['generator_seeds'] != [TARGET_SEED]
            or c['seed'] != TARGET_SEED or c['trials'] != trials or g != generator or t != training
            or c['suite_deadline_seconds'] != 7500 or c['primary_proxy_epoch'] != 10
            or c['data']['history_items'] != 20 or c['data']['train_examples'] != 151166
            or c['evaluation'] != dict(test_after_selection=False, save_validation_predictions=True,
                selection_metric='validation item NDCG@10', save_validation_losses=True)
            or c['execution'] != dict(worker_devices=['cuda:0', 'cuda:1'], cpu_threads_per_worker=1)
            or set(c['parents']) != {'42', '101', '202'} or 'index_input' in c
            or set(c['kaggle']['kernel_sources']) != {p['kernel'] for p in c['parents'].values()}
            or c['kaggle']['machine_shape'] != 'NvidiaTeslaT4'
            or set(c['initialization_pins']) != {'full'}
            or set(c['initialization_pins']['full']) != {str(TARGET_SEED)}
            or len(c['initialization_pins']['full'][str(TARGET_SEED)]) != 64):
        raise ValueError('independent target check must remain exactly six full30 fits, one new seed, VAL only')
    reference = c['proxy_reference']
    if (reference_digest(reference) != c['proxy_reference_sha256']
            or reference['architecture'] != 'narrow' or reference['epoch'] != 10
            or reference['generator_seeds'] != [31415, 27182]
            or set(reference['indexes']) != set(INDEX_IDS)):
        raise ValueError('historical proxy reference changed')
    for i in INDEX_IDS:
        seed = i.split('_seed')[1]
        if set(c['parents'][seed]['artifacts']) != {f'{m}_seed{seed}' for m in ['sum', 'gradnorm']}:
            raise ValueError('only the six specified parent indexes may be staged')
        row = reference['indexes'][i]
        if row['sids_sha256'] != c['parents'][seed]['artifacts'][i]['semantic-ids.npy']:
            raise ValueError('proxy and target must use identical SID bytes')
        if set(row['scores_by_seed']) != {'31415', '27182'} or not all(
                np.isfinite(v) and 0 <= v <= 1 for v in row['scores_by_seed'].values()):
            raise ValueError('both historical epoch10 proxy scores are required')


def _stats(scores, target, ids):
    scores, target = np.asarray(scores), np.asarray(target)
    r = selection_stats(scores, target)
    order = np.argsort(-scores, kind='stable')
    best = int(np.argmax(target))
    r.update(selected_index=ids[int(order[0])], best_target_index=ids[best],
             relative_target_regret=float((target.max()-target[order[0]])/target.max()) if target.max() else None,
             best_in_top3=bool(best in order[:3]),
             shortlist3_target_regret=float(target.max()-target[order[:3]].max()))
    return r


def correlate_target(c, records, references):
    """Fixed epoch30 is primary; selected checkpoints never define that target."""
    ids = c['index_ids']
    if len(records) != 6 or set(records) != {t['id'] for t in c['trials']}:
        raise ValueError('six completed independent full fits are required')
    by_index = {}
    for row in records.values():
        if row['seed'] != TARGET_SEED or row['architecture'] != 'full' or row['index_id'] in by_index:
            raise ValueError('wrong or duplicate full-generator reference')
        if set(row['epochs']) != {'5', '10', '15', '20', '25', '30'}:
            raise ValueError('all six fixed-epoch validation scores are required')
        by_index[row['index_id']] = row
    proxy = c['proxy_reference']['indexes']
    scores = {f'mini_gen{s}_10': [proxy[i]['scores_by_seed'][str(s)] for i in ids] for s in [31415, 27182]}
    scores['mini_mean10'] = np.mean([scores['mini_gen31415_10'], scores['mini_gen27182_10']], axis=0).tolist()
    scores['historical_full30'] = [references[i]['epochs']['30'] for i in ids]
    for epoch in [5, 10, 15, 20, 25]:
        scores[f'early_new_full{epoch}'] = [by_index[i]['epochs'][str(epoch)] for i in ids]
    target = [by_index[i]['epochs']['30'] for i in ids]
    groups = {'all': ids, **{f'method_{m}': [i for i in ids if i.startswith(m+'_')] for m in ['sum', 'gradnorm']},
              **{f'index_seed{s}': [i for i in ids if i.endswith(str(s))] for s in [42, 101, 202]}}
    result = dict(target='new full TIGER fixed epoch30 VAL NDCG@10', target_seed=TARGET_SEED,
                  test_evaluated=False, index_ids=ids, target_scores=dict(zip(ids, target)),
                  estimator_scores=scores, proxy_reference_sha256=c['proxy_reference_sha256'],
                  limitations=['Six related, previously examined indexes; one new full-generator seed.',
                    'Early new full scores share their target trajectory and are not independent transfer.',
                    'Existing same validation users; no new tokenizer family or convergence claim.',
                    'Two-candidate seed subgroups are descriptive, not evidence of robust correlation.'], groups={})
    for group, members in groups.items():
        ix = [ids.index(i) for i in members]
        result['groups'][group] = dict(index_ids=members, estimators={
            name: _stats([v[j] for j in ix], [target[j] for j in ix], members) for name, v in scores.items()})
    return result
