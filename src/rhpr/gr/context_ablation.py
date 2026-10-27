"""Inference-only history truncation with fixed users, targets, batches and padding."""
import hashlib
import json
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _evaluate, _event, _write
from rhpr.gr.data import file_sha256
from rhpr.gr.sid_study import state_digest


def truncate_examples(examples, length):
    """Mask old events in place-shaped copies; never move surviving positions."""
    contexts, users, targets = examples
    if (type(length) is not int or not 1 <= length <= contexts.shape[1]):
        raise ValueError('context length must be an integer within the fixed padded width')
    result = contexts.copy()
    result[:, :contexts.shape[1]-length] = -1
    return result, users, targets


def user_values(predictions, sids, targets):
    hit = np.all(predictions == sids[targets, None, :], axis=-1)
    rank = np.where(hit.any(1), hit.argmax(1)+1, 999)
    values = {f'ndcg@{k}': np.where(rank <= k, 1/np.log2(rank+1), 0) for k in [5, 10]}
    values.update({f'recall@{k}': (rank <= k).astype(float) for k in [5, 10]})
    return rank, values


def evaluate_context_ablation(model, index, examples, users, trial, config, paths,
                             before, output, device):
    """All passes use the original full user order/batch shape, then slice cohorts."""
    spec = config['context_ablation']; lengths = spec['lengths']; width = examples[0].shape[1]
    if (not lengths or any(type(n) is not int or not 1 <= n <= width for n in lengths)
            or lengths != sorted(set(lengths)) or lengths[-1] != width):
        raise ValueError('context sweep needs sorted unique lengths ending at the original width')
    if (len(users) != len(examples[1]) or not np.array_equal(
            np.asarray([u.user_id for u in users]), examples[1])):
        raise ValueError('history metadata and validation user order differ')
    for cohort in spec['cohorts']:
        if (type(cohort['minimum_history']) is not int or cohort['minimum_history'] < 1
                or not cohort['lengths'] or len(set(cohort['lengths'])) != len(cohort['lengths'])
                or any(n not in lengths or n > cohort['minimum_history'] for n in cohort['lengths'])):
            raise ValueError('every cohort must contain the same users with enough history for all its lengths')
    directory = Path(output)/'trials'/trial['id']; directory.mkdir(parents=True, exist_ok=True)
    selection = json.loads(paths['selection'].read_text())
    if selection['selected_epoch'] != config['validation']['checkpoint_epoch']:
        raise ValueError('reference selection epoch differs from frozen checkpoint')
    reference = selection['selected_validation']
    history_lengths = np.asarray([len(u.train) for u in users])
    sids_digest = hashlib.sha256(index.sids.tobytes()).hexdigest()
    all_values = {}; ranks = {}; controls = []; anchor = None; anchor_score = None
    parent_prediction_match = None
    # Evaluate the unchanged baseline first. Other passes retain identical batches.
    for length in [width] + [n for n in lengths if n != width]:
        truncated = truncate_examples(examples, length)
        score, predictions, probabilities = _evaluate(model, truncated, index,
            config['validation']['batch_size'], device, precision=config['validation']['precision'])
        rank, values = user_values(predictions, index.sids, examples[2])
        for key, value in values.items():
            if abs(float(value.mean())-score[key]) > 1e-12:
                raise RuntimeError('per-user context metrics differ from aggregate')
        if length == width:
            anchor = predictions.copy(); anchor_score = score
            with np.load(paths['reference_predictions'], allow_pickle=False) as original:
                np.testing.assert_array_equal(original['user_ids'], examples[1])
                np.testing.assert_array_equal(original['target_item_ids'], examples[2])
                parent_prediction_match = bool(np.array_equal(predictions, original['semantic_ids']))
        unchanged = np.all(truncated[0] == examples[0], axis=1)
        drift = int(np.any(predictions[unchanged] != anchor[unchanged], axis=(1, 2)).sum())
        controls.append(dict(length=length,unchanged_input_users=int(unchanged.sum()),
                             changed_predictions_with_unchanged_input=drift))
        if drift:
            raise RuntimeError('predictions changed for identical inputs in fixed-layout context sweep')
        np.savez_compressed(directory/f'context-L{length:03d}-predictions.npz',
            semantic_ids=predictions,scores=probabilities,user_ids=examples[1],target_item_ids=examples[2])
        ranks[length] = rank; all_values[length] = values
        _event('context_ablation_length_complete',trial=trial['id'],length=length,validation=score)
    if (state_digest(model.state_dict()) != before
            or hashlib.sha256(index.sids.tobytes()).hexdigest() != sids_digest):
        raise RuntimeError('weights or SID lookup changed during context ablation')
    cohorts = []
    for cohort in spec['cohorts']:
        mask = history_lengths >= cohort['minimum_history']
        if not mask.any():
            raise ValueError('declared fixed context cohort is empty')
        cohorts.append(dict(id=cohort['id'],minimum_history=cohort['minimum_history'],
            primary=cohort.get('primary',False),users=int(mask.sum()),
            user_ids_sha256=hashlib.sha256(examples[1][mask].tobytes()).hexdigest(),
            metrics_by_length={str(length):{key:float(value[mask].mean())
                for key,value in all_values[length].items()} for length in cohort['lengths']}))
    arrays = dict(user_ids=examples[1],target_item_ids=examples[2],history_lengths=history_lengths,
                  context_lengths=np.asarray(lengths),rank=np.stack([ranks[n] for n in lengths],axis=1))
    for key in all_values[width]:
        arrays[key] = np.stack([all_values[n][key] for n in lengths],axis=1)
    np.savez_compressed(directory/'context-user-metrics.npz',**arrays)
    result = dict(trial=trial['id'],seed=config['seed'],epoch=config['validation']['checkpoint_epoch'],
        validation=anchor_score,original_validation=reference,
        replay_delta={key:anchor_score[key]-reference[key] for key in all_values[width]},
        checkpoint_sha256=file_sha256(paths['checkpoint']),sids_sha256=file_sha256(paths['sids']),
        loaded_state_sha256=before,model_unchanged=True,optimizer_steps=0,training_performed=False,
        test_evaluations=0,validation_evaluations=len(lengths),context_cohorts=cohorts,
        unchanged_input_controls=controls,parent_full_context_predictions_exact=parent_prediction_match,
        padding_width_preserved=width,batch_size=config['validation']['batch_size'])
    _write(directory/'result.json',result)
    return result
