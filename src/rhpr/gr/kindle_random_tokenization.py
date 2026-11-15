"""Random chronological monomer/dimer tokenization of the older half only."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr import kindle_mixed_views as base
from rhpr.gr.experiment import _event, _write

TRIALS = [dict(id='random32_wide_42', seed=42, history_items=32)]
POLICY = dict(merge_fraction=.5, schedule='constant', count_unit='optimizer_batch',
    positions_unit='individual_history', tiling='uniform_matching_given_count',
    short_history_count='binomial_at_available_capacity',
    inference='pair32', count_rng_stream=7401, tiling_rng_stream=7402,
    preserve_native_positions=True, force_original_pair_endpoint=True,
    compiler_recompile_limit=32)


def validate(c):
    if (c['kind'] != 'gr_kindle_random_tokenization' or c['trials'] != TRIALS
        or c['random_tokenization'] != POLICY or 'resume_parent' in c or 'mixed_views' in c):
        raise ValueError('expected one constant-p random-tiling fit; no schedule/HPO/resume')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_mixed_views', trials=base.TRIALS, mixed_views=base.POLICY)
    base.validate(normalized)


def matching_starts(length, count, rng):
    """Uniform size-count matching on a chronological path of length nodes.

    Choose count sorted distinct values from0..length-count-1,then add0..count-1.
    This bijection covers every possible nonoverlapping adjacent matching once.
    """
    if not 0 <= count <= length//2:
        raise ValueError('too many adjacent pairs')
    return np.sort(rng.choice(length-count, size=count, replace=False)) + np.arange(count)


class RandomTokenization:
    def __init__(self, contexts, item_sids, keys, pair_sids, batch_size=512, seed=42, merge_fraction=.5):
        h = np.asarray(contexts)
        if (h.ndim != 2 or h.shape[1] != 64 or item_sids.ndim != 2 or item_sids.shape[1] != 4
            or pair_sids.shape != (len(keys), 4) or batch_size <= 0 or not 0 <= merge_fraction <= 1
            or np.any(h < -1) or np.any(h >= len(item_sids)) or np.any(keys[1:] <= keys[:-1])):
            raise ValueError('expected original64-item cases and frozen unique3+1SID indexes')
        h = h[:, -32:]; self.lengths = (h >= 0).sum(1)
        if np.any((np.arange(32)[None, :] < (32-self.lengths)[:, None]) != (h < 0)):
            raise ValueError('histories must use contiguous left padding')
        self.n, self.batch_size, self.seed, self.probability = len(h), batch_size, seed, merge_fraction
        positions = np.minimum((32-self.lengths)[:, None]+np.arange(32), 31)
        compact = np.take_along_axis(h, positions, axis=1)
        self.singles = (item_sids[np.maximum(compact, 0)] + np.arange(4)*256).astype(np.int32)
        self.pairs = np.zeros((self.n, 15, 4), dtype=np.int32)
        valid = np.arange(15)[None, :]+1 < (self.lengths//2)[:, None]
        query = compact[:, :15].astype(np.int64)*len(item_sids)+compact[:, 1:16]
        needed = query[valid]; at = np.searchsorted(keys, needed)
        if np.any(at >= len(keys)) or not np.array_equal(keys[at], needed):
            raise ValueError('random adjacent pair missing from frozen index; never silently drop it')
        self.pairs[valid] = pair_sids[at] + (4+np.arange(4))*256
        self.unique_adjacent_candidates = len(np.unique(needed))
        self.epoch = None

    def schedule(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, POLICY['count_rng_stream']]))
        return (rng.random(((self.n+self.batch_size-1)//self.batch_size, 8)) < self.probability).sum(1).astype(np.int8)

    def merge_slots(self, count, epoch, batch):
        """Shared conditional subset gives every short history its own Binomial(cap,p).

        It preserves native32-count width:missing merges<=8-cap<=32-length.
        Slots govern only counts;actual chronological matching is sampled separately.
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, batch, POLICY['count_rng_stream'], 1]))
        slots = np.zeros(8, dtype=bool); slots[rng.choice(8, size=count, replace=False)] = True
        return slots

    def encode_batch(self, part, count, epoch, batch):
        """Exact native max-width; only short original histories receive padding."""
        part = np.asarray(part); out = np.full((len(part), 4*(32-count)), -1, dtype=np.int32)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, batch, POLICY['tiling_rng_stream']]))
        slot_counts = np.r_[0, self.merge_slots(count, epoch, batch).cumsum()]
        pair_count = odd_starts = 0
        for row, index in enumerate(part):
            n = int(self.lengths[index]); old = n//2; k = int(slot_counts[old//2])
            # Endpoints replay Full32 and originalPair32 exactly,including short windows.
            starts = np.arange(0, 2*k, 2) if count == 8 else matching_starts(old, k, rng)
            merge = set(starts.tolist()); tokens = []; j = 0
            while j < n:
                if j in merge:
                    tokens.append(self.pairs[index, j]); j += 2
                else:
                    tokens.append(self.singles[index, j]); j += 1
            if tokens:
                x = np.concatenate(tokens)
                if len(x) > out.shape[1]: raise RuntimeError('random layout exceeds native bound')
                out[row, -len(x):] = x
            pair_count += k; odd_starts += int((starts % 2).sum())
        return out, pair_count, odd_starts

    def __call__(self, epoch, batch, part):
        if self.epoch != epoch:
            if self.epoch is not None: self.epoch_summary(self.epoch)
            self.epoch = epoch; self.choices = self.schedule(epoch)
            self.order = np.random.default_rng(self.seed+epoch).permutation(self.n)
            self.inverse = np.argsort(self.order); self.counts = np.zeros(self.n, dtype=np.int32)
            self.batch_counts = np.zeros(9, dtype=np.int64); self.example_counts = np.zeros(9, dtype=np.int64)
            self.pair_occurrences = self.odd_pair_occurrences = self.padding_positions = 0
            self.digest = hashlib.sha256(); self.generated = set(); self.cached_batch = None
        part = np.asarray(part)
        if (part.ndim != 1 or part.dtype.kind not in 'iu' or np.any(part < 0) or np.any(part >= self.n)
            or batch < 0 or batch >= len(self.choices)):
            raise ValueError('invalid batch selection')
        begin = batch*self.batch_size; end = min(self.n, begin+self.batch_size)
        positions = self.inverse[part]-begin
        if np.any(positions < 0) or np.any(positions >= end-begin):
            raise ValueError('tokenizer must retain original seed/epoch training permutation')
        if self.cached_batch != batch:
            chosen = self.order[begin:end]; count = int(self.choices[batch])
            self.cached, pairs, odd = self.encode_batch(chosen, count, epoch, batch)
            self.cached_batch = batch
            if batch not in self.generated:
                self.digest.update(chosen.astype('<i8').tobytes()); self.digest.update(self.cached.astype('<i4').tobytes())
                self.batch_counts[count] += 1; self.example_counts[count] += len(chosen)
                self.pair_occurrences += pairs; self.odd_pair_occurrences += odd
                self.padding_positions += int((self.cached < 0).sum()); self.generated.add(batch)
        np.add.at(self.counts, part, 1)
        return self.cached[positions]

    def epoch_summary(self, epoch):
        if self.epoch != epoch or not np.all(self.counts == 1) or len(self.generated) != len(self.choices):
            raise RuntimeError('random tokenization must consume every original example exactly once')
        return dict(merge_fraction=self.probability, merge_count_batches=self.batch_counts.tolist(),
            merge_count_examples=self.example_counts.tolist(), pair_summary_occurrences=int(self.pair_occurrences),
            odd_start_pair_occurrences=int(self.odd_pair_occurrences), padding_sid_positions=self.padding_positions,
            native_sid_widths=[4*(32-i) for i in range(9)], examples_consumed_once=True,
            schedule_sha256=hashlib.sha256(self.choices.tobytes()).hexdigest(), tokenized_epoch_sha256=self.digest.hexdigest())


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    status = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('random_tokenization_trial_start', trial=trial['id'], policy=POLICY)
        try:
            # Nine native lengths plus their final-batch shapes exceed default8.
            # Raise only the compile-cache limit in this isolated worker;no math change.
            with torch._dynamo.config.patch(recompile_limit=POLICY['compiler_recompile_limit']):
                row = base.run_trial(trial, p, view_factory=RandomTokenization,
                    policy=POLICY, mode='random_older_half_tiling')
            status['results'][trial['id']] = row
            _event('random_tokenization_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('random_tokenization_trial_failed', **status['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', status)
        gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    return base.run(c, output, validator=validate, trials=TRIALS,
        worker_module='rhpr.gr.kindle_random_tokenization', event_prefix='random_tokenization', policy=POLICY)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); worker(args.plan, args.slot)
