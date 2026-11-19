import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.ml1m_last8_pairs import (
    EXTENDED_TRIALS, MERGE_PREFIXES, MIXED_TRIALS, PARENT_KERNEL, REPLICATION_TRIALS, TRIALS, encode,
    full_context_selector, pair_keys, profiling_batch, validate,
)


class Last8PairsTests(unittest.TestCase):
    def setUp(self):
        self.nitems=300
        self.sids=np.array([[i%256,i//256,0,0] for i in range(self.nitems)],dtype=np.int32)
        self.h=np.full((129,128),-1,dtype=np.int32)
        for n in range(129):self.h[n,128-n:]=np.arange(n)
        self.keys=pair_keys(self.h,self.nitems,120)
        self.pairs=np.array([[i,1,2,0] for i in range(len(self.keys))],dtype=np.int32)

    def test_depth_pilot_changes_depth_with_matched_history_and_training_recipes(self):
        from rhpr.gr.ml1m_last8_pairs import depth_pilot_reference, encode_full

        configs = [json.loads(Path(f'experiments/ml1m_last8_depth{depth}_capacity_colab.json').read_text())
                   for depth in (4, 8)]
        for config in configs:
            validate(config)
            self.assertEqual(config['seeds'], [101])
            self.assertEqual(config['resources']['include_prefixes'], ['grid-source.bin', 'wheels/'])
            self.assertEqual([trial['merge_items'] for trial in config['trials']], [0, 64])
        self.assertEqual(configs[0]['training'], configs[1]['training'])
        self.assertEqual(configs[0]['frozen_parent'], configs[1]['frozen_parent'])
        self.assertEqual(configs[0]['generator'], depth_pilot_reference(configs[1])['generator'])
        thinned = encode(self.h, self.sids, self.keys, self.pairs, 64, keep_second=True)
        full = encode_full(self.h, self.sids)
        for config in configs:
            selector = full_context_selector(self.h, self.sids, thinned, config, 64)
            np.testing.assert_array_equal(full, selector.full)
        self.assertEqual(full.shape[1], 512)
        self.assertEqual(thinned.shape[1], 384)

    def test_every_length_and_merge_cap_preserves_events_order_odd_prefix_and_tail(self):
        single_map={tuple(s):i for i,s in enumerate(self.sids)}
        pair_map={tuple(s):divmod(int(k),self.nitems) for k,s in zip(self.keys,self.pairs)}
        for m in MERGE_PREFIXES:
            x=encode(self.h,self.sids,self.keys,self.pairs,m)
            self.assertEqual(x.shape,(129,4*(128-m//2)))
            for n,row in enumerate(x):
                blocks=row.reshape(-1,4);valid=blocks[:,0]>=0
                self.assertTrue(np.all(blocks[~valid]==-1))
                self.assertTrue(np.all(np.diff(valid.astype(int))>=0))
                rebuilt=[];types=[]
                for block in blocks[valid]:
                    if block[0]>=1024:
                        rebuilt.extend(pair_map[tuple(block-(4+np.arange(4))*256)]);types.append('pair')
                    else:rebuilt.append(single_map[tuple(block-np.arange(4)*256)]);types.append('single')
                self.assertEqual(rebuilt,list(range(n)),(m,n))
                expected_pairs=max(n-(128-m),0)//2
                self.assertEqual(types.count('pair'),expected_pairs)
                self.assertEqual(types,['pair']*expected_pairs+['single']*(n-2*expected_pairs))

    def test_thinning_preserves_mixed64_positions_with_second_item_sids(self):
        config = json.loads(Path('experiments/ml1m_last8_mixed64_thinned_2t4.json').read_text())
        validate(config)
        thinned = encode(self.h, self.sids, self.keys, self.pairs, 64, keep_second=True)
        paired = encode(self.h, self.sids, self.keys, self.pairs, 64)
        np.testing.assert_array_equal(thinned >= 0, paired >= 0)
        offsets = np.arange(4) * 256
        for length, row in enumerate(thinned.reshape(129, 96, 4)):
            count = max(length - 64, 0) // 2
            items = [*range(1, 2 * count, 2), *range(2 * count, length)]
            np.testing.assert_array_equal(row[row[:, 0] >= 0], self.sids[items] + offsets)
        for seed in config['seeds']:
            config['seed'] = seed
            thin_views = full_context_selector(self.h, self.sids, thinned, config, 64)
            pair_config = {k: v for k, v in config.items() if k != 'history_reduction'}
            pair_views = full_context_selector(self.h, self.sids, paired, pair_config, 64)
            np.testing.assert_array_equal(thin_views.full, pair_views.full)
            self.assertEqual(thin_views.widths, {'full128': 512, 'thin128': 384})
            for epoch in (1, 40, 60):
                np.testing.assert_array_equal(thin_views.schedule(epoch), pair_views.schedule(epoch))

    def test_smallest_tail_covers_all_fixed_pair_indexes(self):
        for m in MERGE_PREFIXES:
            self.assertTrue(np.isin(pair_keys(self.h,self.nitems,m),self.keys).all())
        for m in MERGE_PREFIXES:
            short=np.full((1,128),-1,dtype=np.int32);short[0,-(128-m):]=np.arange(128-m)
            self.assertEqual(len(pair_keys(short,self.nitems,m)),0)

    def test_missing_pairs_and_invalid_padding_fail_without_dropping_events(self):
        with self.assertRaisesRegex(ValueError,'missing pair'):encode(self.h,self.sids,self.keys[1:],self.pairs[1:],120)
        for bad in (self.h.astype(float),self.h[:,:127],np.full((2,128),300),np.full((2,128),-2)):
            with self.assertRaises(ValueError):pair_keys(bad,self.nitems,64)
        bad=self.h[-1:].copy();bad[0,50]=-1
        with self.assertRaises(ValueError):encode(bad,self.sids,self.keys,self.pairs,64)
        with self.assertRaises(ValueError):encode(self.h,self.sids,self.keys,self.pairs,128)

    def test_actual_targets_remain_single_sid_in_model_batch(self):
        from rhpr.gr.pair_unique4 import InputIndex4
        from rhpr.gr.sid import SidIndex
        index=InputIndex4(SidIndex.from_full_sids(self.sids,width=256))
        x=encode(self.h[-2:],self.sids,self.keys,self.pairs,120)
        batch=profiling_batch(index,x,np.array([1,2]),np.array([299,298]),self.sids,'cpu')
        np.testing.assert_array_equal(batch['future_ids'].numpy(),self.sids[[299,298]])
        self.assertEqual(batch['future_ids'].shape,(2,4))

    def test_only_four_same_seed_fixed_screens_and_existing40epoch_control(self):
        c=json.loads(Path('experiments/ml1m_last8_pair128_2t4.json').read_text());validate(c)
        self.assertEqual(c['trials'],TRIALS)
        for key,value in [('trials',TRIALS+[TRIALS[0]]),('windows',[256]),('seed',101)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)
        for key,value in [('max_epochs',50),('precision','fp32'),('learning_rate',.001)]:
            bad=copy.deepcopy(c);bad['training'][key]=value
            with self.assertRaises(ValueError):validate(bad)
        for key,value in [('random_training',True),('pair_max_tokens',256)]:
            bad=copy.deepcopy(c);bad['policy'][key]=value
            with self.assertRaises(ValueError):validate(bad)

    def test_embedding_cache_uses_requested_batch_without_changing_order_or_dimensions(self):
        from rhpr.gr.kindle_pairs import encode_pair_cache
        class Encoder:
            def encode(self,rows,**kwargs):
                self.rows=rows;self.kwargs=kwargs
                return np.arange(len(rows)*768,dtype=np.float32).reshape(len(rows),768)
        e=Encoder()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'vec.npy';encode_pair_cache(e,['A','B'],np.array([1,2]),p,['First movie','Second movie'],batch_size=128,event_prefix='ml1m_pairs')
            self.assertEqual(e.kwargs['batch_size'],128)
            self.assertEqual(e.rows,['First movie: A Second movie: B','First movie: B Second movie: A'])
            self.assertEqual(np.load(p).shape,(2,768))
            with self.assertRaises(ValueError):encode_pair_cache(e,['A'],np.array([0]),p,['a','b'],batch_size=0)

    def test_mixed_screen_reuses_batch_schedule_and_keeps_native_full_and_pair_widths(self):
        from rhpr.gr.kindle_mixed_views import MixedViews

        config = json.loads(Path('experiments/ml1m_last8_mixed128_2t4.json').read_text())
        validate(config)
        self.assertEqual(config['trials'], MIXED_TRIALS)
        altered = copy.deepcopy(config)
        altered['mixed_views']['full_probability'] = .6
        with self.assertRaises(ValueError):
            validate(altered)

        config['training']['batch_size'] = 8
        legacy = MixedViews(np.zeros((129, 128)), np.zeros((129, 96)), batch_size=8)
        order = np.random.default_rng(43).permutation(129)
        expected_full = np.full((129, 128, 4), -1, dtype=np.int32)
        for length in range(1, 129):
            expected_full[length, -length:] = self.sids[:length] + 256 * np.arange(4)
        for merge in MERGE_PREFIXES:
            pair = encode(self.h, self.sids, self.keys, self.pairs, merge)
            views = full_context_selector(self.h, self.sids, pair, config, merge)
            np.testing.assert_array_equal(views.full, expected_full.reshape(129, 512))
            schedule = views.schedule(1)
            np.testing.assert_array_equal(schedule, legacy.schedule(1))
            self.assertEqual(set(schedule), {False, True})
            for batch, begin in enumerate(range(0, len(order), 8)):
                chosen = order[begin:begin + 8]
                for start in range(0, len(chosen), 3):
                    part = chosen[start:start + 3]
                    expected = views.full if schedule[batch] else pair
                    np.testing.assert_array_equal(views(1, batch, part), expected[part])
            report = views.epoch_summary(1)
            self.assertEqual(sum(report['examples'].values()), len(order))
            self.assertEqual(report['native_sid_widths'], {'full128': 512, 'pair128': pair.shape[1]})

    def test_replication_accepts_only_two_new_merge120_fits_and_pinned_parent(self):
        config = json.loads(Path('experiments/ml1m_last8_mixed120_seeds_2t4.json').read_text())
        validate(config)
        self.assertEqual(config['trials'], REPLICATION_TRIALS)
        self.assertEqual(config['kaggle']['kernel_sources'], [PARENT_KERNEL])
        for edited in ('trial', 'parent', 'initializer'):
            with self.subTest(edited=edited):
                altered = copy.deepcopy(config)
                if edited == 'trial':
                    altered['trials'][0]['merge_items'] = 112
                elif edited == 'parent':
                    del altered['frozen_parent']['files']['frozen-indexes/pairs-42.npy']
                else:
                    altered['replication_initializers']['101'] = altered['replication_initializers']['202']
                with self.assertRaises(ValueError):
                    validate(altered)

    def test_extended_mixed64_uses_original_schedule_to_sixty(self):
        config = json.loads(Path('experiments/ml1m_last8_mixed64_extended_2t4.json').read_text())
        prior = json.loads(Path('experiments/ml1m_last8_mixed120_seeds_2t4.json').read_text())
        validate(config)
        self.assertEqual(config['trials'], EXTENDED_TRIALS)
        self.assertEqual(config['mixed_views'], prior['mixed_views'])
        self.assertEqual(config['training']['scheduler'], prior['training']['scheduler'])
        for position in range(len(EXTENDED_TRIALS)):
            altered = copy.deepcopy(config)
            altered['trials'][position]['merge_items'] = 120
            with self.assertRaises(ValueError):
                validate(altered)


if __name__=='__main__':unittest.main()
