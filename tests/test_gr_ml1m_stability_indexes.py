"""Frozen-index provenance and modality ablation invariants."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.ml1m_stability_indexes import _aligned_features, _checked_sids, stage_stability_indexes
from rhpr.gr.sid import SidIndex


class StabilityIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.data = self.root / 'data'; self.data.mkdir()
        self.features = self.root / 'features'; self.features.mkdir()
        self.ids = [f'film-{i}' for i in range(2874)]
        for root in [self.data, self.features]: (root / 'source-ids.json').write_text(json.dumps(self.ids))
        n = len(self.ids)
        ids = np.arange(n, dtype=np.int64)
        self.codes = np.stack([ids // 256, ids % 256, np.zeros(n, dtype=np.int64)], axis=1)
        index = SidIndex(self.codes, width=256)
        self.parents = {}
        for seed in [42, 101]:
            path = self.root / f'parent-{seed}.npy'
            np.save(path, index.sids, allow_pickle=False)
            self.parents[seed] = dict(path=path, sha256=file_sha256(path))
        np.savez_compressed(self.features / 'features.npz', text=np.ones((n,768),np.float32),
            poster=np.ones((n,3),np.float32), plot=np.ones((n,4),np.float32), people=np.zeros((n,2),np.float32),
            text_observed=np.ones(n,bool),poster_observed=np.ones(n,bool),plot_observed=np.ones(n,bool),people_observed=np.zeros(n,bool))
        self.feature_sha = file_sha256(self.features / 'features.npz')
        self.spec = dict(num_hierarchies=3,codebook_size=256,max_steps=3000,head_dim=128,head_hidden=256,
            ae_warmup={'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}},
            expected_initial_state_sha256_by_seed={'42':'a'*64,'101':'b'*64})

    def tearDown(self): self.tmp.cleanup()

    def test_refuses_mismatched_catalogue_and_control_bytes(self):
        self.assertEqual(len(_aligned_features(self.features,self.data,self.feature_sha)[0]),2874)
        (self.features/'source-ids.json').write_text(json.dumps(list(reversed(self.ids))))
        with self.assertRaisesRegex(ValueError,'catalogue'):
            _aligned_features(self.features,self.data,self.feature_sha)
        with self.assertRaisesRegex(ValueError,'pinned'):
            _checked_sids(self.parents[42]['path'],'0'*64,2874)

    def test_stages_six_distinct_index_provenance_records(self):
        calls=[]
        def fake_kmeans(text,spec,seed,directory):
            calls.append(('kmeans',seed,spec.copy(),np.asarray(text).copy()))
            return SidIndex(self.codes,width=256), np.zeros(1)
        def fake_rqvae(arrays,masks,spec,seed,device,source,recipe,directory,active):
            calls.append(('rqvae',seed,spec.copy(),list(active)))
            return self.codes, dict(steps=3000,warmup={'steps':3500},quantizer_initial_state_sha256=spec['expected_initial_state_sha256_by_seed'][str(seed)])
        kw=dict(output=self.root/'out',features_root=self.features,data_root=self.data,source='source',original='original',
            quantizer_spec=self.spec,multimodal_sources=self.parents,expected_features_sha256=self.feature_sha,device='cpu')
        with patch('rhpr.gr.sid_study.fit_quantizer',fake_kmeans), patch('rhpr.gr.multimodal_rqvae.fit_multimodal_rqvae',fake_rqvae):
            refs=stage_stability_indexes(**kw)
        self.assertEqual(set(refs),{f'{m}_seed{s}' for m in ['meta_kmeans','meta_rqvae','multimodal_rqvae'] for s in [42,101]})
        self.assertEqual([c[1] for c in calls],[42,42,101,101])
        for call in calls:
            if call[0]=='kmeans':
                self.assertEqual(call[2],dict(levels=3,codebook_size=256,n_init=1,max_iter=100,cpu_threads=2))
                self.assertEqual(call[3].shape,(2874,768))
            else:
                self.assertEqual(call[2]['zero_inactive_slots'],['poster','plot','people'])
                self.assertEqual(call[2]['mtl_mode'],'sum');self.assertEqual(call[3],['text'])
        for seed in [42,101]:
            name=f'multimodal_rqvae_seed{seed}'
            self.assertEqual(file_sha256(self.root/'out'/(name+'-semantic-ids.npy')),self.parents[seed]['sha256'])
        with self.assertRaises(FileExistsError): stage_stability_indexes(**kw)


if __name__=='__main__': unittest.main()
