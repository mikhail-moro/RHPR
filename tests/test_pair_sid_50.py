import copy,importlib.util,json,unittest
from pathlib import Path
from rhpr.gr.pair_sid_50 import validate,TRIALS
from rhpr.gr.ae_suite import assignments

class PairSid50Protocol(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_sid_50_2t4.json').read_text())
    def test_three_matched_seeds_and_only_101_resumed(self):
        validate(self.c)
        self.assertEqual({t['id'] for t in TRIALS}, {f'{m}_{s}' for m in ('control','pair') for s in (42,101,202)})
        self.assertEqual({t['id'] for t in TRIALS if t['resume']},{'control_101','pair_101'})
        jobs=assignments(self.c,2)
        self.assertEqual([len(j['trials']) for j in jobs],[3,3])
        self.assertEqual({t['id'] for j in jobs for t in j['trials']},{t['id'] for t in TRIALS})
    def test_rejects_wrong_training_or_checkpoint_policy(self):
        for edit in ('epoch','validation','seed','resume','test','quantizer'):
            c=copy.deepcopy(self.c)
            if edit=='epoch':c['training']['max_epochs']=49
            if edit=='validation':c['training']['validation_interval']=10
            if edit=='seed':c['seeds']=[42,101,303]
            if edit=='resume':c['trials'][2]['resume']=False
            if edit=='test':c['evaluation']['test_after_selection']=True
            if edit=='quantizer':c['quantizer']['passes_per_level']=11
            with self.subTest(edit=edit),self.assertRaises(ValueError):validate(c)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
    def test_prediction_uses_same_direct_cdist_backend_as_fitting(self):
        import numpy as np
        import torch
        from unittest.mock import patch
        from rhpr.gr.pair_sid import predict_rq
        original = torch.cdist
        calls = []
        def checked(*args, **kwargs):
            calls.append(kwargs.get('compute_mode'))
            return original(*args, **kwargs)
        values = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        centers = np.tile(np.array([[[1., 0.], [0., 1.]]], dtype=np.float32), (3, 1, 1))
        with patch.object(torch, 'cdist', side_effect=checked):
            result = predict_rq(values, centers, 'cpu', batch_size=2)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(calls, ['donot_use_mm_for_euclid_dist'] * 3)

if __name__=='__main__':unittest.main()
