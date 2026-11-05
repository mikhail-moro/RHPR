import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from rhpr.gr.batch_pilot import validate,schedule,slices,adam_step
ROOT=Path(__file__).resolve().parents[1]

class BatchContractTests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'experiments/t5_batch_t4_pilot.json').read_text())
    def test_exact_16_trials_and_paired_backends(self):
        jobs=schedule(self.config());self.assertEqual(len(jobs),16);self.assertEqual(len({j['id']for j in jobs}),16)
        for i in range(0,16,2):
            self.assertEqual(jobs[i]['backend'],'legacy_eager');self.assertEqual(jobs[i+1]['backend'],'sdpa')
            self.assertEqual([(jobs[i][k],jobs[i+1][k])for k in ['batch_size','physical_batch_size','history_items']],[(jobs[i][k],jobs[i][k])for k in ['batch_size','physical_batch_size','history_items']])
    def test_rejects_extension_or_different_matrix(self):
        for key,value in [('timed_steps',10000),('batches',[[2048,2048]]),('backends',['sdpa']),('max_seconds',3600)]:
            c=self.config();c['pilot'][key]=value
            with self.assertRaises(ValueError):validate(c)
    def test_microbatches_cover_effective_batch(self):
        for b,p in [(256,128),(256,256),(512,512),(1024,1024)]:
            parts=slices(b,p);self.assertEqual([j for s in parts for j in range(s.start,s.stop)],list(range(b)))
        with self.assertRaises(ValueError):slices(256,512)
        with self.assertRaises(ValueError):slices(256,100)
    def test_offline_private_sources_preserved(self):
        from scripts.build_kaggle_bundle import build
        with tempfile.TemporaryDirectory()as d:
            build(ROOT/'experiments/t5_batch_t4_pilot.json','carasik/rhpr-t5-batch-pilot','gpu',Path(d)/'bundle')
            m=json.loads((Path(d)/'bundle/kernel-metadata.json').read_text());self.assertTrue(m['is_private']);self.assertFalse(m['enable_internet']);self.assertEqual(m['kernel_sources'],['carasik/rhpr-ml100k-singleton-mtl'])
    def test_run_dispatches_only_requested_pilot(self):
        from rhpr.experiment import run_experiment
        from unittest.mock import patch
        with patch('rhpr.gr.batch_pilot.run_batch_pilot',return_value={'complete':16})as f:
            self.assertEqual(run_experiment(self.config(),Path('/tmp/unused')),{'complete':16});f.assert_called_once()

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
class AccumulationTests(unittest.TestCase):
    def test_split_and_full_batch_have_same_update_without_dropout(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,4)
            def rhpr_loss(self,b):return torch.nn.functional.cross_entropy(self.layer(b['x']),b['y'])
        torch.manual_seed(31);a=Model();b=copy.deepcopy(a);x=torch.randn(8,3);y=torch.tensor([0,1,2,3,1,0,3,2])
        oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=.0001);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=.0001)
        scaler=lambda:torch.amp.GradScaler('cuda',enabled=False)
        la=adam_step(a,oa,[{'x':x,'y':y}],[1.],scaler(),'cpu')
        lb=adam_step(b,ob,[{'x':x[:4],'y':y[:4]},{'x':x[4:],'y':y[4:]}],[.5,.5],scaler(),'cpu')
        torch.testing.assert_close(la,lb)
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)
    def test_nonfinite_loss_does_not_silently_update(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.ones(()))
            def rhpr_loss(self,b):return self.p*float('nan')
        m=Model();o=torch.optim.Adam(m.parameters())
        with self.assertRaises(FloatingPointError):adam_step(m,o,[{}],[1.],torch.amp.GradScaler('cuda',enabled=False),'cpu')
        self.assertEqual(float(m.p.detach()),1.)
