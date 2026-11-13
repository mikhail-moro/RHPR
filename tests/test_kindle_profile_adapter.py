import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_gr_profiling as profile_fixture
from rhpr.gr.profiling import benchmark_training,detailed_training_profile,ProfileDeadline
from rhpr.gr.experiment import _model_batch

@unittest.skipUnless(profile_fixture._HAS_TORCH,'requires torch')
class ProfileAdapterTest(unittest.TestCase):
    def test_custom_sid_batch_builder_is_used_and_profile_restores_weights(self):
        f=profile_fixture.ProfileExecutionTests();f.setUpClass()
        try:
            f.setUp();calls=[]
            def builder(*args):calls.append(len(args[0]));return _model_batch(*args)
            before={k:v.clone()for k,v in f.model.state_dict().items()}
            with patch('rhpr.gr.experiment._model_batch',side_effect=AssertionError('default item builder must not be used')):
                row=benchmark_training(f.model,f.examples,f.index.sids,f.np.arange(8),2,'fp32','full_pipeline',f.training,f.specification,ProfileDeadline(30),'cpu',batch_builder=builder)
                with tempfile.TemporaryDirectory()as tmp:
                    detailed_training_profile(f.model,f.examples,f.index.sids,f.np.arange(8),2,'fp32',f.training,'cpu',Path(tmp),batch_builder=builder,export_trace=True)
                    self.assertTrue((Path(tmp)/'training-profile-trace.json').is_file())
            self.assertTrue(calls);self.assertEqual(row['measured_updates_applied'],2)
            for k,v in f.model.state_dict().items():self.assertTrue(f.torch.equal(before[k],v))
        finally:f.tearDownClass()
