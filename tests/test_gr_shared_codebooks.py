import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from rhpr.gr.shared_codebooks import assign,initialize,install,validate
from rhpr.gr.sid import SidIndex


class SharedCodebooksTests(unittest.TestCase):
    def test_zero_lift_preserves_assignments_and_residual_distances(self):
        torch.manual_seed(8);z=torch.randn(10,64);b=torch.randn(3,256,64)
        np.testing.assert_array_equal(assign(z,b),assign(torch.nn.functional.pad(z,(0,64)),torch.nn.functional.pad(b,(0,64))))

    def test_real_parameter_drives_assignments_and_checkpoint_restores_index(self):
        model=torch.nn.Module();model.item_sid_embedding_table_encoder=torch.nn.Embedding(1024,128)
        b=torch.zeros(3,256,64);b[:,1:]=100.;b[0,1]=1.
        initialize(model,b.numpy());latent=np.zeros((2,64),dtype='float32');latent[1]=1.
        index=SidIndex(assign(torch.tensor(latent),torch.tensor(b)))
        with tempfile.TemporaryDirectory()as d:
            install(model,index,latent,True,d);model.rhpr_epoch_start(1,index,Path(d))
            initial=model.rhpr_active_sids.clone()
            loss=model.item_sid_embedding_table_encoder(torch.tensor([1])).sum();loss.backward()
            self.assertTrue(torch.all(model.item_sid_embedding_table_encoder.weight.grad[1]==1))
            with torch.no_grad():model.item_sid_embedding_table_encoder.weight[1].fill_(-100.)
            model.rhpr_epoch_start(2,index,Path(d));self.assertFalse(torch.equal(initial,model.rhpr_active_sids))
            model.rhpr_active_sids.copy_(initial);model.rhpr_after_checkpoint_load(index)
            np.testing.assert_array_equal(index.sids,initial.numpy())
            self.assertEqual(len(index.sid_to_item),2)

    def test_contract_has_exact_two_arms_and_no_shortening(self):
        p=Path(__file__).resolve().parents[1]/'experiments/ml1m_shared_codebooks_2t4.json'
        c=json.loads(p.read_text());validate(c);c['training']['max_epochs']=10
        with self.assertRaises(ValueError):validate(c)
