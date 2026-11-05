import json,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
from scripts.prepare_ml1m_als import train_users
from rhpr.gr.ml1m_clean_mtl import validate
from rhpr.gr.ml1m_als import attach_channel
ROOT=Path(__file__).resolve().parents[1]
class ALSTests(unittest.TestCase):
 def test_only_supplied_train_edges_and_no_sliding_window_multiplicity(self):
  users=train_users(np.array([[-1,1],[1,2],[1,2]]),np.array([7,7,7]),np.array([2,3,3]))
  self.assertEqual(len(users),1);np.testing.assert_array_equal(users[0].train,[1,2,3])
 def test_screen_requires_real_and_shuffled_matched_seed(self):
  c=json.loads((ROOT/'experiments/ml1m_als_channel_2t4.json').read_text());validate(c)
  c['trials'][1]['cf_variant']='als'
  with self.assertRaises(ValueError):validate(c)
 def test_unsupported_items_remain_masked_and_source_hash_required(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);np.save(p/'als.npy',np.array([[1.,2.],[0.,0.]],dtype='float32'));np.save(p/'observed.npy',np.array([True,False]));(p/'als-report.json').write_text('{}')
   sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
   (p/'resources-manifest.json').write_text(json.dumps(dict(files=[dict(path=f.name,size_bytes=f.stat().st_size,sha256=sha(f))for f in p.iterdir()])))
   arrays={'people':np.zeros((2,2),dtype='float32')};masks={}
   report=attach_channel(arrays,masks,p,'als',sha(p/'resources-manifest.json'));self.assertEqual(report['observed_items'],1);self.assertFalse(masks['people'][1])
   with self.assertRaises(ValueError):attach_channel(arrays,masks,p,'als','0'*64)

class ObjectiveTests(unittest.TestCase):
 def test_objective_screen_requires_two_distinct_als_objectives(self):
  c=json.loads((ROOT/'experiments/ml1m_als_objectives_2t4.json').read_text());validate(c)
  c['trials'][1]['mtl_mode']='gradnorm'
  with self.assertRaises(ValueError):validate(c)
 def test_zero_cf_objective_preserves_inputs_and_avoids_weight_override(self):
  from rhpr.gr.ml1m_als import configure_objective
  spec={'mtl_mode':'sum','zero_inactive_slots':[]};trial={'cf_objective':'no_cf_reconstruction'}
  configure_objective(spec,trial)
  self.assertEqual(spec['reconstruction_weights'],{'people':0.});self.assertEqual(spec['zero_inactive_slots'],[]);self.assertNotIn('mtl_mode',spec)
