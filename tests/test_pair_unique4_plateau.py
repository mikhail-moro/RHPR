import copy,json,unittest
from pathlib import Path
from rhpr.gr.pair_unique4_plateau import validate
class PlateauProtocolTest(unittest.TestCase):
 def test_resume_all_three_models_on_three_seeds(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_unique4_plateau_2t4.json').read_text());validate(c);self.assertEqual(len(c['trials']),9)
  for k,v in [('max_epochs',100),('min_epochs',50),('patience_epochs',10),('validation_interval',10)]:
   d=copy.deepcopy(c);d['training'][k]=v
   with self.assertRaises(ValueError):validate(d)
  d=copy.deepcopy(c);d['resume_epoch']=0
  with self.assertRaises(ValueError):validate(d)
if __name__=='__main__':unittest.main()
