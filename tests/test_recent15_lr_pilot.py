import copy,json,unittest,importlib.util
from pathlib import Path
from rhpr.gr.recent15_lr_pilot import validate
class Recent15LrPilotTest(unittest.TestCase):
 def test_fixed_single_pilot_and_patience(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_recent15_lr_pilot.json').read_text());validate(c)
  for field,value in [('learning_rate',.001),('patience_epochs',15),('max_epochs',30)]:
   d=copy.deepcopy(c);d['training'][field]=value
   with self.assertRaises(ValueError):validate(d)
  d=copy.deepcopy(c);d['evaluation']['save_validation_losses']=False
  with self.assertRaises(ValueError):validate(d)
 @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
 def test_reduce_after_three_failed_checks(self):
  import torch
  from rhpr.gr.generator_resume import build_scheduler
  c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_recent15_lr_pilot.json').read_text())
  opt=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.002);scheduler=build_scheduler(opt,c['training']['scheduler'])
  for score in (.1,.1,.1):scheduler.step(score)
  self.assertEqual(opt.param_groups[0]['lr'],.002)
  scheduler.step(.1);self.assertEqual(opt.param_groups[0]['lr'],.001)
if __name__=='__main__':unittest.main()
