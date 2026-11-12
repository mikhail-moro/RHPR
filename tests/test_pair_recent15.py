import json,unittest
from pathlib import Path
import numpy as np
from rhpr.gr.pair_recent15 import validate,recent_preencode
class Recent15Test(unittest.TestCase):
 def test_protocol_and_last_15(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_recent15_2t4.json').read_text());validate(c)
  ids=np.arange(20*4).reshape(20,4); rows=np.array([list(range(20)),[-1]*17+[2,3,4]])
  actual=recent_preencode(rows,ids);self.assertEqual(actual.shape,(2,60));np.testing.assert_array_equal(actual[0],(ids[5:]+np.arange(4)*256).reshape(-1));self.assertTrue((actual[1,:48]==-1).all())
if __name__=='__main__':unittest.main()
