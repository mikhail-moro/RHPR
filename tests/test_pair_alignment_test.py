import copy,json,unittest
from pathlib import Path
from rhpr.gr.pair_alignment_test import validate
from rhpr.gr.pair_alignment import VARIANTS,TRIALS
class AlignmentTestStudy(unittest.TestCase):
    def test_all_predeclared_variants_and_no_training(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_alignment_test_2t4.json').read_text());validate(c)
        self.assertEqual(c['variants'],list(VARIANTS));self.assertEqual(c['trials'],TRIALS)
        for key,value in [('training_forbidden',False),('evaluation_split','validation'),('selection_policy','pick_by_test')]:
            d=copy.deepcopy(c);d[key]=value
            with self.assertRaises(ValueError):validate(d)
if __name__=='__main__':unittest.main()
