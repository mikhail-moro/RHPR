import copy
import json
import tempfile
import unittest
from pathlib import Path

from rhpr.gr.ae_suite import assignments
from rhpr.gr.mini_tiger import MIDDLE_ARCHITECTURES,protocol_shape,validate
from rhpr.gr.proxy_architectures import summarize_middle


class MiddleProxyTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_middle_proxy_2t4.json').read_text())

    def test_crossed_shapes_seeds_and_frozen_six_candidates(self):
        c=self.config();validate(c)
        self.assertEqual(protocol_shape(c),(['middle_width','half_depth'],[42,101,202],10))
        jobs=assignments(c,2);self.assertEqual([len(j['trials']) for j in jobs],[12,12])
        self.assertEqual([{t['seed'] for t in j['trials']} for j in jobs],[{31415},{27182}])
        self.assertEqual(MIDDLE_ARCHITECTURES['middle_width'],dict(encoder_layers=4,decoder_layers=4,d_model=96,d_ff=768,num_heads=6,d_kv=48))
        self.assertEqual(MIDDLE_ARCHITECTURES['half_depth'],dict(encoder_layers=2,decoder_layers=2,d_model=128,d_ff=1024,num_heads=6,d_kv=64))

    def test_reject_extra_compute_test_or_reference_tuning(self):
        for k in ['epoch','deadline','lr','shape','matrix','reference','test']:
            c=self.config()
            if k=='epoch':c['training']['max_epochs']=20
            if k=='deadline':c['suite_deadline_seconds']=20000
            if k=='lr':c['training']['learning_rate']=.01
            if k=='shape':c['trials'][0]['architecture']='narrow'
            if k=='matrix':c['trials'].pop()
            if k=='reference':c['plateau_reference']['indexes'][c['index_ids'][0]]['selected_ndcg10']+=.001
            if k=='test':c['evaluation']['test_after_selection']=True
            with self.subTest(k=k),self.assertRaises(ValueError):validate(c)

    def test_fixed10_against_plateau_not_the_early_proxy_peak(self):
        c=self.config();records={}
        for j,i in enumerate(c['index_ids']):c['plateau_reference']['indexes'][i]['selected_ndcg10']=(j+1)*.01
        for t in c['trials']:
            j=c['index_ids'].index(t['index_id']);records[t['id']]={**t,'trial':t['id'],'epochs':{'5':.9-j*.01,'10':(j+1)*.01}}
        with tempfile.TemporaryDirectory() as d:
            m=summarize_middle(c,records,Path(d));r=json.loads((Path(d)/'correlations.json').read_text())
            self.assertEqual(m['completed'],24)
            self.assertAlmostEqual(r['groups']['all']['mini']['middle_width_epoch10']['mean']['spearman'],1.)
            self.assertAlmostEqual(r['groups']['all']['mini']['middle_width_epoch5']['mean']['spearman'],-1.)
            self.assertFalse(r['test_evaluated'])
            records.pop(next(iter(records)))
            with self.assertRaises(ValueError):summarize_middle(c,records,Path(d))


if __name__=='__main__':unittest.main()
