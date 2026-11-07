import copy
import json
import unittest
from pathlib import Path

from rhpr.gr.mini_tiger import validate, protocol_shape
from rhpr.gr.proxy_plateau import summarize, validate_completion


class ProxyPlateauTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_proxy_plateau_2t4.json').read_text())

    def test_fixed_six_resumes_and_no_training_hyperparameter_changes(self):
        c=self.config();validate(c)
        self.assertEqual(protocol_shape(c),(['full'],[42,101,202],100))
        self.assertEqual(len(c['trials']),6)
        self.assertEqual(c['training']['validation_interval'],5)
        for key in ['lr','epochs','patience','test','seed','checkpoint']:
            with self.subTest(key=key):
                c=self.config()
                if key=='lr':c['training']['learning_rate']=.0001
                if key=='epochs':c['training']['max_epochs']=200
                if key=='patience':c['training']['patience_epochs']=5
                if key=='test':c['evaluation']['test_after_selection']=True
                if key=='seed':c['trials'][0]['seed']=42
                if key=='checkpoint':c['continuation_parent']['artifacts'][c['trials'][0]['id']].pop('generator-last.pt')
                with self.assertRaises(ValueError):validate(c)

    def test_selected_target_and_censored_runs_are_explicit(self):
        import tempfile
        c=self.config();records={};refs={}
        for j,t in enumerate(c['trials']):
            records[t['id']]={**t,'selection':{'validation_ndcg10':.1+j*.01},'epochs':{'30':.2-j*.01},'plateau_reached':j<3,'budget_censored':j>=3}
            refs[t['index_id']]={'epochs':{'30':.1+j*.01}}
        with tempfile.TemporaryDirectory() as d:
            m=summarize(c,records,refs,Path(d));r=json.loads((Path(d)/'correlations.json').read_text())
            self.assertEqual((m['plateau_reached'],m['budget_censored']),(3,3))
            self.assertAlmostEqual(r['estimators']['same_full30']['spearman'],-1.)
            self.assertFalse(m['test_evaluated'])

    def test_amp_skips_count_as_attempted_batches_not_missing_training(self):
        c=self.config()
        for epochs,skipped in [(60,0),(65,2),(100,7)]:
            s=dict(resumed_from_epoch=30,training_epochs=epochs,stopping_reason='early_stopping',
                   training_examples_per_epoch=151166,optimizer_updates=296*epochs-skipped,amp_skipped_updates=skipped)
            validate_completion(s,c)
            s['optimizer_updates']-=1
            with self.assertRaises(RuntimeError):validate_completion(s,c)
        s.update(optimizer_updates=296*epochs-7,stopping_reason='time_budget')
        with self.assertRaises(RuntimeError):validate_completion(s,c)

    def test_recovery_reuses_four_and_only_two_pending_models_can_train(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_proxy_plateau_finish_2t4.json').read_text());validate(c)
        done=set(c['recovery_parent']['completed_trials'])
        self.assertEqual(len(done),4)
        self.assertEqual({t['index_id'] for t in c['trials'] if t['id'] not in done},{'sum_seed202','gradnorm_seed202'})
        c['recovery_parent']['completed_trials'].pop()
        with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
