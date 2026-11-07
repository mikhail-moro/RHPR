import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rhpr.gr.ml1m_stability_suite import INDEX_IDS, summarize, validate, run_stability


class StabilitySuiteTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_tokenizer_generator_stability_2t4.json').read_text())

    def test_exact_crossed_matrix_and_protocol_fail_closed(self):
        c=self.config();validate(c)
        self.assertEqual(c['index_ids'],list(INDEX_IDS));self.assertEqual(len(c['trials']),18)
        for mutation in ('validation','test','features','index','seed','recipe','parent','budget'):
            q=copy.deepcopy(c)
            if mutation=='validation':q['training']['validation_interval']=5
            if mutation=='test':q['evaluation']['test_after_selection']=True
            if mutation=='features':q['feature_sha256']='0'*64
            if mutation=='index':q['index_ids'][0]='wrong_seed42'
            if mutation=='seed':q['generator_seeds']=[31415,27182]
            if mutation=='recipe':q['quantizer']['max_steps']=2000
            if mutation=='parent':q['parents']['42']['kernel']='other/repo'
            if mutation=='budget':q['suite_deadline_seconds']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(q)

    def test_reports_fixed30_and_selected_separately_and_censoring(self):
        c=self.config();rows={}
        for t in c['trials']:
            seed=t['seed'];value={31415:.1,27182:.2,161803:.3}[seed]
            rows[t['id']]=dict(index_id=t['index_id'],seed=seed,architecture='narrow',
                epochs={'30':value},selection={'validation_ndcg10':value+.01,'best_epoch':30,'training_epochs':60},
                plateau_reached=seed==31415,budget_censored=seed!=31415)
        with tempfile.TemporaryDirectory() as d:
            m=summarize(c,rows,Path(d));r=json.loads((Path(d)/'stability-summary.json').read_text())
            self.assertEqual((m['completed'],m['plateau_reached'],m['budget_censored']),(18,6,12))
            a=r['indexes'][c['index_ids'][0]]
            self.assertAlmostEqual(a['fixed_epoch30_mean'],.2)
            self.assertAlmostEqual(a['selected_mean'],.21)
            self.assertAlmostEqual(a['fixed_epoch30_sample_std'],.1)
            self.assertFalse(r['test_evaluated'])
            rows.pop(next(iter(rows)))
            with self.assertRaises(ValueError):summarize(c,rows,Path(d))

    def test_complete_control_flow_uses_staged_indexes_and_val_only(self):
        """Exercise real orchestration with IO/GPU replaced and real config."""
        c=self.config();ids=[11,12];seeds=c['generator_seeds']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();(data/'source-ids.json').write_text(json.dumps(ids))
            parents={}
            for seed in ('42','101'):
                p=root/seed;f=p/'features/shared-modalities';f.mkdir(parents=True)
                (p/'plan.json').write_text(json.dumps({'source_ids':ids,'config':{'data':c['data']},
                    'features_sha256':c['feature_sha256']}));parents[seed]=p
            roots={'resources':root/'resource','movie_input':data,'modern_runtime':root/'modern'}
            def resolve(path):
                return next(roots[k] for k in roots if path==c[k]['path'])
            def parent(spec):return parents['42' if spec['kernel'].endswith('clean-mtl') else '101']
            def stage(output,feature_root,data_root,source,original,qspec,sources,*,expected_features_sha256,device):
                self.assertEqual(data_root,data);self.assertEqual(device,'cuda:0')
                self.assertEqual(set(sources),{42,101});self.assertEqual(expected_features_sha256,c['feature_sha256'])
                return {i:{'sids_sha256':'a'*64} for i in INDEX_IDS}
            def workers(plan_path,*,worker_module,event_prefix):
                p=json.loads(Path(plan_path).read_text())
                self.assertEqual(worker_module,'rhpr.gr.generator_stability')
                self.assertEqual([len(x['trials']) for x in p['assignments']],[9,9])
                rows={}
                for t in c['trials']:
                    rows[t['id']]=dict(index_id=t['index_id'],seed=t['seed'],architecture='narrow',epochs={'30':.05},
                        selection={'validation_ndcg10':.05,'best_epoch':30,'training_epochs':60},
                        plateau_reached=False,budget_censored=True)
                return {'results':rows}
            with mock.patch('torch.cuda.device_count',return_value=2), \
                 mock.patch('torch.cuda.get_device_capability',return_value=(7,5)), \
                 mock.patch('rhpr.offline_runtime.resolve_dataset_directory',side_effect=resolve), \
                 mock.patch('rhpr.gr.resources.verify_resources'), \
                 mock.patch('rhpr.gr.resources.verify_parent_artifacts',side_effect=parent), \
                 mock.patch('rhpr.gr.resources.unpack_grid',return_value=root/'grid-original'), \
                 mock.patch('rhpr.gr.grid.prepare_grid_source',return_value=root/'grid-prepared'), \
                 mock.patch('rhpr.gr.ml1m_clean_mtl.case_examples',side_effect=lambda data,history,split:split), \
                 mock.patch('rhpr.gr.movie_data.case_digest',side_effect=lambda split:c['data']['case_sha256'][split]), \
                 mock.patch('rhpr.gr.ml1m_stability_suite.file_sha256',return_value=c['feature_sha256']), \
                 mock.patch('rhpr.gr.ml1m_stability_indexes.stage_stability_indexes',side_effect=stage), \
                 mock.patch('rhpr.gr.ml1m_stability_suite.subprocess.run'), \
                 mock.patch('rhpr.gr.ae_suite.launch_workers',side_effect=workers), \
                 mock.patch.object(sys,'path',sys.path[:]):
                c['resources']['scratch_path']=str(root/'scratch')
                result=run_stability(c,root/'output')
            self.assertEqual(result['completed'],18)
            self.assertEqual(result['budget_censored'],18)
            self.assertFalse(result['test_evaluated'])


if __name__=='__main__':unittest.main()
