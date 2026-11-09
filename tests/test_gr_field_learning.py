import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rhpr.gr.field_learning import expected_trials, make_texts, prepare_texts, run, validate, validate_completion
from rhpr.gr.data import file_sha256


class FieldLearningTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_field_learning_curves_2t4.json').read_text())

    def test_registered_matrix_and_protocol(self):
        c=self.config;validate(c)
        self.assertEqual(len(c['trials']),16)
        self.assertEqual(len({x['index_id'] for x in c['trials']}),12)
        self.assertEqual(len({x['id'] for x in c['trials']}),16)
        self.assertEqual(Counter(x['seed'] for x in c['trials']),{31415:12,27182:2,161803:2})
        for mutate in (lambda x:x['training'].update(validation_interval=5),
                       lambda x:x['evaluation'].update(test_after_selection=True),
                       lambda x:x.update(shuffle_seed=1),
                       lambda x:x['trials'].pop(),
                       lambda x:x['quantizer'].update(codebook_size=128)):
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate(bad)

    def test_field_templates_and_shuffle_match_single_catalogue(self):
        rows=[f'Title: Film {i} (1995); Genres: Drama, Comedy; Year: 1995; Runtime: {i+70} minutes; Keywords: words.' for i in range(20)]
        rows[3]=rows[3].replace('73 minutes','unknown')
        texts,owner,report=make_texts(rows,20261109)
        texts2,owner2,report2=make_texts(rows,20261109)
        self.assertEqual(texts,texts2);np.testing.assert_array_equal(owner,owner2)
        self.assertEqual(report,report2);self.assertEqual(owner[3],3)
        self.assertEqual(sorted(owner.tolist()),list(range(20)))
        self.assertEqual(texts['title'][0],'Title: Film 0 (1995).')
        self.assertEqual(texts['genres'][0],'Title: Film 0 (1995); Genres: Drama, Comedy.')
        values=[r.split('Runtime: ')[1] for r in texts['duration']]
        shuffled=[r.split('Runtime: ')[1] for r in texts['shuffled_duration']]
        self.assertEqual(Counter(values),Counter(shuffled))
        for i in range(len(rows)):self.assertEqual(shuffled[i],values[owner[i]])
        self.assertEqual(texts['duration'][3],texts['shuffled_duration'][3])
        with self.assertRaises(ValueError):make_texts(['malformed'],20261109)

    def test_actual_catalogue_checked_against_source_fields(self):
        source=Path('/tmp/ml1m-clean-mtl-input')
        reference=Path('/workspace/artifacts/movielens-text-preflight/catalogue-fields.json.gz')
        if not (source/'metadata-texts.json').exists() or not reference.exists():
            self.skipTest('external real metadata input unavailable')
        import gzip
        rows=json.loads((source/'metadata-texts.json').read_text());ids=json.loads((source/'source-ids.json').read_text())
        catalogue={r['item_id']:r['fields'] for r in json.load(gzip.open(reference,'rt'))}
        texts,owner,report=make_texts(rows,20261109)
        self.assertEqual(report['unknown'],5);self.assertGreater(report['changed_value'],2400)
        for i,item_id in enumerate(ids):
            f=catalogue[item_id];self.assertEqual(texts['title'][i],f"Title: {f['title']}.")
            self.assertEqual(texts['genres'][i],f"Title: {f['title']}; Genres: {', '.join(f['genres'])}.")
            self.assertEqual(texts['duration'][i],f"Title: {f['title']}; Runtime: {f['runtime']}.")

    def test_completion_distinguishes_early_stopping_and_budget_cap(self):
        c=self.config
        def specimen(epoch,best,stop):
            curves=[]
            for e in range(1,epoch+1):
                row={'epoch':e}
                if e%10==0:row.update(validation={'ndcg@10':.5 if e==best else .1},validation_loss={'ce_by_level':[1.,2.,3.,4.]})
                curves.append(row)
            sel=dict(training_epochs=epoch,best_epoch=best,stopping_reason=stop,optimizer_updates=296*epoch,
                amp_skipped_updates=0,training_examples_per_epoch=151166)
            return sel,curves
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('generator-best.pt','generator-last.pt','generator.json'):(root/name).touch()
            for e in range(10,101,10):(root/f'validation-epoch-{e:03d}.npz').touch()
            sel,curves=specimen(60,30,'early_stopping')
            self.assertEqual(validate_completion(sel,curves,c,root),{'plateau_reached':True,'budget_censored':False})
            sel,curves=specimen(100,90,'epoch_limit')
            self.assertEqual(validate_completion(sel,curves,c,root),{'plateau_reached':False,'budget_censored':True})
            sel['training_epochs']=80
            with self.assertRaises(RuntimeError):validate_completion(sel,curves,c,root)

    def test_orchestration_uses_all_indices_and_never_evaluates_test(self):
        import sys
        c=copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();(data/'source-ids.json').write_text('[1,2]')
            c['resources']['scratch_path']=str(root/'scratch')
            roots={key:root/key for key in ('resources','modern_runtime')};roots['movie_input']=data
            def resolve(value):return next(roots[k] for k in roots if c[k]['path']==value)
            def indices(*args,**kwargs):
                return {f'{field}_q{seed}':{'sids_sha256':'a'*64} for field in
                    ('title','genres','duration','shuffled_duration') for seed in (42,101,202)}
            def workers(path,*,worker_module,event_prefix):
                plan=json.loads(Path(path).read_text())
                self.assertEqual(worker_module,'rhpr.gr.field_learning')
                self.assertEqual([len(x['trials']) for x in plan['assignments']],[8,8])
                self.assertFalse(plan['config']['evaluation']['test_after_selection'])
                return {'results':{x['id']:{'plateau_reached':False,'budget_censored':True}
                                   for x in c['trials']}}
            with patch('torch.cuda.device_count',return_value=2), \
                 patch('torch.cuda.get_device_capability',return_value=(7,5)), \
                 patch('rhpr.offline_runtime.resolve_dataset_directory',side_effect=resolve), \
                 patch('rhpr.gr.resources.verify_resources'), \
                 patch('rhpr.gr.resources.unpack_grid',return_value=root/'original'), \
                 patch('rhpr.gr.grid.prepare_grid_source',return_value=root/'grid'), \
                 patch('rhpr.gr.field_learning.case_examples',side_effect=lambda data,history,split:split), \
                 patch('rhpr.gr.field_learning.case_digest',side_effect=lambda split:c['data']['case_sha256'][split]), \
                 patch('rhpr.gr.field_learning.prepare_texts',return_value=([1,2],{},{})), \
                 patch('rhpr.gr.field_learning.prepare_indices',side_effect=indices), \
                 patch('rhpr.gr.ae_suite.launch_workers',side_effect=workers), \
                 patch('rhpr.gr.field_learning.subprocess.run'), \
                 patch.object(sys,'path',sys.path[:]):
                result=run(c,root/'output')
            self.assertEqual(result['models'],16)
            self.assertEqual(result['indexes'],12)
            self.assertFalse(result['test_evaluated'])


if __name__=='__main__':unittest.main()
