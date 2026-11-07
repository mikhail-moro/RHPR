import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.prepare_ml1m_clean_mtl import build_clean_cases
from rhpr.gr.ml1m_clean_mtl import case_examples,validate
ROOT=Path(__file__).resolve().parents[1]

class RawCleanCasesTests(unittest.TestCase):
    def rows(self,n):return np.asarray([[1,i,4,i]for i in range(n)],dtype=np.int64)
    def test_literal_adjacent_targets_and_all_earlier_cases(self):
        cases,users=build_clean_cases(self.rows(30),list(range(30)))
        self.assertEqual((users[0]['validation_time'],users[0]['test_time']),(28,29));self.assertEqual(len(cases['train'][2]),27)
        self.assertEqual(cases['validation'][0][0],list(range(8,28)));self.assertEqual(cases['test'][0][0],list(range(9,29)))
    def test_no_splice_across_uncovered_item_and_later_recovery(self):
        ids=[i for i in range(52)if i!=25];cases,users=build_clean_cases(self.rows(52),ids)
        self.assertEqual((users[0]['validation_time'],users[0]['test_time']),(50,51));self.assertEqual(users[0]['train_cases'],28)
        decoded=[ids[i]for i in cases['train'][2]];self.assertEqual(decoded,list(range(1,25))+list(range(46,50)))
    def test_ambiguous_raw_group_cannot_become_singleton_after_filter(self):
        a=self.rows(31);a[29,3]=28
        cases,users=build_clean_cases(a,[i for i in range(31)if i!=29])
        self.assertEqual((users[0]['validation_time'],users[0]['test_time']),(26,27))
    def test_left_boundary_group_is_checked_in_full(self):
        a=self.rows(25);a=np.vstack([a,[1,999,3,0]])
        cases,users=build_clean_cases(a,list(range(25)))
        self.assertEqual(users[0]['train_cases'],2);self.assertEqual(cases['train'][2],[21,22])
    def test_excludes_zero_supervision_before_val(self):
        a=self.rows(23);a=np.vstack([a,[1,999,3,0]])
        cases,users=build_clean_cases(a,list(range(23)));self.assertEqual(users,[])

class ContractTests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'experiments/ml1m_clean_mtl_2t4.json').read_text())
    def test_declared_budget_and_four_methods(self):validate(self.config())
    def replicates(self):return json.loads((ROOT/'experiments/ml1m_clean_mtl_replicates_2t4.json').read_text())
    def test_two_additional_seeds_have_all_methods_and_balanced_workers(self):
        from rhpr.gr.ae_suite import assignments
        c=self.replicates();validate(c)
        jobs=assignments(c,2)
        self.assertEqual([len(j['trials'])for j in jobs],[4,4])
        self.assertEqual({r['seed']for r in c['trials']},{101,202})
        for seed in [101,202]:
            self.assertEqual(len([r for r in c['trials']if r['seed']==seed]),4)
    def test_rejects_duplicate_method_in_place_of_missing_seed_method(self):
        c=self.replicates();c['trials'][-1]['mtl_mode']='sum'
        with self.assertRaises(ValueError):validate(c)
    def test_every_seed_requires_both_initialization_pins(self):
        for branch in ['generator','quantizer']:
            c=self.replicates();pins=c['quantizer']if branch=='quantizer'else c
            del pins['expected_initial_state_sha256_by_seed']['202']
            with self.assertRaises(ValueError):validate(c)
    def test_replicates_preserve_all_non_seed_training_and_data_settings(self):
        original=self.config();c=self.replicates()
        for key in ['training','generator','data','representation','fields','movie_input','execution','evaluation','kaggle','plot_encoder','plot_assets','modern_runtime']:
            self.assertEqual(c[key],original[key])
        q=copy.deepcopy(c['quantizer']);o=copy.deepcopy(original['quantizer'])
        q.pop('expected_initial_state_sha256_by_seed');o.pop('expected_initial_state_sha256_by_seed')
        self.assertEqual(q,o)
    def test_rejects_unapproved_precision_or_protocol_changes(self):
        for branch,key,value in [('training','batch_size',1024),('training','max_epochs',60),('data','users',2787),('generator','attn_implementation','eager')]:
            c=self.config();c[branch][key]=value
            with self.assertRaises(ValueError):validate(c)
    def test_cached_cases_are_immutable_and_split_checked(self):
        with tempfile.TemporaryDirectory()as d:
            np.savez_compressed(Path(d)/'train-cases.npz',contexts=np.zeros((2,20),dtype=np.int64),user_ids=np.array([1,2]),targets=np.array([0,1]))
            x=case_examples(d,20,'train');self.assertFalse(x[0].flags.writeable)
            with self.assertRaises(ValueError):case_examples(d,50,'train')
    def test_private_bundle_declares_all_existing_and_new_sources(self):
        from scripts.build_kaggle_bundle import build
        with tempfile.TemporaryDirectory()as d:
            build(ROOT/'experiments/ml1m_clean_mtl_2t4.json','carasik/rhpr-ml1m-clean-mtl','gpu',Path(d)/'bundle')
            m=json.loads((Path(d)/'bundle/kernel-metadata.json').read_text());self.assertTrue(m['is_private']);self.assertFalse(m['enable_internet']);self.assertEqual(len(m['dataset_sources']),4);self.assertEqual(m['kernel_sources'],[])
