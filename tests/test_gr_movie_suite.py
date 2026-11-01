import importlib.util
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class MovieSuiteContract(unittest.TestCase):
    def test_two_extra_pipeline_seeds_preserve_the_data_and_budget(self):
        from rhpr.gr.movie_suite import validate
        from rhpr.gr.ae_suite import assignments
        old=json.loads((ROOT/'experiments/movielens100k_text_ladder_2t4.json').read_text())
        new=json.loads((ROOT/'experiments/movielens100k_text_ladder_seeds_2t4.json').read_text())
        validate(new)
        for key in ['training','generator','representation','fields','execution','data','movie_input','quantizer','kaggle']:
            self.assertEqual(old[key],new[key])
        self.assertEqual(new['replicate_seeds'],[101,202])
        jobs=assignments(new,2)
        self.assertEqual([len(j['trials']) for j in jobs],[7,7])
        for job,seed in zip(jobs,new['replicate_seeds']):
            self.assertEqual([t['stage'] for t in job['trials']],list(range(1,8)))
            self.assertEqual({t['seed'] for t in job['trials']},{seed})
        for mutation in ['missing_trial','duplicate_pair','missing_seed_hash','wrong_default_hash']:
            bad=copy.deepcopy(new)
            if mutation=='missing_trial':bad['trials'].pop()
            elif mutation=='duplicate_pair':bad['trials'][-1]['stage']=6
            elif mutation=='missing_seed_hash':bad['expected_initial_state_sha256_by_seed'].pop('202')
            else:bad['expected_initial_state_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(bad)

    def test_100k_switch_changes_data_not_training_recipe(self):
        from rhpr.gr.movie_suite import validate
        old=json.loads((ROOT/'experiments/movielens_text_ladder_2t4.json').read_text())
        new=json.loads((ROOT/'experiments/movielens100k_text_ladder_2t4.json').read_text())
        validate(new)
        for key in ['training','generator','representation','fields','execution','trials']:
            self.assertEqual(old[key],new[key])
        self.assertEqual(new['data']['items'],1273)
        self.assertEqual(new['data']['users'],943)
        self.assertEqual(new['data']['train_examples'],85470)
        self.assertEqual(new['quantizer']['init_buffer_size'],1273)

    def test_fixed_epochs_shared_encoder_and_approved_subset(self):
        from rhpr.gr.movie_suite import validate
        c=json.loads((ROOT/'experiments/movielens_text_ladder_2t4.json').read_text())
        validate(c)
        self.assertEqual(c['data']['items'],2875)
        self.assertEqual(c['data']['users'],6039)
        self.assertEqual(c['quantizer']['init_buffer_size'],2875)
        self.assertEqual(c['quantizer']['codebook_size'],256)
        self.assertEqual(c['quantizer']['num_hierarchies'],3)
        c['representation']['max_length']=256
        with self.assertRaises(ValueError):validate(c)


@unittest.skipUnless(os.environ.get('RHPR_SENTENCE_MODEL') and os.environ.get('RHPR_GRID_SOURCE'),
                     'requires pinned local Sentence-T5 and GRID')
class NativeMovieSmoke(unittest.TestCase):
    def test_real_encoder_quantizer_and_twenty_epoch_temporal_generator(self):
        script=r'''
import json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.movie_suite import run_trial
from rhpr.gr.movie_data import TemporalUser
from rhpr.gr.sid_study import state_digest
root=Path(sys.argv[1]);original=Path(sys.argv[2]);resources=Path(sys.argv[3])
source=prepare_grid_source(original,root/'grid');torch.set_num_threads(1)
c=json.load(open(sys.argv[4]));c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=8,evaluation_batch_size=2,max_seconds=120)
c['quantizer'].update(codebook_size=16,init_buffer_size=16,batch_size=16,max_steps=2,initialization_steps=1)
data=root/'data';data.mkdir();rows=[dict(item_id=i,text=f'Title: Movie {i} (1995).') for i in range(16)]
(data/'catalog-stage-01.json').write_text(json.dumps(rows))
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16)
model=build_grid_generator(dummy,c['generator'],101,'cpu',source)
expected=state_digest(dict(model.named_parameters()));del model
plan=dict(output=str(root/'out'),data_root=str(data),source_ids=list(range(16)),resources=str(resources),original=str(original),initial_state_sha256='0'*64,initial_state_sha256_by_seed={'101':expected})
r=run_trial(dict(id='smoke',stage=1,seed=101),c,plan,users,source,'cpu')
assert r['selection']['training_epochs']==20 and r['selection']['validation_checks']==4
assert r['selection']['training_examples_per_epoch']==4
assert r['initial_state_sha256']==expected and r['quantizer_steps']==2
assert r['seed']==101 and c['seed']==42
assert json.loads((root/'out/trials/smoke/quantizer.json').read_text())['seed']==101
assert torch.load(root/'out/trials/smoke/generator-last.pt',weights_only=False)['config']['seed']==101
assert (root/'out/trials/smoke/test-predictions.npz').is_file()
assert (root/'out/trials/smoke/generator-last.pt').is_file()
'''
        with tempfile.TemporaryDirectory() as d:
            # Resource path contains sentence-t5-base, as on Kaggle.
            resources=Path(os.environ['RHPR_SENTENCE_MODEL']).parent
            p=subprocess.run([sys.executable,'-c',script,d,os.environ['RHPR_GRID_SOURCE'],str(resources),
                              str(ROOT/'experiments/movielens_text_ladder_2t4.json')],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
