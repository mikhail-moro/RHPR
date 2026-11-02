import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_full_training_seed42_2t4.json'


class MovieContinuationTest(unittest.TestCase):
    def test_extra_seeds_retain_all_four_variants_and_reject_lineage_mismatches(self):
        from rhpr.gr.movie_continuation import validate
        from rhpr.gr.ae_suite import assignments
        c=json.loads((ROOT/'experiments/movielens100k_full_training_extra_seeds_2t4.json').read_text());validate(c)
        jobs=assignments(c,2)
        self.assertEqual([[t['predictor'] for t in j['trials']].count('tiger') for j in jobs],[3,3])
        self.assertEqual({t['seed'] for t in c['trials']},{101,202})
        for mutation in ['seed','source_trial','predictor','parent','missing_variant','initialization']:
            bad=copy.deepcopy(c)
            if mutation=='seed':bad['trials'][0]['seed']=202
            elif mutation=='source_trial':bad['trials'][0]['source_trial']='seed42_stage_03'
            elif mutation=='predictor':bad['trials'][0]['predictor']='sasrec'
            elif mutation=='parent':bad['trials'][0]['parent']='controls'
            elif mutation=='missing_variant':bad['trials'].pop()
            else:bad['compare_initialization_by_seed']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(bad)

    def test_run_trial_uses_checkpoint_seed_for_both_predictors_and_initialization_groups(self):
        import numpy as np,torch
        from unittest.mock import patch
        from rhpr.gr.movie_continuation import run_trial
        from rhpr.gr.sid import SidIndex
        from rhpr.gr.sid_study import state_digest
        base=json.loads(CONFIG.read_text())
        for seed,predictor in [(101,'tiger'),(202,'sasrec')]:
            with self.subTest(seed=seed,predictor=predictor),tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                root=Path(tmp);source=root/'parent';source.mkdir();c=copy.deepcopy(base);c['seed']=seed;c['quantizer']={'codebook_size':256}
                model=torch.nn.Linear(2,2);initial=state_digest(dict(model.named_parameters()))
                (source/'generator.json').write_text(json.dumps(dict(initial_state_sha256=initial)))
                torch.save(dict(epoch=20,updates=6680,config=c),source/'generator-last.pt')
                codes=np.array([[0,0,0],[1,0,0]])
                np.save(source/'semantic-codes.npy',codes);np.save(source/'semantic-ids.npy',SidIndex(codes,width=256).sids)
                trial=dict(id=f'{predictor}_seed{seed}',seed=seed,predictor=predictor,parent='ae',source_trial='source',files={})
                plan=dict(directories={trial['id']:str(source)},output=str(root/'out'),source='grid')
                selection=dict(training_epochs=30,stopping_reason='early_stopping')
                example=(None,np.array([1]),np.array([0]))
                with patch('rhpr.gr.grid.build_grid_generator',return_value=model) as tiger,patch('rhpr.gr.movie_sasrec.build_movie_sasrec',return_value=model) as sasrec,patch('rhpr.gr.experiment._train_generator',return_value=selection) as train,patch('rhpr.gr.movie_data.temporal_examples',return_value=example),patch('rhpr.gr.experiment._evaluate',return_value=({'ndcg@10':0.1},np.zeros((1,1),dtype=int),np.zeros((1,1)))):
                    result=run_trial(trial,c,plan,[],'cpu')
                    builder=tiger if predictor=='tiger' else sasrec
                    self.assertEqual(builder.call_args.args[2 if predictor=='tiger' else 1],seed)
                    self.assertEqual(train.call_args.args[3]['seed'],seed)
                    self.assertEqual(result['seed'],seed)
                    self.assertEqual(result['initialization_group'],f'{predictor}_seed{seed}')
                    self.assertEqual(train.call_args.kwargs['resume_from'],source)

    def test_four_frozen_seed42_continuations_and_bounded_policy(self):
        from rhpr.gr.movie_continuation import validate
        from rhpr.gr.ae_suite import assignments
        c=json.loads(CONFIG.read_text());validate(c)
        jobs=assignments(c,2);self.assertEqual([len(j['trials']) for j in jobs],[2,2])
        self.assertEqual([t['predictor'] for t in jobs[0]['trials']],['tiger','tiger'])
        with self.assertRaises(RuntimeError):assignments(c,1)
        for key in ['min_epochs','max_epochs','validation_interval','patience_epochs']:
            bad=copy.deepcopy(c);bad['training'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['trials'][0]['files'].pop('generator-last.pt')
        with self.assertRaises(ValueError):validate(bad)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires pinned GRID')
    def test_native_tiger_resume_restores_optimizer_and_four_level_validation(self):
        script=r'''
import copy,json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.sid import SidIndex
from rhpr.gr.experiment import _train_generator
from rhpr.gr.movie_data import TemporalUser,temporal_examples
root=Path(sys.argv[1]);source=prepare_grid_source(sys.argv[2],root/'grid');torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text());c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(max_epochs=4,min_epochs=1,patience_epochs=20,precision='fp32',evaluation_precision='fp32',batch_size=4,physical_batch_size=2,evaluation_batch_size=2,max_seconds=120)
c['training']['scheduler']['start_epoch']=3
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
index=SidIndex(np.column_stack([np.arange(16),np.zeros(16,dtype=int),np.zeros(16,dtype=int)]),width=16)
for name in ['full','part','resumed']:(root/name).mkdir()
build=lambda:build_grid_generator(index.sids,c['generator'],42,'cpu',source)
_train_generator(build(),users,index,c,root/'full','cpu',example_builder=temporal_examples)
part=copy.deepcopy(c);part['training']['max_epochs']=2
_train_generator(build(),users,index,part,root/'part','cpu',example_builder=temporal_examples)
r=_train_generator(build(),users,index,c,root/'resumed','cpu',example_builder=temporal_examples,resume_from=root/'part')
a=torch.load(root/'full/generator-last.pt',weights_only=False);b=torch.load(root/'resumed/generator-last.pt',weights_only=False)
assert a['updates']==b['updates']==4
assert a['scheduler']==b['scheduler']
for name in a['model']:assert torch.equal(a['model'][name],b['model'][name]),name
rows=json.loads((root/'resumed/training-curves.json').read_text())
assert len(rows)==4 and all(len(x['validation_loss']['ce_by_level'])==4 for x in rows)
assert r['resumed_from_epoch']==2 and r['continuation_epochs']==2
assert json.loads((root/'resumed/resume-verification.json').read_text())['rng_restored_and_verified']
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            r=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
