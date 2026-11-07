import ast,copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_actor_full_2t4.json'


class ActorFullTest(unittest.TestCase):
    def test_selected_candidate_three_seed_full_protocol(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        jobs=movie_assignments(c,2);self.assertEqual([len(j['trials']) for j in jobs],[3,3])
        old=json.loads((ROOT/'experiments/movielens100k_actor_screen_2t4.json').read_text())
        for key in ['actor_graph_input','data','generator','modality_input','fields']:
            self.assertEqual(c[key],old[key])
        for change in ['weight','graph','epochs','seeds','scheduler']:
            bad=copy.deepcopy(c)
            if change=='weight':bad['trials'][2]['reconstruction_weights']['people']=.3
            elif change=='graph':bad['actor_graph_input']['people_sha256']='0'*64
            elif change=='epochs':bad['training']['max_epochs']=30
            elif change=='seeds':bad['replicate_seeds']=[42,101]
            else:bad['training']['scheduler']['start_epoch']=31
            with self.assertRaises(ValueError):validate(bad)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires native GRID')
    def test_native_actor_full_preserves_weight_and_verified_resume(self):
        # Reuse the native actor-input fixture, changing only the full training
        # schedule and the frozen selected candidate; assertions cover both.
        tree=ast.parse((ROOT/'tests/test_gr_movie_actor_screen.py').read_text())
        script=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
        script=script.replace("c['training'].update(device=", "c['training'].update(min_epochs=26,max_epochs=26,device=")
        script=script.replace("'actor_w03'", "'actor_w01'")
        script=script.replace(" and r['fixed_epoch_screen'] and not r['budget_censored']", " and not r['fixed_epoch_screen']")
        script=script.replace("==30 and r['selection']['validation_checks']==6", "==26 and r['selection']['validation_checks']==10")
        script=script.replace("==[5,10,15,20,25,30]", "==[5,10,15,20]+list(range(21,27))")
        script=script.replace("==.3", "==.1")
        script+="\nassert r['selection']['resumed_from_epoch']==20\nassert json.loads((p/'resume-verification.json').read_text())['rng_restored_and_verified']\nassert (p/'prefix-training-curves.json').exists()\n"
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
