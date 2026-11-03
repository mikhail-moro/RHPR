import copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from rhpr.gr.movie_context_training import validate,verify_schedules
from rhpr.gr.movie_data import TemporalUser,temporal_examples,case_digest
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_context30_40_30epochs.json'


class MovieContextTrainingTest(unittest.TestCase):
    def test_pilot_matrix_and_original_unmasked_fp16_protocol_are_fixed(self):
        c=json.loads(CONFIG.read_text());validate(c)
        self.assertEqual(len(c['trials']),4)
        for change in ['mask','budget','window','seed','attention','batch']:
            b=copy.deepcopy(c)
            if change=='mask':b['known_item_filter']=True
            elif change=='budget':b['training']['max_epochs']=40
            elif change=='window':b['trials'][0]['history_items']=100
            elif change=='seed':b['replicate_seeds']=[42,101]
            elif change=='attention':b['generator']['attention_backend']='sdpa'
            else:b['training']['physical_batch_size']=64
            with self.assertRaises(ValueError):validate(b)

    def test_longer_windows_preserve_each_target_user_and_recent_suffix(self):
        u=TemporalUser(7,tuple(range(45)),tuple(range(45)),45,45,46,46)
        hashes={s:case_digest(temporal_examples([u],20,s))for s in ['train','validation','test']}
        for h in [30,40]:
            r=verify_schedules([u],h,hashes)
            self.assertEqual(r['train']['examples'],44)
            self.assertTrue(all(x['targets_and_users_unchanged']and x['last20_suffix_unchanged']for x in r.values()))
            ex=temporal_examples([u],h,'test');np.testing.assert_array_equal(ex[0][0],np.arange(46-h,46))
        hashes['test']='bad'
        with self.assertRaises(ValueError):verify_schedules([u],30,hashes)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires native GRID')
    def test_native_fresh_predictor_frozen_index_and_exact20epoch_resume_to30(self):
        import torch
        from rhpr.gr.movie_context_training import run_trial
        from rhpr.gr.grid import build_grid_generator
        from rhpr.gr.sid import SidIndex
        from rhpr.gr.sid_study import state_digest
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
            root=Path(tmp);source=root/'parent/trials/none_seed42';source.mkdir(parents=True)
            codes=np.array([[0,0],[0,1],[1,0],[1,1],[2,0]]);index=SidIndex(codes,width=4)
            np.save(source/'semantic-codes.npy',codes);np.save(source/'semantic-ids.npy',index.sids);(source/'sid-catalogue.json').write_text(json.dumps(dict(original_item_ids=list(range(5)))))
            c=json.loads(CONFIG.read_text());c['generator'].update(codebook_size=4,beam_width=4,d_model=8,d_ff=16,d_kv=4,num_heads=1,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=20)
            m=build_grid_generator(index.sids,c['generator'],42,'cpu',os.environ['RHPR_GRID_SOURCE']);initial=state_digest(dict(m.named_parameters()));del m
            (source/'trial-result.json').write_text(json.dumps(dict(initial_state_sha256=initial,selection={'best_epoch':20},test={'ndcg@10':.1})))
            (source/'training-curves.json').write_text(json.dumps([dict(epoch=i,validation={'ndcg@10':.2})for i in [5,10,15,20,*range(21,31)]]))
            c['expected_initial_state_sha256_by_seed']={'42':initial};c['training'].update(device='cpu',precision='fp32',amp=False,batch_size=2,physical_batch_size=2,evaluation_batch_size=2,max_seconds=100)
            c['data'].update(items=5,users=3);users=[TemporalUser(u,(0,1,2),(1,2,3),3,4,4,5)for u in [1,2,3]]
            trial=c['trials'][0];plan=dict(parent=str(root/'parent'),output=str(root/'output'),source=os.environ['RHPR_GRID_SOURCE'])
            r=run_trial(trial,c,plan,users,'cpu');out=root/'output/trials'/trial['id'];selection=r['selection']
            self.assertEqual(selection['training_epochs'],30);self.assertEqual(selection['validation_checks'],14);self.assertEqual(selection['resumed_from_epoch'],20)
            self.assertTrue(r['generator_fresh']and r['index_frozen']);self.assertFalse(r['known_item_filter']);self.assertTrue(r['budget_censored'])
            self.assertEqual(r['reference20_matched_budget_validation']['checks'],[5,10,15,20,*range(21,31)])
            provenance=json.loads((out/'frozen-index-source.json').read_text());self.assertFalse(provenance['learned_generator_weights_loaded']);self.assertFalse(provenance['quantizer_retrained'])
            resume=json.loads((out/'resume-verification.json').read_text());self.assertTrue(resume['rng_restored_and_verified'])
            np.testing.assert_array_equal(np.load(out/'semantic-ids.npy'),index.sids)


if __name__=='__main__':unittest.main()
