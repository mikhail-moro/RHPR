import copy
import unittest

from rhpr.gr.generator_stability import validate_completion, validate_training_protocol


def config():
    seeds=[31415,27182,161803]
    ids=[f'index_{i}' for i in range(6)]
    return dict(generator_seeds=seeds,index_ids=ids,
        trials=[dict(id=f'{i}_gen{s}',index_id=i,seed=s,architecture='narrow') for i in ids for s in seeds],
        initialization_pins={'narrow':{str(s):'a'*64 for s in seeds}},
        data=dict(history_items=20,train_examples=151166,users=2786,items=2874),
        execution=dict(worker_devices=['cuda:0','cuda:1'],cpu_threads_per_worker=1),
        training=dict(min_epochs=30,max_epochs=60,patience_epochs=20,validation_interval=10,
            validate_first_epoch=False,batch_size=512,physical_batch_size=512,precision='fp16',
            evaluation_precision='fp32',learning_rate=.001,weight_decay=.0001,
            grad_clip_norm=1.0,evaluation_batch_size=32,max_seconds=3600),
        generator=dict(encoder_layers=4,decoder_layers=4,d_model=64,d_ff=512,d_kv=32,num_heads=6,
            dropout=.1,attn_implementation='sdpa',beam_width=20,should_check_prefix=False,
            num_user_bins=2000,codebook_size=256),
        evaluation=dict(save_validation_predictions=True,save_validation_losses=True,test_after_selection=False))


def result(epoch=60,best=60,stop='epoch_limit',skips=0):
    rows=[]
    for e in range(1,epoch+1):
        row=dict(epoch=e,updates=296*e-min(skips,e),amp_skipped_updates=min(skips,e),
                 train_examples=151166,train_loss_sum_over_codes=6.)
        if e%10==0:
            row['validation']=dict(**{'ndcg@10':.1-abs(e-best)*.001})
            row['validation_loss']=dict(total_ce=6.)
        rows.append(row)
    sel=dict(training_epochs=epoch,best_epoch=best,stopping_reason=stop,amp_skipped_updates=skips,
        optimizer_updates=296*epoch-skips,training_examples_per_epoch=151166,training_precision='fp16',
        evaluation_precision='fp32',validation_checks=epoch//10,validation_ndcg10=.1)
    return sel,rows


class GeneratorStabilityTests(unittest.TestCase):
    def test_requires_exact_crossed_design_and_val_only(self):
        c=config();validate_training_protocol(c)
        for mutate in [lambda x:x['trials'].pop(),lambda x:x['training'].update(validation_interval=5),
                       lambda x:x['training'].update(max_epochs=30),lambda x:x['evaluation'].update(test_after_selection=True),
                       lambda x:x['generator'].update(d_model=128)]:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):validate_training_protocol(bad)

    def test_cap_vs_plateau_and_amp_skips(self):
        c=config()
        sel,rows=result(60,60,'epoch_limit',3)
        self.assertEqual(validate_completion(sel,rows,c),dict(plateau_reached=False,budget_censored=True))
        sel,rows=result(40,20,'early_stopping',2)
        self.assertEqual(validate_completion(sel,rows,c),dict(plateau_reached=True,budget_censored=False))
        sel['optimizer_updates']+=1
        with self.assertRaises(RuntimeError):validate_completion(sel,rows,c)

    def test_incomplete_and_fake_plateau_are_rejected(self):
        c=config()
        for epoch,best,stop in [(20,20,'epoch_limit'),(40,30,'early_stopping'),(60,40,'epoch_limit')]:
            sel,rows=result(epoch,best,stop)
            with self.assertRaises(RuntimeError):validate_completion(sel,rows,c)
        sel,rows=result();rows[19].pop('validation')
        with self.assertRaises(RuntimeError):validate_completion(sel,rows,c)


if __name__ == '__main__':unittest.main()
