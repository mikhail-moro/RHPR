"""Label-choice sensitivity of saved MTL rankings, without new model calls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('sequences',type=Path);p.add_argument('embargoed',type=Path);p.add_argument('output',type=Path);a=p.parse_args();load=lambda p:json.loads(p.read_text())
 ids=load(a.raw/'trials/sum_seed42/sid-catalogue.json')['original_item_ids'];im={i:j for j,i in enumerate(ids)};users=load(a.sequences);ui={u['user_id']:j for j,u in enumerate(users)}
 assert hashlib.sha256(a.embargoed.read_bytes()).hexdigest()=='7217258a7c35e48eb09596b9381a4f03d2c17b8de83f968463d4be4ee91dee65'
 groups={s:[[u[s][0]]for u in users]for s in ['validation','test']}
 for r in load(a.embargoed):
  u=users[ui[r['user_id']]];assert r['timestamp']==u[r['split']][1];groups[r['split']][ui[r['user_id']]].append(r['movie_id'])
 N=len(users);assert N==943;models=[];comparisons=[];rg=np.random.default_rng(4204);bootrg=np.random.default_rng(4205)
 for split,gg in groups.items():
  ks=np.array(list(map(len,gg)));choice=rg.integers(0,ks,size=(5000,N));raw={};score={}
  for mode in ['sum','uw','gradnorm','pcgrad']:
   d=a.raw/'trials'/f'{mode}_seed42';r=load(d/'trial-result.json');ep=r['selection']['best_epoch'];sid=np.load(d/'semantic-ids.npy');lookup={tuple(s):i for i,s in enumerate(sid)}
   with np.load(d/('test-predictions.npz'if split=='test'else f'validation-epoch-{ep:03d}.npz')) as z:
    np.testing.assert_array_equal(z['user_ids'],[u['user_id']for u in users]);np.testing.assert_array_equal(z['target_item_ids'],[im[u[split][0]]for u in users]);pred=z['semantic_ids'][:,:10]
   values=np.zeros((N,max(ks)))
   for j,g in enumerate(gg):
    assert len(g)==len(set(g))
    ranking={lookup.get(tuple(s),-1):k+1 for k,s in enumerate(pred[j])}
    for k,i in enumerate(g):
     rank=ranking.get(im[i]);values[j,k]=1/np.log2(rank+1) if rank else 0.
   official=float(values[:,0].mean());expected=r['test']['ndcg@10']if split=='test'else r['selection']['validation_ndcg10'];assert abs(official-expected)<1e-12
   raw[mode]=values;score[mode]=values.sum(1)/ks
   models.append(dict(split=split,mode=mode,original_best_epoch=ep,official_ndcg10=official,uniform_target_expected_ndcg10=float(score[mode].mean()),singleton_ndcg10=float(values[ks==1,0].mean()),singleton_users=int((ks==1).sum())))
  for mode in ['uw','gradnorm','pcgrad']:
   delta=score[mode]-score['sum'];bootstrap=np.array([bootrg.choice(delta,N,replace=True).mean()for _ in range(5000)])
   rd=raw[mode]-raw['sum'];draw=rd[np.arange(N)[None,:],choice].mean(1)
   singlebase=raw['sum'][ks==1,0].mean();singletreatment=raw[mode][ks==1,0].mean()
   comparisons.append(dict(split=split,mode=mode,uniform_delta=float(delta.mean()),uniform_relative=float(score[mode].mean()/score['sum'].mean()-1),conditional_user95=np.quantile(bootstrap,[.025,.975]).tolist(),uniform_label_draw_delta_quantiles=np.quantile(draw,[.025,.5,.975]).tolist(),label_draw_positive_fraction=float((draw>0).mean()),singleton_relative=float(singletreatment/singlebase-1)))
 result=dict(users=N,split_counts={s:dict(singleton=sum(len(g)==1 for g in gg),multiple=sum(len(g)>1 for g in gg))for s,gg in groups.items()},models=models,comparisons=comparisons,scope='Fixed ORIGINAL validation-selected checkpoints and original test histories (train plus original selected val item). Expected NDCG under uniform choice of one target within each timestamp group; diagnostic, not replacement benchmark or checkpoint reselection. Label ranges are conditional sensitivity, not confidence intervals. User CI excludes seed uncertainty, target clustering and multiplicity.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
