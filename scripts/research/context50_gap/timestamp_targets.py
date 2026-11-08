"""Saved-ranking diagnostic for arbitrary target choice inside timestamp groups."""
import json,hashlib
from pathlib import Path
import numpy as np
from rhpr.gr.movie_seen_serving import decode_paths
P=Path('/tmp/movie-actor-full-completed');O=Path('/tmp/context50-split-gap');ids=json.loads((P/'trials/none_seed42/sid-catalogue.json').read_text())['original_item_ids'];imap={i:j for j,i in enumerate(ids)};users=json.loads(Path('/tmp/movie-graph-channel-audit/sequences.json').read_text());N=len(users);ui={u['user_id']:j for j,u in enumerate(users)};ep=Path('/tmp/context50-split-gap-inputs/embargoed-events.json');assert hashlib.sha256(ep.read_bytes()).hexdigest()=='7217258a7c35e48eb09596b9381a4f03d2c17b8de83f968463d4be4ee91dee65';emb=json.loads(ep.read_text());groups={s:[[u[s][0]]for u in users]for s in ['validation','test']}
for r in emb:
 u=users[ui[r['user_id']]];assert r['timestamp']==u[r['split']][1];groups[r['split']][ui[r['user_id']]].append(r['movie_id'])
for split,gg in groups.items():
 for u,g in zip(users,gg):
  assert len(g)==len(set(g))and all(i in imap for i in g);assert not set(g)&{i for i,t in u['train']}
  if split=='test':assert u['validation'][0]not in g
maxk=max(len(g)for gg in groups.values()for g in gg);res=[];mat={};seedids=[42,101,202];disc=1/np.log2(np.arange(1,11)+1)
for h in [40,50]:
 for split in ['validation','test']:
  ks=np.array(list(map(len,groups[split])));values={};raw={}
  for seed in seedids:
   if h==50:root=Path('/tmp/context50-completed')if seed==42 else Path('/tmp/context50-replication-completed')
   else:root=Path('/tmp/movie-context-completed')if seed==42 else Path('/tmp/context40-replication-completed')
   for cand in ['none','actor_w01']:
    p=root/'trials'/f'{cand}_L{h}_seed{seed}';r=json.loads((p/'trial-result.json').read_text());e=r['selection']['best_epoch'];sid=np.load(p/'semantic-ids.npy')
    with np.load(p/('test-predictions.npz'if split=='test'else f'validation-epoch-{e:03d}.npz'))as z:
     np.testing.assert_array_equal(z['user_ids'],[u['user_id']for u in users]);np.testing.assert_array_equal(z['target_item_ids'],[imap[u[split][0]]for u in users]);d=decode_paths(z['semantic_ids'],sid)[:,:10]
    score=np.zeros((N,maxk));hits=np.zeros(N)
    for j,g in enumerate(groups[split]):
     rank={int(i):k for k,i in enumerate(d[j])if i>=0}
     for k,i in enumerate(g):
      pos=rank.get(imap[i]);score[j,k]=0 if pos is None else disc[pos]
     hits[j]=(score[j]>0).sum()
    raw[cand,seed]=score;ex=score.sum(1)/ks;expected=float(ex.mean());multindcg=float((score.sum(1)/np.array([disc[:min(k,10)].sum()for k in ks])).mean());values[cand,seed]=ex
    res.append(dict(history_items=h,split=split,candidate=cand,seed=seed,official_ndcg10=float(score[:,0].mean()),uniform_target_expected_ndcg10=expected,multi_relevant_ndcg10=multindcg,recall_group10=float((hits/ks).mean()),any_group_hit10=float((hits>0).mean())))
  a=np.mean([raw['actor_w01',s]for s in seedids],axis=0);b=np.mean([raw['none',s]for s in seedids],axis=0);mat[h,split]=(a-b,ks)
rg=np.random.default_rng(5040303);bootrg=np.random.default_rng(5040301);W=bootrg.multinomial(N,np.ones(N)/N,size=4000)/N;comparisons=[];labelmeans={}
for h in [40,50]:
 for split in ['validation','test']:
  delta,ks=mat[h,split];expected=delta.sum(1)/ks;choice=rg.integers(0,ks,size=(4000,N));draw=delta[np.arange(N)[None,:],choice].mean(1);labelmeans[h,split]=draw
  base=np.mean([r['uniform_target_expected_ndcg10']for r in res if r['history_items']==h and r['split']==split and r['candidate']=='none']);graph=np.mean([r['uniform_target_expected_ndcg10']for r in res if r['history_items']==h and r['split']==split and r['candidate']=='actor_w01'])
  comparisons.append(dict(history_items=h,split=split,control_uniform_expected=float(base),graph_uniform_expected=float(graph),delta_uniform_expected=float(expected.mean()),relative_uniform_expected=float(graph/base-1),delta_by_seed=[next(r['uniform_target_expected_ndcg10']for r in res if r['history_items']==h and r['split']==split and r['candidate']=='actor_w01'and r['seed']==s)-next(r['uniform_target_expected_ndcg10']for r in res if r['history_items']==h and r['split']==split and r['candidate']=='none'and r['seed']==s)for s in seedids],conditional_user95=np.quantile(W@expected,[.025,.975]).tolist(),label_choice_delta_quantiles=np.quantile(draw,[.025,.5,.975]).tolist(),label_draw_positive_fraction=float((draw>0).mean()),singleton_users=int((ks==1).sum()),multitarget_users=int((ks>1).sum()),official_delta_singletons=float(delta[ks==1,0].mean()),official_delta_multitarget=float(delta[ks>1,0].mean()),uniform_expected_delta_multitarget=float(expected[ks>1].mean())))
summary=dict(users=N,embargoed_events=len(emb),split_groups={s:dict(singleton=int(sum(len(g)==1 for g in groups[s])),multitarget=int(sum(len(g)>1 for g in groups[s])),mean_size=float(np.mean(list(map(len,groups[s])))),max_size=max(map(len,groups[s])))for s in groups},comparison=comparisons,per_model=res,scope='Alternative labels evaluated only against existing saved top10 rankings; same fitted checkpoints and original test context. Uniform target expectation changes the evaluation target-choice convention and is not the official benchmark. Multi-relevant metrics are a different objective; all ratings treated as interactions, not preference labels. Label-draw ranges are conditional sensitivity, not confidence intervals or new model runs.')
(O/'timestamp-target-sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='per_model'},indent=2))
