"""Uniformly shuffle positive-history users; retain ~100K events and real text."""
import argparse,collections,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np

def prepare(root,output,target=100000,seed=42):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter()
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            if float(r['rating'])>=4:counts[r['user_id']]+=1
    users=sorted(u for u,n in counts.items()if n>=5);rng=np.random.default_rng(seed);rng.shuffle(users)
    pool=[];total=0
    for u in users:
        pool.append(u);total+=counts[u]
        if total>=target*4:break
    print(json.dumps(dict(phase='candidate_users',users=len(pool),events=total)),flush=True)
    selected=set(pool);histories=collections.defaultdict(list);wanted=set()
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            if r['user_id']in selected and float(r['rating'])>=4:
                histories[r['user_id']].append((int(r['timestamp']),r['parent_asin']));wanted.add(r['parent_asin'])
    print(json.dumps(dict(phase='candidate_items',items=len(wanted))),flush=True)
    meta={};seen=0
    with gzip.open(root/'metadata.jsonl.gz','rt')as f:
        for line in f:
            r=json.loads(line);seen+=1;asin=r.get('parent_asin');title=r.get('title')
            if asin in wanted and isinstance(title,str)and title.strip():
                cat=r.get('categories')or[]
                if not isinstance(cat,list):raise ValueError('unexpected category schema')
                meta[asin]={'title':' '.join(title.split()),'categories':cat}
            if seen%100000==0:print(json.dumps(dict(phase='metadata_progress',rows=seen,matched=len(meta))),flush=True)
    retained=[];total=0
    for u in pool:
        h=sorted((t,i)for t,i in histories[u]if i in meta)
        if len(h)<5:continue
        retained.append((u,h));total+=len(h)
        if total>=target:break
    if total<target*.95:raise ValueError('candidate pool insufficient after metadata join')
    # Stop at the nearest whole-user prefix; never trim a selected user history.
    if len(retained)>1 and abs(total-len(retained[-1][1])-target)<abs(total-target):total-=len(retained.pop()[1])
    items=sorted({i for u,h in retained for t,i in h});mapping={v:i for i,v in enumerate(items)}
    cases={s:[[],[],[]]for s in ('train','validation','test')};audit=[]
    for uid,(source_uid,h)in enumerate(retained,start=1):
        seq=[mapping[i]for t,i in h]
        for split,positions in [('train',range(1,len(h)-2)),('validation',[len(h)-2]),('test',[len(h)-1])]:
            for at in positions:
                context=np.full(20,-1,dtype=np.int64);prefix=seq[max(0,at-20):at];context[-len(prefix):]=prefix
                for a,v in zip(cases[split],(context,uid,seq[at])):a.append(v)
        audit.append(dict(user_id=uid,source_user_id=source_uid,events=len(h),train_last_time=h[-3][0],validation_time=h[-2][0],test_time=h[-1][0]))
    for split,arrays in cases.items():
        h,u,t=(np.asarray(a,dtype=np.int64)for a in arrays);np.savez_compressed(out/(split+'-cases.npz'),contexts=h,user_ids=u,targets=t)
    texts=['Title: '+meta[i]['title']+'. Categories: '+', '.join(str(v)for v in meta[i]['categories'])for i in items]
    (out/'texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n');(out/'source-ids.json').write_text(json.dumps(items)+'\n')
    (out/'catalog.json').write_text(json.dumps([dict(parent_asin=i,**meta[i])for i in items],ensure_ascii=False)+'\n')
    (out/'user-audit.json').write_text(json.dumps(audit)+'\n')
    trainids=set(cases['train'][2])|{int(i)for h in cases['train'][0]for i in h if i>=0}
    report=dict(users=len(retained),items=len(items),events=total,train_examples=len(cases['train'][1]),min_interactions=5,rating_min=4,selection_seed=seed,
        source='Amazon Reviews2023 official5core rating_only Kindle_Store; no second item-core after user sampling',
        selection='Uniform random permutation of users with >=5positive ratings; first whole histories totalling approximately100K after title join',
        metadata_candidate_items=len(wanted),metadata_candidate_items_with_title=len(meta),metadata_rows=seen,
        items_with_categories=sum(bool(meta[i]['categories'])for i in items),val_test_same_timestamp=sum(a['validation_time']==a['test_time']for a in audit),
        lengths_quantiles=np.quantile([len(h)for u,h in retained],[0,.25,.5,.75,.9,.95,.99,1]).tolist(),
        users_history_before_val_at_least20=sum(len(h)>=22 for u,h in retained),
        validation_cold_targets=sum(t not in trainids for t in cases['validation'][2]),test_cold_targets=sum(t not in trainids for t in cases['test'][2]),
        unique_train_items=len(trainids),train_target_singletons=sum(n==1 for n in collections.Counter(cases['train'][2]).values()),
        tie_break='timestamp then parent_asin',test_includes_val=True,
        sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()for f in ['ratings.csv.gz','metadata.jsonl.gz']})
    (out/'data-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args();prepare(a.root,a.output)
