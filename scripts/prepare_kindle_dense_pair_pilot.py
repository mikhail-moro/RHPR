"""Fixed popular-catalogue, active-reader pilot; user selection is random."""
import argparse,collections,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np

def prepare(root,output,cached_catalog,target=100000,seed=42,top_items=5000):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pop=collections.Counter()
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            if float(r['rating'])>=4:pop[r['parent_asin']]+=1
    selected=set(sorted(pop,key=lambda i:(-pop[i],i))[:top_items]);cache=json.loads(Path(cached_catalog).read_text());meta={r['parent_asin']:{k:r[k]for k in ('title','categories')}for r in cache if r['parent_asin']in selected}
    missing=selected-set(meta)
    if missing:
        with gzip.open(root/'metadata.jsonl.gz','rt')as f:
            for line in f:
                r=json.loads(line);i=r.get('parent_asin');title=r.get('title')
                if i in missing and isinstance(title,str)and title.strip():meta[i]=dict(title=' '.join(title.split()),categories=r.get('categories')or[])
    print(json.dumps(dict(phase='catalogue',selected=len(selected),with_title=len(meta))),flush=True)
    histories=collections.defaultdict(list)
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            if float(r['rating'])>=4 and r['parent_asin']in meta:histories[r['user_id']].append((int(r['timestamp']),r['parent_asin']))
    users=sorted(u for u,h in histories.items()if len(h)>=22);np.random.default_rng(seed).shuffle(users)
    retained=[];total=0
    for u in users:
        h=sorted(histories[u]);retained.append((u,h));total+=len(h)
        if total>=target:break
    if total<target*.95:raise ValueError('not enough active-reader events')
    if abs(total-len(retained[-1][1])-target)<abs(total-target):total-=len(retained.pop()[1])
    items=sorted({i for u,h in retained for t,i in h});mapping={i:k for k,i in enumerate(items)};cases={s:[[],[],[]]for s in ('train','validation','test')};audit=[]
    for uid,(source_uid,h)in enumerate(retained,start=1):
        seq=[mapping[i]for t,i in h]
        for split,positions in [('train',range(1,len(h)-2)),('validation',[len(h)-2]),('test',[len(h)-1])]:
            for at in positions:
                row=np.full(20,-1,dtype=np.int64);prefix=seq[max(0,at-20):at];row[-len(prefix):]=prefix
                for a,v in zip(cases[split],(row,uid,seq[at])):a.append(v)
        audit.append(dict(user_id=uid,source_user_id=source_uid,events=len(h),train_last_time=h[-3][0],validation_time=h[-2][0],test_time=h[-1][0]))
    for split,a in cases.items():np.savez_compressed(out/(split+'-cases.npz'),**dict(zip(('contexts','user_ids','targets'),(np.asarray(v,dtype=np.int64)for v in a))))
    (out/'texts.json').write_text(json.dumps(['Title: '+meta[i]['title']+'. Categories: '+', '.join(str(v)for v in meta[i]['categories'])for i in items],ensure_ascii=False)+'\n')
    (out/'source-ids.json').write_text(json.dumps(items)+'\n');(out/'catalog.json').write_text(json.dumps([dict(parent_asin=i,**meta[i])for i in items],ensure_ascii=False)+'\n');(out/'user-audit.json').write_text(json.dumps(audit)+'\n')
    trfreq=collections.Counter(cases['train'][2]);trainids=set(trfreq)|{int(i)for h in cases['train'][0]for i in h if i>=0}
    report=dict(users=len(retained),items=len(items),events=total,train_examples=len(cases['train'][1]),rating_min=4,min_interactions=22,selection_seed=seed,
        source='Amazon Reviews2023 official5core rating_only Kindle_Store',
        selection='Top5000catalogue by full static positive-event frequency, then random eligible users with >=22 retained events, nearest whole-user prefix to100K',
        catalogue_selection_uses_static_full_dataset_counts=True,not_a_full_catalogue_or_population_representative_evaluation=True,
        requested_catalogue=top_items,catalogue_with_title=len(meta),eligible_users=len(users),items_with_categories=sum(bool(meta[i]['categories'])for i in items),
        val_test_same_timestamp=sum(a['validation_time']==a['test_time']for a in audit),lengths_quantiles=np.quantile([len(h)for u,h in retained],[0,.25,.5,.75,.9,.95,.99,1]).tolist(),
        validation_cold_targets=sum(t not in trainids for t in cases['validation'][2]),test_cold_targets=sum(t not in trainids for t in cases['test'][2]),unique_train_items=len(trainids),
        train_target_frequency_quantiles=np.quantile([trfreq.get(i,0)for i in range(len(items))],[0,.25,.5,.75,.9,1]).tolist(),tie_break='timestamp then parent_asin',test_includes_val=True)
    (out/'data-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');p.add_argument('cached_catalog');a=p.parse_args();prepare(a.root,a.output,a.cached_catalog)
