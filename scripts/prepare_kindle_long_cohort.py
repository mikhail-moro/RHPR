"""Prepare the fixed 800K-candidate (66-event, 5-user) Kindle core, no GPU."""
from __future__ import annotations
import argparse, array, collections, csv, gzip, hashlib, json, time
from pathlib import Path
import numpy as np
from rhpr.gr.kindle_cohort import candidate_users, core_mask, prefix_cases

RATINGS_SHA = '0b56484ab8e0e86403f0265779ce9b826f82fe8852975be6fee96765aa5fcb7e'
METADATA_SHA = '743be77d95e16cc341e9b3a005325a89ec327e93ff86255b32232967d7f8a9ca'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def event(phase,**kw):print(json.dumps(dict(phase=phase,**kw)),flush=True)

def prepare(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for filename,expected in [('ratings.csv.gz',RATINGS_SHA),('metadata.jsonl.gz',METADATA_SHA)]:
        if digest(root/filename)!=expected:raise ValueError('source digest mismatch: '+filename)
    counts=collections.Counter()
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            if float(r['rating'])>=4:counts[r['user_id']]+=1
    names=sorted(u for u,n in counts.items()if n>=66);user_map={u:i for i,u in enumerate(names)};del counts
    u=array.array('I');i=array.array('I');t=array.array('Q');item_map={}
    with gzip.open(root/'ratings.csv.gz','rt',newline='')as f:
        for r in csv.DictReader(f):
            uid=user_map.get(r['user_id'])
            if uid is not None and float(r['rating'])>=4:
                asin=r['parent_asin']
                if asin not in item_map:item_map[asin]=len(item_map)
                u.append(uid);i.append(item_map[asin]);t.append(int(r['timestamp']))
    U=np.asarray(u,dtype=np.int32);I=np.asarray(i,dtype=np.int32);T=np.asarray(t,dtype=np.int64);asins=np.array(list(item_map));del u,i,t,user_map,item_map
    sample=candidate_users(U);U,I,T=U[sample],I[sample],T[sample]
    candidate_report=dict(users=len(np.unique(U)),events=len(U),items=len(np.unique(I)),seed=42,event_budget=800000)
    mask,steps=core_mask(U,I);U,I,T=U[mask],I[mask],T[mask]
    event('initial_core',users=len(np.unique(U)),events=len(U),items=len(np.unique(I)),iterations=steps)
    wanted=set(asins[np.unique(I)]);meta={}
    with gzip.open(root/'metadata.jsonl.gz','rt')as f:
        for number,line in enumerate(f,1):
            r=json.loads(line);asin=r.get('parent_asin');title=r.get('title')
            if asin in wanted and isinstance(title,str)and title.strip():
                categories=r.get('categories')or[]
                if not isinstance(categories,list):raise ValueError('unexpected category schema')
                meta[asin]=dict(title=' '.join(title.split()),categories=categories)
            if number%200000==0:event('metadata_progress',rows=number,matched=len(meta),wanted=len(wanted))
    observed=np.array([a in meta for a in asins]);mask=observed[I];missing_events=int((~mask).sum());U,I,T=U[mask],I[mask],T[mask]
    if not len(U):raise ValueError('empty after title join')
    mask,final_steps=core_mask(U,I);U,I,T=U[mask],I[mask],T[mask]
    order=np.lexsort((asins[I],T,U));U,I,T=U[order],I[order],T[order]
    item_ids=sorted(set(asins[I]));mapping={v:j for j,v in enumerate(item_ids)}
    compact=np.array([mapping[a]for a in asins[I]],dtype=np.int32)
    ends=np.r_[np.flatnonzero(U[1:]!=U[:-1])+1,len(U)];begins=np.r_[0,ends[:-1]]
    histories=[compact[a:b]for a,b in zip(begins,ends)]
    cases=prefix_cases(histories)
    for split,arrays in cases.items():np.savez_compressed(output/(split+'-cases.npz'),**dict(zip(('contexts','user_ids','targets'),arrays)))
    np.savez_compressed(output/'histories.npz',items=compact,offsets=np.r_[0,ends],timestamps=T)
    audits=[dict(user_id=j+1,source_user_id=names[int(U[a])],events=int(b-a),train_last_time=int(T[b-3]),validation_time=int(T[b-2]),test_time=int(T[b-1]))for j,(a,b)in enumerate(zip(begins,ends))]
    write(output/'source-ids.json',item_ids);write(output/'user-audit.json',audits)
    write(output/'catalog.json',[dict(parent_asin=a,**meta[a])for a in item_ids])
    write(output/'texts.json',['Title: '+meta[a]['title']+'. Categories: '+', '.join(str(c)for c in meta[a]['categories'])for a in item_ids])
    trainitems=set(int(v)for h in histories for v in h[:-2]);trainfreq=collections.Counter(map(int,cases['train'][2]));lengths=np.array([len(h)for h in histories])
    pairs=np.unique(U.astype(np.int64)*(int(I.max())+1)+I);support=np.bincount(pairs%(int(I.max())+1))
    report=dict(users=len(histories),items=len(item_ids),events=len(compact),train_examples=len(cases['train'][2]),history_items=64,
        positive_rating_min=4,min_user_events=66,min_item_distinct_users=5,observed_min_item_users=int(support[support>0].min()),
        source='Amazon Reviews2023 official5core rating_only Kindle_Store',candidate_sample=candidate_report,
        initial_core_iterations=steps,post_metadata_core_iterations=final_steps,missing_title_items=len(wanted-set(meta)),missing_title_events=missing_events,
        validation_cold_targets=sum(int(v)not in trainitems for v in cases['validation'][2]),test_cold_targets=sum(int(v)not in trainitems for v in cases['test'][2]),
        validation_no_train_target=sum(trainfreq[int(v)]==0 for v in cases['validation'][2]),test_no_train_target=sum(trainfreq[int(v)]==0 for v in cases['test'][2]),
        val_test_same_timestamp=sum(a['validation_time']==a['test_time']for a in audits),
        lengths_quantiles=np.quantile(lengths,[0,.25,.5,.75,.9,.95,.99,1]).tolist(),
        split='leave-last-two timestamp then parent_asin; test includes observed validation item',
        filtering='Unique maximal (user_events>=66,item_distinct_users>=5) induced subgraph of the fixed candidate sample, intersect title-present catalogue, repeat maximal core',
        catalogue_and_core_use_static_full_histories=True,not_population_representative=True,metadata_rows=number)
    write(output/'data-report.json',report)
    write(output/'source-provenance.json',dict(ratings=dict(url='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Kindle_Store.csv.gz',sha256=RATINGS_SHA),metadata=dict(url='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Kindle_Store.jsonl.gz',sha256=METADATA_SHA),script='scripts/prepare_kindle_long_cohort.py'))
    event('prepared',**report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args();prepare(a.root,a.output)
