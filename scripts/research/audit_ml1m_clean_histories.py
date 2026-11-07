"""CPU audit of raw-timeline coverage and unambiguous heldout protocols."""
from pathlib import Path
import argparse,collections,csv,hashlib,io,json,zipfile
import numpy as np,pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--movielens',type=Path,required=True)
parser.add_argument('--masks',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True);SOURCE=args.movielens
masks=pd.read_parquet(args.masks)
allowed=set(masks.loc[masks['filter'].eq('tmdb_rich4_and_zenodo_plot_vit'),'movie_id'].astype(int))
assert len(allowed)==2874
with zipfile.ZipFile(SOURCE)as z:
    ratings=np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::',',')),delimiter=',',dtype=np.int64)
assert ratings.shape==(1000209,4)
order=np.lexsort((ratings[:,1],ratings[:,3],ratings[:,0]));ratings=ratings[order]
users=np.split(ratings,np.flatnonzero(np.diff(ratings[:,0]))+1)
assert len(users)==6040
lookup=np.zeros(int(ratings[:,1].max())+1,dtype=bool);lookup[list(allowed)]=True
results=collections.defaultdict(list);base_audit=[]

def save(name,uid,g,vg,tg,train_events,train_cases,**kw):
    results[name].append(dict(user_id=uid,validation_time=int(vg),test_time=int(tg),train_events=int(train_events),train_cases=int(train_cases),**kw))

for g in users:
    uid=int(g[0,0]);times,starts,counts=np.unique(g[:,3],return_index=True,return_counts=True)
    valid=lookup[g[:,1]];gv=np.logical_and.reduceat(valid,starts)
    row_group=np.repeat(np.arange(len(times)),counts);singles=counts==1
    bad=np.r_[0,np.cumsum(~valid)]
    # Window start moves to the boundary of the touched timestamp group for
    # eligibility only: an arbitrary within-second item order cannot hide a bad item.
    hstart=np.maximum(starts-20,0);boundary=starts[row_group[hstart]]
    clean20=(starts>=20)&((bad[starts]-bad[boundary])==0)
    clean_short=(starts>0)&((bad[starts]-bad[boundary])==0)
    eligible=gv&singles&clean20
    group_cases20=np.where(gv&clean20,counts,0)
    group_cases_short=np.where(gv&clean_short,counts,0)
    cum_cases20=np.r_[0,np.cumsum(group_cases20)]
    cum_cases_short=np.r_[0,np.cumsum(group_cases_short)]
    # Historical protocol: filter item events then locate latest two singleton groups.
    kept=g[valid];kt,ks,kc=np.unique(kept[:,3],return_index=True,return_counts=True)
    ix=np.flatnonzero(kc==1)
    if len(ix)>=2:
        v,t=ix[-2:]
        if v>=2:
            vt,tt=int(kt[v]),int(kt[t]);vraw=int(np.searchsorted(times,vt));traw=int(np.searchsorted(times,tt))
            n=int(ks[v]);save('current_filtered',uid,g,vt,tt,n,n-int(kc[0]))
            def holes(j):
                i=int(starts[j]);h=max(i-20,0);b=int(starts[row_group[h]])
                return bool(bad[i]-bad[h]),bool(bad[i]-bad[b])
            vh,vhg=holes(vraw);th,thg=holes(traw)
            base_audit.append(dict(user_id=uid,raw_validation_ambiguous=bool(counts[vraw]>1),raw_test_ambiguous=bool(counts[traw]>1),validation_context_has_missing=vh,test_context_has_missing=th,validation_touched_group_has_missing=vhg,test_touched_group_has_missing=thg,validation_has_full20=bool(starts[vraw]>=20),test_has_full20=bool(starts[traw]>=20)))
    # First fully-valid prefix: stop before the first timestamp containing any bad item.
    first_bad=int(np.flatnonzero(~gv)[0])if not gv.all()else len(times)
    si=np.flatnonzero(singles[:first_bad])
    if len(si)>=2:
        v,t=si[-2:]
        if v>=2:
            n=int(starts[v]);save('first_valid_prefix',uid,g,times[v],times[t],n,n-int(counts[0]),raw_prefix_events=int(sum(counts[:first_bad])),val_full20=bool(n>=20))
            if n>=20:save('first_valid_prefix_full20',uid,g,times[v],times[t],n,n-int(counts[0]))
    # Latest uninterrupted run of fully-valid timestamp groups with valid holdouts.
    segments=[];a=None
    for j,ok in enumerate([*gv,False]):
        if ok and a is None:a=j
        elif not ok and a is not None:segments.append((a,j));a=None
    for require20,name in [(False,'latest_clean_run'),(True,'latest_clean_run_full20')]:
        for a,b in reversed(segments):
            si=np.flatnonzero(singles[a:b])+a
            if len(si)<2:continue
            v,t=si[-2:];n=int(starts[v]-starts[a])
            if v-a<2 or(require20 and n<20):continue
            save(name,uid,g,times[v],times[t],n,n-int(counts[a]),segment_events=int(sum(counts[a:b])));break
    # Whole training history may contain barriers, but every admitted example
    # uses an original, clean last20 window. No rejoining across unknown items.
    si=np.flatnonzero(eligible)
    if len(si)>=2:
        v,t=si[-2:]
        if v>=2:
            save('raw_clean_windows_full20',uid,g,times[v],times[t],starts[v],cum_cases20[v],train_cases_allow_short=int(cum_cases_short[v]),raw_training_events=int(starts[v]),validation_index=int(v),test_index=int(t),intervening_raw_events=int(starts[t]-starts[v]-1))
    # Preserve the existing max-window20 policy, permitting naturally short
    # ORIGINAL prefixes only; never manufacture a short history after a barrier.
    si=np.flatnonzero(gv&singles&clean_short)
    if len(si)>=2:
        v,t=si[-2:]
        if v>=2 and cum_cases_short[v]>0:
            save('raw_clean_windows_upto20',uid,g,times[v],times[t],starts[v],cum_cases_short[v],val_full20=bool(starts[v]>=20),test_full20=bool(starts[t]>=20))
    # Literal contiguous [20 contexts, val, test] at some earlier endpoint:
    # val/test are adjacent raw singleton groups, all touched history groups valid.
    pairs=np.flatnonzero(eligible[:-1]&eligible[1:])
    if len(pairs):
        v=int(pairs[-1]);t=v+1
        if v>=2:save('contiguous_20_val_test',uid,g,times[v],times[t],starts[v],cum_cases20[v],train_cases_allow_short=int(cum_cases_short[v]))

# Require at least one actual training example before val, as in the old cohort.
for old,new,field in [('raw_clean_windows_full20','raw_clean_windows_full20_admitted','train_cases_allow_short'),('raw_clean_windows_full20','raw_clean_windows_fixed20_training','train_cases'),('contiguous_20_val_test','contiguous_20_val_test_admitted','train_cases_allow_short')]:
    results[new]=[dict(r,train_cases=r[field])for r in results[old]if r[field]>0]
summary={}
for name,rows in results.items():
    summary[name]=dict(users=len(rows),train_events=sum(r['train_events']for r in rows),train_cases=sum(r['train_cases']for r in rows))
    if 'train_cases_allow_short'in rows[0]:summary[name]['train_cases_allow_short']=sum(r['train_cases_allow_short']for r in rows)
    if 'val_full20'in rows[0]:summary[name]['users_with_full20_validation']=sum(r['val_full20']for r in rows)
    fields=sorted({k for r in rows for k in r})
    with(ROOT/(name+'-users.csv')).open('w')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
assert summary['current_filtered']==dict(users=5672,train_events=852295,train_cases=839388)
audit=dict(users=len(base_audit),counts={k:sum(r[k]for r in base_audit)for k in base_audit[0]if k!='user_id'})
audit['raw_ambiguous_either_target']=sum(r['raw_validation_ambiguous']or r['raw_test_ambiguous']for r in base_audit)
audit['either_context_missing']=sum(r['validation_context_has_missing']or r['test_context_has_missing']for r in base_audit)
audit['either_touched_group_missing']=sum(r['validation_touched_group_has_missing']or r['test_touched_group_has_missing']for r in base_audit)
with(ROOT/'current-filtered-raw-audit.csv').open('w')as f:
    w=csv.DictWriter(f,fieldnames=list(base_audit[0]));w.writeheader();w.writerows(base_audit)
out=dict(mask_sha256=hashlib.sha256(args.masks.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),items=2874,original_users=6040,original_events=1000209,events_with_feature_coverage=int(lookup[ratings[:,1]].sum()),policies=summary,current_raw_audit=audit,notes=['Targets are singleton groups in ORIGINAL raw timeline for all new protocols.','Any timestamp group containing an uncovered film is a barrier, including a group touched at the left edge of a20item context.','Raw-clean training cases require20past raw events; optional short contexts allowed only if their whole available past is covered.','Train-event counts for raw-clean policies are raw rows beforeval, NOT retained valid supervision; use train_cases for compute.','No GPU, training, dataset upload or production protocol change performed.'])
(ROOT/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
