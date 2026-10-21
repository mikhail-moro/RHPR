from __future__ import annotations
# BEGIN ARCHIVE PATH ADAPTER: no research logic below this block is changed.
import os as _archive_os
from pathlib import Path as _ArchivePath

_archive_repo_root = _ArchivePath(__file__).resolve().parents[4]
_archive_workspace_root = _ArchivePath(
    _archive_os.environ.get("RHPR_RESEARCH_WORKSPACE", "/workspace")
).expanduser().resolve()
if (_archive_workspace_root == _archive_repo_root
        or _archive_repo_root in _archive_workspace_root.parents):
    raise ValueError("RHPR_RESEARCH_WORKSPACE must be outside the repository checkout")


def _archive_workspace_path(relative):
    return str(_archive_workspace_root / relative)


def _archive_repo_src_path():
    return str(_archive_repo_root / "src")
# END ARCHIVE PATH ADAPTER

from collections import Counter, defaultdict
import hashlib,json,math,time,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
sys.path.insert(0,_archive_repo_src_path())
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.daily_evaluation_data import daily_context_sha256

started=time.monotonic()
recipe_bytes=Path(_archive_workspace_path('music-data-audit-recipe.json')).read_bytes()
recipe=json.loads(recipe_bytes)
cohort=load_prepared_cohort(_archive_workspace_path('plateau-cohort'))
summary=json.loads(Path(_archive_workspace_path('baseline-evaluation/daily-evaluation-summary.json')).read_text())
low,high=summary['train_end_timestamp'],summary['validation_end_timestamp']
anchor_table=pq.read_table(_archive_workspace_path('baseline-evaluation/daily-anchors.parquet'))
anchor_table=anchor_table.filter(pc.and_(pc.greater_equal(anchor_table['target_timestamp'],low),pc.less(anchor_table['target_timestamp'],high)))
anchors=anchor_table.to_pylist();first_by_key={(r['user_id'],r['day']):r for r in anchors}
future_table=pq.read_table(_archive_workspace_path('baseline-evaluation/daily-future.parquet'))
# File also stores later test observations; truncate on read before constructing any analysis targets.
future={}
for i in range(future_table.num_rows):
    uid=int(future_table['uid'][i].as_py())
    ts=future_table['future_timestamps'][i].values.to_numpy(zero_copy_only=False)
    end=int(np.searchsorted(ts,high,side='left'))
    future[uid]=(future_table['future_items'][i].values.to_numpy(zero_copy_only=False)[:end],ts[:end])
del future_table
print(json.dumps({'phase':'loaded','seconds':time.monotonic()-started}),flush=True)
counts=np.zeros(cohort.num_items+1,dtype=np.int64)
for user in cohort.users:
    ids,ct=np.unique(user.train,return_counts=True);counts[ids]+=ct
catalogue=np.arange(1,cohort.num_items+1,dtype=np.int32)
global_order=catalogue[np.lexsort((catalogue,-counts[1:]))]
global_rank=np.empty(cohort.num_items+1,dtype=np.int32);global_rank[0]=-1;global_rank[global_order]=np.arange(cohort.num_items)
data={'daily_first':[],'daily_uniform':[],'train_uniform32':[]}
hashes={'daily_first':hashlib.sha256(),'daily_uniform':hashlib.sha256(),'train_uniform32':hashlib.sha256()}
tie_omissions=Counter();all_events=Counter();user_train=[];reference_hash_checks=0

def seed_for(uid,tag):
    raw=json.dumps(['music-data-audit-v1',42,int(uid),tag],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')

def scores(target,context):
    if target==0:return {'global_ndcg10':0.,'global_ndcg100':0.,'global_hr10':0.,'global_hr100':0.,'frequency_ndcg10':0.,'frequency_ndcg100':0.,'frequency_hr10':0.,'frequency_hr100':0.}
    recent=context[-128:]
    ids,rev,ct=np.unique(recent[::-1],return_index=True,return_counts=True)
    known=ids>0;ids,rev,ct=ids[known],rev[known],ct[known]
    ordered=ids[np.lexsort((ids,rev,-ct))]
    where=np.flatnonzero(ordered==target)
    if len(where):frank=int(where[0])
    else:
        gr=int(global_rank[target]);frank=len(ids)+gr-int(np.count_nonzero(global_rank[ids]<gr))
    result={}
    for model,rank in [('global',int(global_rank[target])),('frequency',frank)]:
        for k in (10,100):
            result[f'{model}_ndcg{k}']=1/math.log2(rank+2) if rank<k else 0.
            result[f'{model}_hr{k}']=float(rank<k)
    return result

def observe(kind,uid,items,times,target_pos,day=None):
    target=int(items[target_pos]);target_time=int(times[target_pos])
    end=int(np.searchsorted(times,target_time,side='left'))
    tie_omissions[kind]+=target_pos-end
    start=max(0,end-1152);context=items[start:end];cts=times[start:end]
    assert len(context)==1152 and int(cts[-1])<target_time
    recent=context[-128:];old=context[:-128]
    if target==0:group='oov'
    elif bool(np.any(recent==target)):group='seen_recent128'
    elif bool(np.any(old==target)):group='seen_only_old1024'
    else:group='not_seen_visible1152'
    occurrences=np.flatnonzero(items[:end]==target) if target else np.empty(0,dtype=np.int64)
    last=int(occurrences[-1]) if len(occurrences) else None
    row={'user_id':uid,'day':day,'group':group,'known':target>0,'seen_any_past':last is not None,
         'immediate_repeat':bool(target>0 and int(context[-1])==target),
         'recent_duration_hours':(int(cts[-1])-int(cts[-128]))/3600,
         'visible_duration_days':(int(cts[-1])-int(cts[0]))/86400,
         'gap_since_last_event_hours':(target_time-int(cts[-1]))/3600,
         'session_start_30min':(target_time-int(cts[-1]))>=1800,
         'last_occurrence_gap_events':end-last if last is not None else None,
         'last_occurrence_gap_hours':(target_time-int(times[last]))/3600 if last is not None else None,
         'recent_unique_fraction':len(np.unique(recent[recent>0]))/128}
    if kind!='train_uniform32':row.update(scores(target,context))
    data[kind].append(row)
    hashes[kind].update(np.asarray([uid,day if day is not None else -1,target_pos,target_time,target],dtype='<i8').tobytes())

for uidx,user in enumerate(cohort.users):
    uid=user.user_id;fi,ft=future[uid]
    items=np.concatenate((user.train,fi));times=np.concatenate((user.timestamps,ft))
    train_n=len(user.train)
    unique,ct=np.unique(user.train,return_counts=True)
    user_train.append({'events':train_n,'unique':len(unique),'cumulative_repeat_fraction':1-len(unique)/train_n,'top_item_share':float(ct.max()/train_n),'duration_days':(int(user.timestamps[-1])-int(user.timestamps[0]))/86400})
    eligible=np.arange(1152,train_n,dtype=np.int32)
    rng=np.random.default_rng(seed_for(uid,'train32'))
    positions=np.sort(rng.choice(eligible,size=min(32,len(eligible)),replace=False))
    for p in positions:
        # Ties may leave fewer1152 strictly prior events near earliest eligible index; skip deterministically.
        if np.searchsorted(times,int(times[p]),side='left')>=1152:observe('train_uniform32',uid,items,times,int(p))
    days=ft.astype(np.int64)//86400
    unique_days,day_start,day_counts=np.unique(days,return_index=True,return_counts=True)
    for day,offset,size in zip(unique_days,day_start,day_counts):
        assert low//86400<=int(day)<high//86400
        reference=first_by_key[(uid,int(day))]
        assert int(offset)==reference['future_index'] and int(fi[offset])==reference['target']
        p=train_n+int(offset)
        # Verify the original full16K reference context against its existing canonical hash.
        assert daily_context_sha256(uid,items[max(0,p-16384):p],times[max(0,p-16384):p])==reference['reference_context_hash']
        reference_hash_checks+=1
        observe('daily_first',uid,items,times,p,int(day))
        rng=np.random.default_rng(seed_for(uid,int(day)))
        random_p=train_n+int(offset)+int(rng.integers(int(size)))
        observe('daily_uniform',uid,items,times,random_p,int(day))
    if len(ft):
        gaps=np.diff(np.concatenate((user.timestamps[-1:],ft)).astype(np.int64))
        all_events['validation_events']+=len(ft)
        all_events['gap_30min']+=int(np.count_nonzero(gaps>=1800))
        all_events['gap_3hours']+=int(np.count_nonzero(gaps>=10800))
        all_events['tied_timestamp_events']+=int(np.count_nonzero(gaps==0))
        all_events['immediate_known_repeats']+=int(np.count_nonzero((np.concatenate((user.train[-1:],fi))[:-1]==fi)&(fi>0)))
    if (uidx+1)%1000==0:print(json.dumps({'phase':'users','users':uidx+1,'seconds':time.monotonic()-started}),flush=True)

def dist(values):
    a=np.asarray([v for v in values if v is not None],dtype=np.float64)
    return {'count':len(a),'mean':float(a.mean()),'p10':float(np.quantile(a,.10)),'p25':float(np.quantile(a,.25)),'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),'p90':float(np.quantile(a,.90)),'p99':float(np.quantile(a,.99))} if len(a) else {'count':0}

def aggregate(rows):
    groups=Counter(r['group'] for r in rows);byuser=defaultdict(list)
    for r in rows:byuser[r['user_id']].append(r)
    bools={'seen_recent128':lambda r:r['group']=='seen_recent128','seen_visible1152':lambda r:r['group'] in ('seen_recent128','seen_only_old1024'),'seen_any_past':lambda r:r['seen_any_past'],'immediate_repeat':lambda r:r['immediate_repeat'],'oov':lambda r:not r['known'],'session_start_30min':lambda r:r['session_start_30min']}
    result={'anchors':len(rows),'users':len(byuser),'anchor_group_counts':dict(groups),'anchor_group_fractions':{g:groups[g]/len(rows) for g in ['seen_recent128','seen_only_old1024','not_seen_visible1152','oov']},
        'anchor_fractions':{k:sum(fn(r) for r in rows)/len(rows) for k,fn in bools.items()},
        'macro_user_fractions':{k:float(np.mean([np.mean([fn(r) for r in rr]) for rr in byuser.values()])) for k,fn in bools.items()},
        'distributions':{k:dist([r[k] for r in rows]) for k in ['recent_duration_hours','visible_duration_days','gap_since_last_event_hours','last_occurrence_gap_events','last_occurrence_gap_hours','recent_unique_fraction']},
        'per_user_recent_repeat_fraction_distribution':dist([np.mean([r['group']=='seen_recent128' for r in rr]) for rr in byuser.values()])}
    if 'global_ndcg10' in rows[0]:result['macro_user_metrics']={k:float(np.mean([np.mean([r[k] for r in rr]) for rr in byuser.values()])) for k in rows[0] if k.startswith(('global_','frequency_'))}
    return result

assert len(data['daily_first'])==len(data['daily_uniform'])==26335
metrics={k:aggregate(v) for k,v in data.items()}
paired={}
by_selection={}
for selection in ['daily_first','daily_uniform']:
    per_user=defaultdict(list)
    for row in data[selection]:per_user[row['user_id']].append(row)
    by_selection[selection]=per_user
uids=sorted(by_selection['daily_first'])
assert uids==sorted(by_selection['daily_uniform'])
for label,fn in [('recent_repeat_fraction',lambda r:float(r['group']=='seen_recent128')),
                 ('any_past_repeat_fraction',lambda r:float(r['seen_any_past'])),
                 ('frequency_ndcg10',lambda r:r['frequency_ndcg10']),
                 ('global_ndcg10',lambda r:r['global_ndcg10'])]:
    a=np.asarray([np.mean([fn(r) for r in by_selection['daily_first'][u]]) for u in uids])
    b=np.asarray([np.mean([fn(r) for r in by_selection['daily_uniform'][u]]) for u in uids])
    delta=b-a;rng=np.random.default_rng(42);draws=[]
    for _ in range(100):draws.extend(delta[rng.integers(len(uids),size=(20,len(uids)))].mean(axis=1))
    paired[label]={'uniform_minus_first':float(delta.mean()),'ci_lower':float(np.quantile(draws,.025)),
                   'ci_upper':float(np.quantile(draws,.975)),'draws':2000,'seed':42,'matched_users':len(uids),
                   'scope':'paired bootstrap over fixed users and fixed selected anchors; not random-event-seed variation'}
previous=json.loads(Path(_archive_workspace_path('qssm2-repeat-heuristics.json')).read_text())
assert abs(metrics['daily_first']['macro_user_metrics']['frequency_ndcg10']-previous['global_metrics']['FrequencyR128']['ndcg10'])<1e-12
assert abs(metrics['daily_first']['macro_user_metrics']['frequency_ndcg100']-previous['global_metrics']['FrequencyR128']['ndcg100'])<1e-12
result={'complete':True,'recipe':recipe,'recipe_sha256':hashlib.sha256(recipe_bytes).hexdigest(),'seconds':time.monotonic()-started,
    'cohort':{'users':len(cohort.users),'train_events':int(counts.sum()),'catalogue_items':cohort.num_items,'all_train_event_counts_sha256':hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest()},
    'registered_target_schedule_hashes':{k:h.hexdigest() for k,h in hashes.items()},'original_reference_hash_checks':reference_hash_checks,
    'same_timestamp_predecessors_omitted':dict(tie_omissions),'validation_all_event_descriptives':dict(all_events),
    'training_user_distributions':{k:dist([r[k] for r in user_train]) for k in user_train[0]},'selections':metrics,
    'paired_user_bootstrap_uniform_minus_first':paired,
    'previous_neural_and_repeat_metrics_original_daily_first_only':previous['global_metrics'],
    'limitations':['Only very active long-history users; cannot infer prevalence among all Yambda/service users.','Random-event-per-user-day keeps same user/day support but is not event-volume weighted population estimation.','Earlier validation observations enter later contexts; no validation/test events enter popularity prior.','Test outcomes/metrics are not evaluated; the storage file also contains later test arrays, discarded before targets or statistics.','Daily bins use supplied anonymous timestamps, not known local human morning/session labels.','Session starts are gap>=30min proxies, not observed playlist/source sessions.','Only item IDs and timestamps available here; no artist/album/playlist/mood inference is made.','Cumulative repeat counts and item reappearance do not establish enjoyment or causal recommendation utility.','Alternative anchors are diagnostic and do not replace original neural evaluation; neural predictions unavailable for alternative anchors.','Same repeatedly studied validation; no unbiased final-test claim.']}
Path(_archive_workspace_path('music-data-audit.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'complete':True,'seconds':result['seconds'],'selections':metrics,'validation_all_event_descriptives':dict(all_events)},indent=2),flush=True)
