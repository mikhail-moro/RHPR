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

from pathlib import Path
import json,time,sys,gzip,hashlib,math
from collections import Counter,defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,_archive_repo_src_path())
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.daily_evaluation_data import load_prepared_daily_evaluation,iter_daily_evaluation_examples,daily_context_sha256
from rhpr.sasrec.daily_quality import _aggregate

started=time.monotonic();root=Path(_archive_workspace_path('music-origin'))
cohort=load_prepared_cohort(_archive_workspace_path('plateau-cohort'))
evaluation=load_prepared_daily_evaluation(_archive_workspace_path('baseline-evaluation'),cohort)
val_end=evaluation.summary['validation_end_timestamp']
anchors=[a for a in evaluation.anchors if cohort.summary['train_end_timestamp']<=a.target_timestamp<val_end]
by_uid={u.user_id:u for u in cohort.users};future_by_uid={u.user_id:u for u in evaluation.users}
anchor_by_uid=defaultdict(list)
for a in anchors:anchor_by_uid[a.user_id].append(a)
print(json.dumps({'phase':'verified_data_loaded','seconds':time.monotonic()-started}),flush=True)

flags_by_anchor={};found=set();totals=Counter();train_rates=[];validation_event_rates=[]
source=pq.ParquetFile(root/'listens-organic.partial.parquet')
for batch in source.iter_batches(batch_size=128,columns=['uid','is_organic'],use_threads=False):
    for i in range(batch.num_rows):
        uid=int(batch['uid'][i].as_py())
        if uid not in by_uid:continue
        assert uid not in found;found.add(uid)
        values=batch['is_organic'][i].values.to_numpy(zero_copy_only=False)
        train_n=len(by_uid[uid].train)
        n_val=int(np.searchsorted(future_by_uid[uid].future_timestamps,val_end,side='left'))
        assert len(values)>=train_n+n_val
        assert bool(np.isin(values[:train_n+n_val],[0,1]).all())
        train_flags=values[:train_n];val_flags=values[train_n:train_n+n_val]
        totals['train_events']+=train_n;totals['train_organic_events']+=int(train_flags.sum())
        totals['validation_events']+=n_val;totals['validation_organic_events']+=int(val_flags.sum())
        train_rates.append(float(train_flags.mean()))
        if n_val:validation_event_rates.append(float(val_flags.mean()))
        for a in anchor_by_uid[uid]:
            assert a.future_index<n_val
            flags_by_anchor[a.anchor_id]=int(values[train_n+a.future_index])
assert found==set(by_uid) and len(flags_by_anchor)==26335
assert totals['train_events']==cohort.train_events
assert totals['validation_events']==evaluation.summary['validation_period_events']
print(json.dumps({'phase':'source_flags_joined','seconds':time.monotonic()-started,'totals':dict(totals),'anchors':len(flags_by_anchor)}),flush=True)

def gz(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)
Q=Path(_archive_workspace_path('ssm2-state128-final-output'));O=Path(_archive_workspace_path('rotating-final-output'))
neural={'FuXi':gz(O/'fuxi_linear-rotating-p60-validation-anchors.json.gz'),
        'QSSM_state16':gz(O/'query_ssm_m32-rotating-p60-validation-anchors.json.gz'),
        'QSSM2_state128':gz(Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-anchors.json.gz')}
reference={r['anchor_id']:r for r in neural['FuXi']}
contexts={r['anchor_id']:r['actual_context_sha256'] for r in gz(Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-input-contexts.json.gz')}
prior_reference=json.loads(Path(_archive_workspace_path('qssm2-repeat-heuristics.json')).read_text())
all_train=np.concatenate([u.train for u in cohort.users])
counts=np.bincount(all_train,minlength=cohort.num_items+1);del all_train
ids=np.arange(1,cohort.num_items+1,dtype=np.int32)
order=ids[np.lexsort((ids,-counts[1:]))]
assert hashlib.sha256(np.asarray(order,dtype='<i4').tobytes()).hexdigest()==prior_reference['train_global_prior']['global_order_sha256']
global_rank=np.zeros(cohort.num_items+1,dtype=np.int32);global_rank[order]=np.arange(cohort.num_items,dtype=np.int32)
predictions={'FrequencyR128':[],'GlobalPopularity':[]};descriptive={};origin_labels={a:('organic' if v else 'recommendation_driven') for a,v in flags_by_anchor.items()}
for ex in iter_daily_evaluation_examples(cohort,evaluation,split_name='validation',max_seq_len=1152):
    assert daily_context_sha256(ex.user_id,ex.context,ex.timestamps)==contexts[ex.anchor_id]
    assert int(ex.timestamps[-1])<ex.target_timestamp
    rec=reference[ex.anchor_id];target=int(ex.target)
    recent=[int(v) for v in ex.context[-128:] if v>0]
    freq=Counter(recent);last={v:i for i,v in enumerate(recent)}
    priority=sorted(freq,key=lambda v:(-freq[v],-last[v],v))
    r_seen=target>0 and target in freq
    h_seen=bool(target>0 and np.any(ex.context[:-128]==target))
    descriptive[ex.anchor_id]={'seen_R128':int(r_seen),'seen_only_H1024':int(not r_seen and h_seen),
                             'seen_visible1152':int(r_seen or h_seen),'unseen_visible1152':int(target>0 and not r_seen and not h_seen),
                             'oov':int(target==0),'same_as_last_event':int(target>0 and target==int(ex.context[-1]))}
    if target==0:rank_freq=rank_global=None
    elif r_seen:rank_freq=priority.index(target);rank_global=int(global_rank[target])
    else:
        rank_global=int(global_rank[target])
        rank_freq=len(priority)+rank_global-sum(global_rank[v]<rank_global for v in priority)
    for name,rank in [('FrequencyR128',rank_freq),('GlobalPopularity',rank_global)]:
        row={k:rec[k] for k in ('user_id','day','anchor_id','reference_context_hash','target_known')}
        row.update(rank_cutoff=100,rank=(rank+1 if rank is not None and rank<100 else 0),rank_at_max_k=(rank+1 if rank is not None and rank<100 else 0))
        for k in (10,100):
            hit=rank is not None and rank<k
            row[f'hr{k}']=float(hit);row[f'ndcg{k}']=1/math.log2(rank+2) if hit else 0.
        predictions[name].append(row)

combined={**neural,**predictions}
def aggregate(rows):
    metrics,_=_aggregate(rows)
    return {k:metrics[k] for k in ('user_count','anchor_count','oov_target_count','ndcg10','ndcg100','hr10','hr100')}
global_metrics={name:aggregate(rows) for name,rows in combined.items()}
assert abs(global_metrics['FrequencyR128']['ndcg10']-prior_reference['global_metrics']['FrequencyR128']['ndcg10'])<1e-12
assert abs(global_metrics['FrequencyR128']['ndcg100']-prior_reference['global_metrics']['FrequencyR128']['ndcg100'])<1e-12
all_user_counts=Counter(r['user_id'] for r in neural['FuXi']);U=len(all_user_counts)
groups={}
for origin in ['organic','recommendation_driven']:
    selected=[r for r in neural['FuXi'] if origin_labels[r['anchor_id']]==origin]
    count=len(selected);per_user=defaultdict(Counter)
    descriptives=Counter()
    for r in selected:
        features=descriptive[r['anchor_id']];descriptives.update(features)
        per_user[r['user_id']].update({'anchors':1,**features})
    metrics={name:aggregate([r for r in rows if origin_labels[r['anchor_id']]==origin]) for name,rows in combined.items()}
    contributions={name:{f'ndcg{k}':sum(r[f'ndcg{k}']/all_user_counts[r['user_id']] for r in rows if origin_labels[r['anchor_id']]==origin)/U for k in (10,100)} for name,rows in combined.items()}
    groups[origin]={'anchors':count,'users':len(per_user),'anchor_fraction':count/26335,
                    'global_macro_user_weight':sum(1/all_user_counts[r['user_id']] for r in selected)/U,
                    'descriptive_counts':dict(descriptives),'descriptive_anchor_fractions':{k:v/count for k,v in descriptives.items()},
                    'descriptive_macro_user_fractions':{k:float(np.mean([v[k]/v['anchors'] for v in per_user.values()])) for k in descriptives},
                    'within_origin_macro_user_metrics':metrics,'global_macro_user_ndcg_contributions':contributions}
for name in combined:
    for k in (10,100):assert abs(sum(g['global_macro_user_ndcg_contributions'][name][f'ndcg{k}'] for g in groups.values())-global_metrics[name][f'ndcg{k}'])<1e-12
source_manifest=json.loads((root/'fetch-manifest.json').read_text())
report={'complete':True,'no_gpu':True,'no_neural_inference':True,'no_test_analysis':True,'source_manifest':source_manifest,
        'split':'original_daily_first_validation','support':{'anchors':26335,'users':U,'cohort_users_with_raw_flags_joined':len(found),
           'actual_context_hashes_match_saved_neural_inputs':True,'source_revision_matches_original_preparation':True,
           'source_offset_alignment_basis':'Exact pinned source, original preparation verified complete UID/item/timestamp prefixes; organic parallel list indexed at original train length+future index. New audit independently checks source UID presence, flag binary values and list-length bounds; it does not redownload item/time values.'},
        'origin_scope':{'organic_flag_1':'organic interaction','organic_flag_0':'recommendation-driven interaction',
                        'target_origin_is_posthoc_label_only':True,'used_as_model_or_ranking_input':False},
        'train_validation_event_counts':dict(totals),
        'train_organic_event_fraction':totals['train_organic_events']/totals['train_events'],
        'train_macro_user_organic_fraction':float(np.mean(train_rates)),
        'validation_all_event_organic_fraction':totals['validation_organic_events']/totals['validation_events'],
        'validation_all_event_macro_user_organic_fraction':float(np.mean(validation_event_rates)),
        'global_metrics':global_metrics,'origin_groups':groups,'seconds':time.monotonic()-started,
        'limitations':['Origin is the recorded event source, not randomized assignment; origin groups differ in users/items/context. Their quality differences are descriptive, not causal recommendation effects.',
                       'An organic event does not necessarily mean explicit choice of this particular track, and a recommendation event does not prove satisfaction.',
                       'Only user/item/time and origin are used here; playback completion and track duration were not fetched or analysed.',
                       'Target-origin grouping happens after prediction and must not become an input that leaks the target-generation process.',
                       'Users have exceptionally long histories; results do not estimate all listeners. Validation selects the first event per occupied day.',
                       'Within-origin macro-user denominators differ and users overlap groups. Global-denominator contributions are supplied separately.',
                       'All models are frozen, validation-selected p60 single-seed runs; no new test or neural inference was run.']}
Path(_archive_workspace_path('music-origin-audit.json')).write_text(json.dumps(report,indent=2)+'\n')
lines=['# Organic versus recommendation-driven repeats on original validation','',
       'Read-only CPU audit. Only uid/is_organic source column chunks were downloaded; no item/time/playback columns or GPU inference.',
       '',f"Training organic fraction: {100*report['train_organic_event_fraction']:.2f}% of events. Validation all-event organic fraction: {100*report['validation_all_event_organic_fraction']:.2f}%.",'',
       '| Target origin | Anchors | Share | Repeat R128 | Seen1152 | FrequencyR128 NDCG10 | FuXi NDCG10 | QSSM2 NDCG10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,g in groups.items():
    m=g['within_origin_macro_user_metrics'];d=g['descriptive_anchor_fractions']
    lines.append(f"| {name} | {g['anchors']:,} | {100*g['anchor_fraction']:.2f}% | {100*d['seen_R128']:.2f}% | {100*d['seen_visible1152']:.2f}% | {m['FrequencyR128']['ndcg10']:.6f} | {m['FuXi']['ndcg10']:.6f} | {m['QSSM2_state128']['ndcg10']:.6f} |")
lines+=['','Repeat shares are anchor fractions. Quality is macro-user within each origin group. Users overlap and within-group denominators differ; global-weight contributions in JSON reconstruct the original metrics exactly.','',
        'The original source event is joined by UID and original parallel-array offset (train length + future index), with pinned-source provenance and explicit binary/length checks. Actual model input hashes match for every anchor.','',
        'The local Parquet container is sparse: its logical size preserves original offsets, while only requested chunks/footer/header are physically stored. No full4GB source file was transferred. See the fetch manifest for exact network and allocated byte counts.','',
        'Limitations:']+[f'- {value}' for value in report['limitations']]
lines+=['','[Full JSON](/workspace/music-origin-audit.json), [fetch manifest](/workspace/music-origin/fetch-manifest.json), [analysis script](/workspace/analyze-music-origin.py).']
Path(_archive_workspace_path('music-origin-audit.md')).write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete':True,'seconds':report['seconds'],'train_organic':report['train_organic_event_fraction'],
                  'validation_events_organic':report['validation_all_event_organic_fraction'],'groups':groups},indent=2))
