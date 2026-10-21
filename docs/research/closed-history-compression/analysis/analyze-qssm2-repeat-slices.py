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

from collections import Counter
import gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,_archive_repo_src_path())
from rhpr.sasrec.daily_evaluation_data import daily_context_sha256,iter_daily_evaluation_examples,load_prepared_daily_evaluation
from rhpr.sasrec.daily_quality import _aggregate,paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
started=time.monotonic()
cohort=load_prepared_cohort(_archive_workspace_path('plateau-cohort'))
evaluation=load_prepared_daily_evaluation(_archive_workspace_path('baseline-evaluation'),cohort)
print(json.dumps({'phase':'prepared_data_loaded','seconds':time.monotonic()-started}),flush=True)
Q=Path(_archive_workspace_path('ssm2-state128-final-output'));O=Path(_archive_workspace_path('rotating-final-output'))
paths={'FuXi':O/'fuxi_linear-rotating-p60-validation-anchors.json.gz','QSSM_state16':O/'query_ssm_m32-rotating-p60-validation-anchors.json.gz','QSSM2_state128':Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-anchors.json.gz'}
ranks={}
for name,path in paths.items():
 with gzip.open(path,'rt') as f:ranks[name]=json.load(f)
with gzip.open(Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-input-contexts.json.gz','rt') as f:contexts={v['anchor_id']:v for v in json.load(f)}
indexed={name:{r['anchor_id']:r for r in rows} for name,rows in ranks.items()}
assert all(len(rows)==len(indexed[name])==26335 for name,rows in ranks.items())
assert set(contexts)==set(indexed['FuXi'])
labels={};counter=Counter();fingerprint=hashlib.sha256();lengths=[]
for example in iter_daily_evaluation_examples(cohort,evaluation,split_name='validation',max_seq_len=1152):
 assert example.anchor_id not in labels
 assert len(example.context)==1152 and len(example.timestamps)==1152
 assert int(example.timestamps[-1])<example.target_timestamp
 ref=contexts[example.anchor_id]
 assert daily_context_sha256(example.user_id,example.context,example.timestamps)==ref['actual_context_sha256']
 for name,rows in indexed.items():
  row=rows[example.anchor_id]
  assert row['user_id']==example.user_id and row['day']==example.day and row['reference_context_hash']==example.reference_context_hash and row['target_known']==(example.target>0)
 if example.target==0:group='oov_target'
 elif bool(np.any(example.context[-128:]==example.target)):group='seen_in_recent128'
 elif bool(np.any(example.context[:-128]==example.target)):group='seen_only_in_old1024'
 else:group='not_seen_in_visible1152'
 labels[example.anchor_id]=group;counter[group]+=1;lengths.append(len(example.context))
 fingerprint.update(json.dumps([example.anchor_id,group],separators=(',',':')).encode())
assert len(labels)==26335 and set(labels)==set(indexed['FuXi']) and counter['oov_target']==1400
users={r['user_id'] for r in ranks['FuXi']};all_counts=Counter(r['user_id'] for r in ranks['FuXi'])
result={'complete':False,'no_gpu_runs':True,'no_model_inference_or_weights':True,'no_training_or_routing_change':True,'diagnostic_is_posthoc_and_label_conditioned':True,'no_test_model_evaluation':True,'group_definitions':{'seen_in_recent128':'known target occurs in last128 occupied observed events, whether or not it also occurs in H','seen_only_in_old1024':'known target absent from R128, present in the previous1024 observed H events','not_seen_in_visible1152':'known target absent from the entire model-visible1152-event context; it may occur outside that window','oov_target':'target unknown to the train catalogue; zero credit retained'},'denominators':{'within_group_ndcg':'macro mean of each represented user\'s mean NDCG over only anchors in this group; users can appear in several groups','global_macro_ndcg_contribution':'sum of each group anchor\'s NDCG divided by that user\'s total validation anchors, then divided by all4384validationusers; group contributions sum exactly to global macro NDCG'},'support':{'anchors':len(labels),'users':len(users),'context_length_min':min(lengths),'context_length_max':max(lengths),'all_context_hashes_match_evaluated_model_inputs':True,'all_occupied_context_timestamps_precede_target':True,'cohort_uid_length_sha256':cohort.summary['cohort_uid_length_sha256'],'cohort_mapped_train_items_sha256':cohort.summary['cohort_mapped_train_items_sha256'],'evaluation_content_sha256':evaluation.summary['evaluation_content_sha256'],'anchor_group_partition_sha256':fingerprint.hexdigest()},'limitations':['groups use the held-out target only after predictions, so cannot be serving/routing features','exploratory analysis of the same validation-selected checkpoints; no multiple-comparison correction','bootstrap represents user sampling of frozen models only, not seed/selection uncertainty','seen-in-H gain can support repeat/long-term-profile relevance but cannot establish order or periodicity','not-seen-within1152 does not mean a globally novel item or one absent from the user\'s full history'],'groups':{}}
for group in ('seen_in_recent128','seen_only_in_old1024','not_seen_in_visible1152','oov_target'):
 selections={name:[row for row in rows if labels[row['anchor_id']]==group] for name,rows in ranks.items()}
 support=selections['FuXi'];row={'anchors':len(support),'users':len({r['user_id'] for r in support}),'anchor_fraction':len(support)/len(labels),'models':{},'paired_FuXi_minus_compressor':{}}
 for name,rows in selections.items():
  m,_=_aggregate(rows)
  contribution={str(k):sum(r[f'ndcg{k}']/all_counts[r['user_id']] for r in rows)/len(users) for k in (10,100)}
  row['models'][name]={'ndcg10':m['ndcg10'],'ndcg100':m['ndcg100'],'hr10':m['hr10'],'hr100':m['hr100'],'global_macro_ndcg_contribution':contribution}
 if group!='oov_target':
  for name in ('QSSM_state16','QSSM2_state128'):
   row['paired_FuXi_minus_compressor'][name]={str(k):paired_daily_ndcg_bootstrap(selections[name],selections['FuXi'],k=k,seed=42,draws=5000,confidence=.95) for k in (10,100)}
 result['groups'][group]=row
for name,records in ranks.items():
 m,_=_aggregate(records)
 for k in (10,100):
  total=sum(g['models'][name]['global_macro_ndcg_contribution'][str(k)] for g in result['groups'].values())
  assert math.isclose(total,m[f'ndcg{k}'],abs_tol=1e-14)
result['global_gap_contributions']={name:{str(k):{group:g['models']['FuXi']['global_macro_ndcg_contribution'][str(k)]-g['models'][name]['global_macro_ndcg_contribution'][str(k)] for group,g in result['groups'].items()} for k in (10,100)} for name in ('QSSM_state16','QSSM2_state128')}
result.update(complete=True,seconds=time.monotonic()-started)
Path(_archive_workspace_path('qssm2-research-repeat-slices.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'complete':True,'seconds':result['seconds'],'support':result['support'],'groups':{group:{'anchors':v['anchors'],'users':v['users'],'fraction':v['anchor_fraction'],'models':v['models'],'QSSM2_paired_CI':{k:{x:y for x,y in b.items() if x in ('delta_ndcg','ci_lower','ci_upper','relative_delta_percent')} for k,b in v['paired_FuXi_minus_compressor'].get('QSSM2_state128',{}).items()}} for group,v in result['groups'].items()},'global_gap_contributions':result['global_gap_contributions']},indent=2))
