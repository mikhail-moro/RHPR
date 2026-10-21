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
started=time.monotonic();recipe_path=Path(_archive_workspace_path('qssm2-repeat-heuristic-recipes.json'))
recipe_bytes=recipe_path.read_bytes();registration=json.loads(recipe_bytes);expected_recipe_sha='5ed810de5aeeb15b5181d383c28d191589fb1476477e103f4042321bf7a7efb9'
assert hashlib.sha256(recipe_bytes).hexdigest()==expected_recipe_sha
cohort=load_prepared_cohort(_archive_workspace_path('plateau-cohort'));evaluation=load_prepared_daily_evaluation(_archive_workspace_path('baseline-evaluation'),cohort)
print(json.dumps({'phase':'verified_data_loaded','seconds':time.monotonic()-started}),flush=True)
# Counts come exclusively from observed pre-cutoff training events of the fixedcohort.
all_train=np.concatenate([user.train for user in cohort.users]);assert bool((all_train>0).all())
counts=np.bincount(all_train,minlength=cohort.num_items+1);assert len(counts)==cohort.num_items+1;del all_train
catalogue_ids=np.arange(1,cohort.num_items+1,dtype=np.int32)
global_order=catalogue_ids[np.lexsort((catalogue_ids,-counts[1:]))]
global_rank=np.empty(cohort.num_items+1,dtype=np.int32);global_rank[0]=-1;global_rank[global_order]=np.arange(cohort.num_items,dtype=np.int32)
Q=Path(_archive_workspace_path('ssm2-state128-final-output'));O=Path(_archive_workspace_path('rotating-final-output'))
def read_gzip(path):
 with gzip.open(path,'rt') as stream:return json.load(stream)
fuxi=read_gzip(O/'fuxi_linear-rotating-p60-validation-anchors.json.gz');fuxi_index={r['anchor_id']:r for r in fuxi}
contexts={r['anchor_id']:r for r in read_gzip(Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-input-contexts.json.gz')}
model_records={'FuXi':fuxi,'QSSM_state16':read_gzip(O/'query_ssm_m32-rotating-p60-validation-anchors.json.gz'),'QSSM2_state128':read_gzip(Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-anchors.json.gz')}
recipe_names=[r['name'] for r in registration['recipes']];predictions={name:[] for name in recipe_names};labels={};checks=0;all_counts=Counter(r['user_id'] for r in fuxi);users=len(all_counts)
rank_digests={name:hashlib.sha256() for name in recipe_names}
def priority(sequence,mode):
 items,first_reversed,frequencies=np.unique(sequence[::-1],return_index=True,return_counts=True)
 valid=items>0;items,frequencies,first_reversed=items[valid],frequencies[valid],first_reversed[valid]
 last=len(sequence)-1-first_reversed
 order=np.lexsort((items,-last,-frequencies)) if mode=='frequency' else np.lexsort((items,-last))
 priority_ids=items[order];priority_ranks=np.empty(len(items),dtype=np.int32);priority_ranks[order]=np.arange(len(items),dtype=np.int32)
 return items,priority_ids,priority_ranks,np.sort(global_rank[items])
def rank_query(target,prepared):
 if target==0:return None
 items,order,priority_ranks,seen_global_ranks=prepared
 position=int(np.searchsorted(items,target))
 if position<len(items) and items[position]==target:return int(priority_ranks[position])
 g=int(global_rank[target]);removed=int(np.searchsorted(seen_global_ranks,g,side='left'))
 return len(items)+g-removed
for example in iter_daily_evaluation_examples(cohort,evaluation,split_name='validation',max_seq_len=1152):
 assert example.anchor_id not in labels and len(example.context)==1152 and int(example.timestamps[-1])<example.target_timestamp
 reference=fuxi_index[example.anchor_id]
 assert reference['user_id']==example.user_id and reference['day']==example.day and reference['reference_context_hash']==example.reference_context_hash and reference['target_known']==(example.target>0)
 assert daily_context_sha256(example.user_id,example.context,example.timestamps)==contexts[example.anchor_id]['actual_context_sha256']
 # These rankings are constructed without target; the target is used only by rank_query below.
 prepared={'FrequencyR128':priority(example.context[-128:],'frequency'),'FrequencyVisible1152':priority(example.context,'frequency'),'MostRecentUniqueR128':priority(example.context[-128:],'recent')}
 for name in recipe_names:
  p=prepared[name];rank_digests[name].update(example.anchor_id.encode());rank_digests[name].update(np.asarray(p[1],dtype='<i4').tobytes())
  rank=rank_query(example.target,p)
  if checks<10 and example.target>0:
   literal=np.concatenate((p[1],global_order[~np.isin(global_order,p[0],assume_unique=True)]))
   assert len(literal)==cohort.num_items and len(np.unique(literal))==cohort.num_items
   expected=int(np.flatnonzero(literal==example.target)[0]);assert rank==expected
  record={k:reference[k] for k in ('user_id','day','anchor_id','reference_context_hash','target_known')};record['rank_cutoff']=100
  record['rank']=record['rank_at_max_k']=rank+1 if rank is not None and rank<100 else 0
  for k in (10,100):
   hit=rank is not None and rank<k;record[f'hr{k}']=float(hit);record[f'ndcg{k}']=1/math.log2(rank+2) if hit else 0.
  predictions[name].append(record)
 if checks<10 and example.target>0:checks+=1
 if example.target==0:group='oov_target'
 elif bool(np.any(example.context[-128:]==example.target)):group='seen_in_recent128'
 elif bool(np.any(example.context[:-128]==example.target)):group='seen_only_in_old1024'
 else:group='not_seen_in_visible1152'
 labels[example.anchor_id]=group
assert len(labels)==26335 and checks==10 and set(labels)==set(fuxi_index)
combined={**model_records,**predictions}
def aggregate(records):
 m,_=_aggregate(records)
 return {k:m[k] for k in ('user_count','anchor_count','oov_target_count','ndcg10','ndcg100','hr10','hr100')}
result={'complete':False,'no_gpu_runs':True,'no_model_weights_or_neural_forward':True,'registered_recipes':registration,'recipe_registration_sha256':expected_recipe_sha,'train_global_prior':{'events':int(counts.sum()),'users':len(cohort.users),'catalogue_items':cohort.num_items,'items_seen_in_cohort_train':int(np.count_nonzero(counts[1:])),'global_counts_sha256':hashlib.sha256(np.asarray(counts,dtype='<i8').tobytes()).hexdigest(),'global_order_sha256':hashlib.sha256(np.asarray(global_order,dtype='<i4').tobytes()).hexdigest(),'future_events_used':False,'same_supervised_exposure_as_neural_models':False},'exact_full_catalogue_rank':{'formula':'if target is priority item: priority_rank0; else |S| + global_rank0(target) - number of priority items with smaller global_rank0','all_catalogue_items_are_candidates':True,'seen_context_items_are_not_excluded':True,'pad_id_zero_is_not_candidate':True,'literal_full_catalogue_rank_spotcheck_known_anchors':checks,'target_independent_priority_order_hashes':{name:d.hexdigest() for name,d in rank_digests.items()}},'support':{'validation_anchors':len(labels),'validation_users':users,'actual_input_context_hashes_match_saved_neural_inputs':True,'all_contexts_strictly_before_target':True,'evaluation_content_sha256':evaluation.summary['evaluation_content_sha256']},'global_metrics':{name:aggregate(records) for name,records in combined.items()},'paired_heuristic_minus_FuXi':{name:{str(k):paired_daily_ndcg_bootstrap(fuxi,records,k=k,seed=42,draws=5000,confidence=.95) for k in (10,100)} for name,records in predictions.items()},'repeat_groups':{},'limitations':['The three recipes and tie rules were fixed before evaluation; no HPO or recipe selection was performed.','Global popularity prior uses all89m pre-cutoff cohort training events, unlike34m supervised target positions of the trained neural runs; this is not a matched-capacity or matched-exposure comparison.','Heuristics use future evaluation events only when those events are already observed in the current past-only context, exactly as the neural inputs do.','Repeat groups depend on the target after prediction and are diagnostic only; not valid serving/routing features.','Same repeatedly inspected validation anchors, single neural training seed; bootstrap measures user sampling of frozen predictions only.','This is a repeat/popularity diagnostic, not evidence that history order or time is useless.']}
for group in ('seen_in_recent128','seen_only_in_old1024','not_seen_in_visible1152','oov_target'):
 result['repeat_groups'][group]={name:aggregate([r for r in rows if labels[r['anchor_id']]==group]) for name,rows in combined.items()}
result.update(complete=True,seconds=time.monotonic()-started)
Path(_archive_workspace_path('qssm2-repeat-heuristics.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'complete':True,'seconds':result['seconds'],'global_metrics':result['global_metrics'],'paired_heuristic_minus_FuXi':result['paired_heuristic_minus_FuXi'],'repeat_groups':result['repeat_groups']},indent=2))
