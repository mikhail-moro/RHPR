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

import gzip,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,_archive_repo_src_path())
from rhpr.sasrec.daily_quality import _aggregate,paired_daily_ndcg_bootstrap
Q=Path(_archive_workspace_path('ssm2-state128-final-output')); O=Path(_archive_workspace_path('rotating-final-output'))
q=json.loads((Q/'query-ssm2-training.json').read_text()); o=json.loads((O/'query-rotating-windows.json').read_text())
qr=q['trials']['query_ssm2_m32_s128']['fixed_lr']; old={name:o['trials'][name]['rotating'] for name in ('query_ssm_m32','fuxi_linear')}
qc=json.loads((Q/'ssm2-training-coverage.json').read_text()); oc=json.loads((O/'rotating-window-coverage.json').read_text())
qm=json.loads((Q/'metrics.json').read_text()); om=json.loads((O/'metrics.json').read_text())
def load_ranks(folder,result):
 with gzip.open(folder/result['best_validation_rank_file'],'rt') as f:return json.load(f)
def slim_metrics(rows):
 m,_=_aggregate(rows)
 return {k:m[k] for k in ('user_count','anchor_count','oov_target_count','ndcg10','ndcg100','warm_ndcg10','warm_ndcg100')}
def slices(rows):
 earliest={}
 for r in rows: earliest[r['user_id']]=min(earliest.get(r['user_id'],r['day']),r['day'])
 return {'all':rows,'first_anchor':[r for r in rows if r['day']==earliest[r['user_id']]],'later_anchors':[r for r in rows if r['day']!=earliest[r['user_id']]],**{f'day_{d}':[r for r in rows if r['day']==d] for d in sorted({r['day'] for r in rows})}}
models={'QSSM2_state128':qr,'QSSM_state16':old['query_ssm_m32'],'FuXi':old['fuxi_linear']}
assert q['complete'] and o['complete'] and qm['quality_complete'] and om['quality_complete']
for r in models.values():
 assert r['status']=='success' and r['passes_completed']==60 and r['best_epoch']==60 and r['successful_updates']==2100 and r['amp_retried_update_attempts']==0
 assert r['test_evaluations']==0
 assert all(math.isfinite(v) for p in r['passes'] for v in [p['loss'],p['seconds']])
base_coverage=qc['actual_completed_training']
keys=('ordered_schedule_sha256','schedule_sha256','target_positions_sha256','unique_target_positions','target_presentations','unique_endpoints','repeated_endpoint_presentations','unique_users','windows')
for name in old:
 assert all(base_coverage[k]==oc['actual_completed_training'][name][k] for k in keys)
common={k:q['config'][k]==o['config'][k] for k in ('data','model','benchmark','schedule','parent','runtime','seed')}; assert all(common.values())
assert {k:v for k,v in q['config']['training'].items() if k!='budget_seconds'}=={k:v for k,v in o['config']['training'].items() if k!='budget_seconds'}
result={'complete':False,'no_gpu_runs':True,'no_model_weights_or_forward':True,'models':{},'matched_protocol':{'common_config_fields':common,'optimization_equal_except_budget':True,'coverage':{k:base_coverage[k] for k in keys},'native_state128_audit':q['native_scan_audit'],'validation_identity':{k:qr['best_validation'][k] for k in ('anchor_count','user_count','oov_target_count','num_items')},'source_sha':{'QSSM2':q['source_code_sha'],'QSSM_and_FuXi':o['source_code_sha']}},'interpretation_limits':['single training seed','same repeatedly inspected validation set used for checkpoint selection','paired bootstrap measures user sampling only, not training seed or checkpoint-selection uncertainty','all selected checkpoints hit maximum60; validation still improves, so no asymptotic/converged ceiling claim','no test evaluation','reported daily slices are exploratory; no history-periodicity/popularity mechanism identified from ranks alone'],'paired_comparisons':{},'daily_slices':{}}
for name,r in models.items():
 curve=[{'pass':p['epoch'],'train_loss':p['loss'],'ndcg10':p['validation']['ndcg10'],'ndcg100':p['validation']['ndcg100'],'training_seconds':p['seconds'],'validation_seconds':p['validation_wall_seconds']} for p in r['passes'] if 'validation' in p]
 result['models'][name]={'best_pass':r['best_epoch'],'passes_completed':r['passes_completed'],'successful_updates':r['successful_updates'],'amp_retries':r['amp_retried_update_attempts'],'best_ndcg10':r['best_validation']['ndcg10'],'best_ndcg100':r['best_validation']['ndcg100'],'mean_training_pass_seconds':r['mean_training_pass_seconds'],'training_seconds':sum(p['seconds'] for p in r['passes']),'validation_seconds':sum(p.get('validation_wall_seconds',0) for p in r['passes']),'total_parameters':r['total_parameters'],'compressor_parameters':r['compressor_parameters'],'curve':curve,'ndcg10_strictly_rises_every_check':all(a['ndcg10']<b['ndcg10'] for a,b in zip(curve,curve[1:])),'ndcg100_strictly_rises_every_check':all(a['ndcg100']<b['ndcg100'] for a,b in zip(curve,curve[1:])),'relative_50_to_60_ndcg10_gain_percent':100*(curve[-1]['ndcg10']/curve[-2]['ndcg10']-1),'relative_50_to_60_ndcg100_gain_percent':100*(curve[-1]['ndcg100']/curve[-2]['ndcg100']-1)}
ranks={'QSSM2_state128':load_ranks(Q,qr)}
for name,record in [('QSSM_state16',old['query_ssm_m32']),('FuXi',old['fuxi_linear'])]:
 path=O/record['best_validation_rank_file']
 if path.exists():ranks[name]=load_ranks(O,record)
result['rank_files_available']=sorted(ranks)
if len(ranks)==3:
 identity=lambda rows:[(r['user_id'],r['anchor_id'],r['day'],r['reference_context_hash'],r['target_known']) for r in rows]
 expected=identity(ranks['QSSM2_state128'])
 assert all(identity(rows)==expected for rows in ranks.values())
 result['matched_protocol']['per_anchor_identity_and_catalog_support_exact']=True
 result['matched_protocol']['common_validation_anchor_support_sha256']=hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest()
 context_files={'QSSM2_state128':Q/'query_ssm2_m32_s128-fixed_lr-p60-validation-input-contexts.json.gz','QSSM_state16':O/'query_ssm_m32-rotating-p60-validation-input-contexts.json.gz','FuXi':O/'fuxi_linear-rotating-p60-validation-input-contexts.json.gz'}
 if all(p.exists() for p in context_files.values()):
  contexts={}
  for n,p in context_files.items():
   with gzip.open(p,'rt') as f:contexts[n]=json.load(f)
  assert all(rows==contexts['QSSM2_state128'] for rows in contexts.values())
  result['matched_protocol']['actual_input_context_and_query_tail_hash_rows_exact']=True
  result['matched_protocol']['common_actual_input_context_rows_sha256']=hashlib.sha256(json.dumps(contexts['QSSM2_state128'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 for reference in ('QSSM_state16','FuXi'):
  result['paired_comparisons']['QSSM2_minus_'+reference]={str(k):paired_daily_ndcg_bootstrap(ranks[reference],ranks['QSSM2_state128'],k=k,seed=42,draws=5000,confidence=.95) for k in (10,100)}
 subsets={name:slices(rows) for name,rows in ranks.items()}
 for label in subsets['QSSM2_state128']:
  result['daily_slices'][label]={name:slim_metrics(parts[label]) for name,parts in subsets.items()}
  if label in ('first_anchor','later_anchors'):
   result['daily_slices'][label]['paired_QSSM2_minus_FuXi']={str(k):paired_daily_ndcg_bootstrap(subsets['FuXi'][label],subsets['QSSM2_state128'][label],k=k,seed=42,draws=5000,confidence=.95) for k in (10,100)}
 result['complete']=True
Path(_archive_workspace_path('qssm2-research-results.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'complete':result['complete'],'available_ranks':result['rank_files_available'],'models':{n:{k:r[k] for k in ('best_ndcg10','best_ndcg100','mean_training_pass_seconds','relative_50_to_60_ndcg10_gain_percent')} for n,r in result['models'].items()},'paired_comparisons':result['paired_comparisons'],'first_later':{n:v for n,v in result['daily_slices'].items() if n in ('first_anchor','later_anchors')}},indent=2))
