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

import gzip, hashlib, json, math
from pathlib import Path

base=Path(_archive_workspace_path(''))
reports={
    'FuXi': (base/'rotating-final-output/query-rotating-windows.json','fuxi_linear','rotating'),
    'QSSM_state16': (base/'rotating-final-output/query-rotating-windows.json','query_ssm_m32','rotating'),
    'QSSM2_state128': (base/'ssm2-state128-final-output/query-ssm2-training.json','query_ssm2_m32_s128','fixed_lr'),
}
out={'complete':False,'model_weights_loaded':False,'neural_forward_executed':False,'models':{}}
for name,(report_path,model_name,trial_name) in reports.items():
    report=json.loads(report_path.read_text())
    trial=report['trials'][model_name][trial_name]
    assert trial['best_epoch']==60 and trial['status']=='success'
    rank_path=report_path.parent/trial['best_validation_rank_file']
    assert rank_path.name=={
        'FuXi':'fuxi_linear-rotating-p60-validation-anchors.json.gz',
        'QSSM_state16':'query_ssm_m32-rotating-p60-validation-anchors.json.gz',
        'QSSM2_state128':'query_ssm2_m32_s128-fixed_lr-p60-validation-anchors.json.gz',
    }[name]
    with gzip.open(rank_path,'rt') as stream:rows=json.load(stream)
    assert len(rows)==26335 and len({r['anchor_id'] for r in rows})==26335
    assert all('model' not in r for r in rows)
    for r in rows:
        rank=r['rank_at_max_k']; assert 0<=rank<=100 and r['rank_cutoff']==100
        assert r['rank']==(rank if 0<rank<=10 else 0)
        if not r['target_known']:assert rank==0
        for k in (10,100):
            inside=0<rank<=k
            assert r[f'hr{k}']==float(inside)
            assert r[f'ndcg{k}']==(1/math.log2(rank+1) if inside else 0.)
    per_user={}
    for r in rows:per_user.setdefault(r['user_id'],[]).append(r)
    means={f'ndcg{k}':sum(sum(r[f'ndcg{k}'] for r in rs)/len(rs) for rs in per_user.values())/len(per_user) for k in (10,100)}
    for key,value in means.items():assert math.isclose(value,trial['best_validation'][key],abs_tol=1e-14,rel_tol=0)
    out['models'][name]={'best_epoch':60,'trial':trial_name,'rank_file':rank_path.name,
        'rank_file_sha256':hashlib.sha256(rank_path.read_bytes()).hexdigest(),
        'anchors':len(rows),'users':len(per_user),'oov_targets':sum(not r['target_known'] for r in rows),
        'every_record_rank_hr_ndcg_internally_consistent':True,'manually_aggregated_metrics_match_own_trial_best_validation':True,
        'model_name_field_present_in_rank_records':False,'macro_user_metrics':means}
out['complete']=True
(base/'qssm2-saved-ranks-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
