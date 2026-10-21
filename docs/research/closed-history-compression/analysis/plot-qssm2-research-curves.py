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

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path(_archive_workspace_path('qssm2-research-results.json')).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.3,5.5))
fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.78,wspace=.24)
styles=[('FuXi','FuXi','#22854a','o'),('QSSM_state16','QSSM, state16','#3069b2','s'),('QSSM2_state128','QSSM2, state128','#d67a23','^')]
for ax,k in zip(axes,(10,100)):
 for key,label,color,marker in styles:
  curve=r['models'][key]['curve']; x=[p['pass'] for p in curve];y=[p[f'ndcg{k}'] for p in curve]
  ax.plot(x,y,color=color,marker=marker,linewidth=2,markersize=5,label=label)
 ax.set(xlabel='Completed training passes',ylabel=f'Macro-user NDCG@{k}',xticks=[10,20,30,40,50,60],xlim=(8,63),ylim=(0,.027 if k==10 else .044))
 ax.grid(axis='y',color='#dfe3e6',linewidth=.75);ax.set_axisbelow(True)
 sec=ax.secondary_xaxis('top',functions=(lambda x:x*35,lambda x:x/35));sec.set_xticks([350,700,1050,1400,1750,2100]);sec.set_xlabel('Optimizer updates',fontsize=9);sec.tick_params(labelsize=8)
 ax.axvline(60,linewidth=1,color='#777777',linestyle=':',alpha=.6)
axes[0].legend(loc='upper left',frameon=False,fontsize=9)
fig.suptitle('Validation continues improving at the 60-pass budget cap',y=.96,fontsize=13,fontweight='bold')
fig.text(.5,.055,'Same rotating windows, seed42, H1024 + R128; full catalogue, 4,384 users and 26,335 daily anchors.',ha='center',fontsize=9,color='#444444')
fig.text(.5,.02,'All selected checkpoints are at pass60. These curves do not establish convergence or training-seed uncertainty.',ha='center',fontsize=9,color='#444444')
fig.savefig(_archive_workspace_path('qssm2-research-curves.png'),dpi=180,facecolor='white')
fig.savefig(_archive_workspace_path('qssm2-research-curves.svg'),facecolor='white')
print('Saved /workspace/qssm2-research-curves.png and .svg')
