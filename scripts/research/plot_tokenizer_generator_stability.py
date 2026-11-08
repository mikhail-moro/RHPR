"""Plot matched SID/generator stability analysis; run the stats CLI first."""
import argparse
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
p=args.root;o=args.output
if not (o/'statistics.json').is_file():raise FileNotFoundError('Run analyze_tokenizer_generator_stability.py before plotting')
c=json.loads((p/'experiment.json').read_text());s=json.loads((p/'suite-results.json').read_text());r=json.loads((o/'statistics.json').read_text())
methods=['meta_kmeans','meta_rqvae','multimodal_rqvae'];labels=['Meta + KMeans','Meta + RQ-VAE','Multi + RQ-VAE'];colors=['#4c78a8','#f58518','#54a24b'];gs=[31415,27182,161803];ts=[42,101];marks=['o','s','^']
fig,ax=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
for a,name in enumerate(['fixed30','selected']):
 for k,m in enumerate(methods):
  for j,t in enumerate(ts):
   x=2*k+j;v=r['method_summaries'][m][name]['scores'][str(t)]
   for n,(g,y) in enumerate(zip(gs,v)):
    ax[0,a].scatter(x+(-.14+n*.14),y,c=colors[k],marker=marks[n],s=45,zorder=3)
   ax[0,a].plot([x-.23,x+.23],[np.mean(v)]*2,color='black',lw=1.2)
 ax[0,a].set_xticks(range(6),[f'{n}\nindex {t}' for n in labels for t in ts],fontsize=8)
 ax[0,a].set_title('Fixed epoch 30' if name=='fixed30' else 'Best VAL checkpoint (mixed epochs)')
 ax[0,a].set_ylabel('VAL NDCG@10');ax[0,a].set_ylim(.045,.074);ax[0,a].grid(axis='y',alpha=.2)
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker=m,color='gray',ls='',label=f'gen {g}',markersize=6) for m,g in zip(marks,gs)]
handles+=[Line2D([0],[0],color='black',label='mean over 3 seeds')]
ax[0,1].legend(handles=handles,loc='lower right',fontsize=8)
for k,m in enumerate(methods):
 curves=[]
 for t in ts:
  for g in gs:
   trial=p/'trials'/f'narrow_{m}_seed{t}_gen{g}'/'training-curves.json';curves.append(json.loads(trial.read_text()))
 for j,key in enumerate(['ndcg@10','total_ce']):
  xs=[];ys=[];ns=[]
  for epoch in [10,20,30,40,50,60]:
   entries=[x for c1 in curves for x in c1 if x['epoch']==epoch and 'validation' in x]
   vals=[x['validation']['ndcg@10'] if j==0 else x['validation_loss']['total_ce'] for x in entries]
   xs.append(epoch);ys.append(np.mean(vals));ns.append(len(vals))
  ax[1,j].plot(xs,ys,'-o',color=colors[k],label=labels[k],lw=1.8,ms=4)
  if ns[-1]!=6:
   ax[1,j].annotate('n=5',(xs[-1],ys[-1]),xytext=(-25,-13 if j==0 else 8),textcoords='offset points',fontsize=8,color=colors[k])
  assert ns[:5]==[6]*5
 ax[1,0].set(xlabel='Epoch',ylabel='Mean VAL NDCG@10',title='Mean prediction quality')
 ax[1,1].set(xlabel='Epoch',ylabel='Mean teacher-forced VAL CE (sum)',title='Mean validation loss')
for a in ax[1]:a.grid(alpha=.2);a.set_xticks([10,20,30,40,50,60]);a.legend(fontsize=8)
fig.suptitle('ML1M: fixed SID × generator seed — validation only',fontsize=15)
fig.text(.5,-.015,'3 generator seeds × 2 tokenizer indexes per method; at epoch 60 text RQ-VAE has n=5 (early stop), all other points n=6. Lines do not establish statistical certainty.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(o/f'stability-summary.{ext}',dpi=170,bbox_inches='tight')
print('wrote',o/'stability-summary.png')
