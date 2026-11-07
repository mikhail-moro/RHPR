"""Replay a completed batch pilot without GPU or importing the training code."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np


def input_hashes(root):
    rows=json.loads((root/'singleton-input/sequences.json').read_text())
    catalog=json.loads((root/'singleton-input/catalog-stage-01.json').read_text())
    mapping={r['item_id']:i for i,r in enumerate(catalog)};out={}
    for width in [20,50]:
        histories=[];users=[];targets=[]
        for row in rows:
            train=row['train'];times=sorted({v[1]for v in train})
            for t in times[1:]:
                h=[mapping[i]for i,tt in train if tt<t][-width:]
                for i,tt in train:
                    if tt==t:
                        histories.append([-1]*(width-len(h))+h);users.append(row['user_id']);targets.append(mapping[i])
        assert len(targets)==80268
        selected=np.random.default_rng(42).choice(len(targets),1024,replace=False)
        arrays=[np.asarray(v,dtype=np.int64)[selected]for v in [histories,users,targets]]
        for size in [256,512,1024]:out[(width,size)]=hashlib.sha256(b''.join(v[:size].tobytes()for v in arrays)).hexdigest()
    return out


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=True);load=lambda p:json.loads(p.read_text())
    data=load(root/'batch-results.json');manifest=load(root/'run-manifest.json');metrics=load(root/'metrics.json');cfg=load(root/'experiment.json')
    assert manifest['git_sha']=='b777b516014355e967db6ffcb869c307d5717e0f'
    assert manifest['config']==cfg and len(data)==metrics['attempted']==16
    expected=[(backend,width,b,p)for width in [20,50]for b,p in [(256,128),(256,256),(512,512),(1024,1024)]for backend in ['legacy_eager','sdpa']]
    assert [(r['backend'],r['history_items'],r['batch_size'],r['physical_batch_size'])for r in data]==expected
    hashes=input_hashes(root);rows=[];oom=[]
    for r in data:
        assert r==load(root/(r['id']+'.json'))
        assert r['device']=='Tesla T4' and r['torch']=='2.11.0+cu130'
        assert r['transformers']==('4.47.0'if r['backend']=='legacy_eager'else'5.22.1')
        if r['status']=='oom':
            assert r['exception_type']=='OutOfMemoryError';oom.append(r);continue
        assert r['status']=='complete' and r['amp_skipped_updates']==0
        assert r['attention_backends']==['sdpa'if r['backend']=='sdpa'else'eager']
        assert r['dropout']==.1 and r['learning_rate']==.001 and r['train_precision']=='fp16'
        assert len(r['curve'])==50 and len(r['end_to_end_curve'])==10 and r['warmup_steps']==10
        assert r['optimizer_updates']==70 and r['examples_consumed']==70*r['batch_size']
        assert r['input_sha256']==hashes[(r['history_items'],r['batch_size'])]
        for curve in [r['curve'],r['end_to_end_curve']]:
            assert all(math.isfinite(x['loss'])and x['loss']>0 and math.isfinite(x['seconds'])and x['seconds']>0 for x in curve)
        mean=statistics.mean(x['seconds']for x in r['curve']);e2e=statistics.mean(x['seconds']for x in r['end_to_end_curve'])
        assert abs(mean-r['mean_step_seconds'])<1e-12
        assert abs(e2e-r['end_to_end_mean_seconds'])<1e-12
        assert abs(r['batch_size']/mean-r['examples_per_second'])<1e-8
        assert abs(r['batch_size']/e2e-r['end_to_end_examples_per_second'])<1e-8
        row={k:r[k]for k in ['id','backend','history_items','batch_size','physical_batch_size','examples_per_second','end_to_end_examples_per_second','input_sha256']}
        row.update(step_ms=mean*1000,within_run_step_sd_ms=statistics.stdev(x['seconds']for x in r['curve'])*1000,allocated_GiB=r['peak_allocated_bytes']/2**30,reserved_GiB=r['peak_reserved_bytes']/2**30)
        rows.append(row)
    assert len(rows)==metrics['complete']==15 and len(oom)==metrics['oom']==1 and metrics['numerical_failed']==0
    assert (oom[0]['backend'],oom[0]['history_items'],oom[0]['batch_size'])==('legacy_eager',50,1024)
    comparisons=[]
    for w in [20,50]:
        ref=next(r for r in rows if r['backend']=='legacy_eager'and r['history_items']==w and r['physical_batch_size']==128)
        for r in rows:
            if r['history_items']==w:
                r['throughput_vs_old_reference']=r['examples_per_second']/ref['examples_per_second']
                r['e2e_throughput_vs_old_reference']=r['end_to_end_examples_per_second']/ref['end_to_end_examples_per_second']
        a=next(r for r in rows if r['backend']=='sdpa'and r['history_items']==w and r['batch_size']==512)
        b=next(r for r in rows if r['backend']=='sdpa'and r['history_items']==w and r['batch_size']==1024)
        comparisons.append(dict(history_items=w,sdpa_1024_vs_512_throughput_gain_pct=100*(b['examples_per_second']/a['examples_per_second']-1),sdpa_512_fraction_of_1024_throughput=a['examples_per_second']/b['examples_per_second']))
    report=dict(source_code_sha=manifest['git_sha'],duration_seconds=manifest['duration_seconds'],child_section_seconds=metrics['wall_seconds'],rows=rows,oom=oom,comparisons=comparisons,
        verification=dict(timing_samples=750,end_to_end_samples=150,independent_train_case_reconstruction=True,training_cases=80268,all_six_sample_hashes_verified=True),
        limitations=['One sweep on one T4, fresh process per attempt; no independent hardware repetitions.',
            'Fixed real minibatch reused; no full epoch or downstream NDCG evaluation.',
            '512/1024 change updates per example and may change convergence; LR not tuned.',
            'Allocated CUDA memory excludes some non-PyTorch/context allocations.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    inventory=[dict(path=p.relative_to(root).as_posix(),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(root.rglob('*'))if p.is_file()]
    (output/'download-manifest.json').write_text(json.dumps(inventory,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7));labels=['256/128','256/256','512/512','1024/1024']
    for col,w in enumerate([20,50]):
        for backend,color,label in [('legacy_eager','#89929d','Eager 4.47'),('sdpa','#287ca7','SDPA 5.22.1')]:
            values=[next((r for r in rows if r['backend']==backend and r['history_items']==w and r['batch_size']==b and r['physical_batch_size']==p),None)for b,p in [(256,128),(256,256),(512,512),(1024,1024)]]
            for ax,key in [(axes[0,col],'examples_per_second'),(axes[1,col],'allocated_GiB')]:
                ax.plot(range(4),[r[key]if r else np.nan for r in values],marker='o',color=color,label=label)
                ax.set_xticks(range(4),labels);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
        axes[0,col].set_title(f'{w} history items');axes[0,col].legend(frameon=False)
        axes[1,col].set_xlabel('Effective / physical batch')
    axes[0,0].set_ylabel('Training examples / second');axes[1,0].set_ylabel('Peak allocated CUDA memory (GiB)')
    axes[1,1].annotate('Eager 1024: OOM',xy=(2,9.1),xytext=(1.6,12.7),arrowprops=dict(arrowstyle='->'),fontsize=9)
    fig.suptitle('TIGER T4 batch pilot: throughput and memory');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(output/'t5-batch-pilot.png',dpi=180);fig.savefig(output/'t5-batch-pilot.pdf');plt.close(fig)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=analyze(a.root,a.output);print(json.dumps(dict(verification=r['verification'],comparisons=r['comparisons']),indent=2))
