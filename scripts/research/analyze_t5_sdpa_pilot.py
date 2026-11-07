"""CPU-only verification and plotting of the completed native T5 SDPA pilot."""
import argparse,hashlib,json,math,statistics,zipfile
from pathlib import Path

def analyze(root, output):
    output.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text())
    data=load(root/'pilot-results.json');manifest=load(root/'run-manifest.json');metrics=load(root/'metrics.json');cfg=load(root/'experiment.json')
    assert manifest['git_sha']=='d6edbf7815a3814867f915a1cb733a805131b2be'
    assert manifest['config']==cfg and metrics['parity_passed'] is True
    assert cfg['generator']['dropout']==.1 and cfg['pilot']['timed_steps']==50
    rows=[];comparisons=[]
    for backend,v in data.items():
        assert v==load(root/(backend+'-result.json'))
        assert v['device']=='Tesla T4' and v['torch']=='2.11.0+cu130'
        assert v['transformers']==('4.47.0'if backend=='legacy_eager'else'5.22.1')
        assert v['parity']['gradient_tensors']==95 and v['parity']['parameters']==4977920
        if backend!='legacy_eager':assert v['parity']['legacy_beam_exact_fraction']==1.
        assert [b['history_items']for b in v['benchmarks']]==[20,50]
        for b in v['benchmarks']:
            curve=b['loss_curve'];assert len(curve)==50 and b['amp_skipped_updates']==0
            assert b['dropout']==.1 and b['train_precision']=='fp16'
            assert b['batch_size']==256 and b['physical_batch_size']==128
            assert all(math.isfinite(x['loss'])and x['loss']>0 and math.isfinite(x['seconds'])and x['seconds']>0 for x in curve)
            mean=statistics.mean(x['seconds']for x in curve)
            assert abs(mean-b['mean_step_seconds'])<1e-12
            fused=any('_scaled_dot_product_efficient_attention_backward'in op['name']for op in b['sdpa_operators'])
            if backend=='sdpa':assert fused
            rows.append(dict(backend=backend,history_items=b['history_items'],step_ms=mean*1000,
                median_step_ms=b['median_step_seconds']*1000,end_to_end_ms=b['end_to_end_mean_seconds']*1000,
                peak_allocated_GiB=b['peak_allocated_bytes']/2**30,peak_reserved_GiB=b['peak_reserved_bytes']/2**30,
                within_run_step_sd_ms=statistics.stdev(x['seconds']for x in curve)*1000,
                fused_efficient_backward_observed=fused,amp_skipped_updates=0))
    for w in [20,50]:
        by={r['backend']:r for r in rows if r['history_items']==w};old=by['legacy_eager'];modern=by['eager'];sdpa=by['sdpa']
        assert abs(old['step_ms']/sdpa['step_ms']-metrics[f'sdpa_speedup_vs_legacy_eager_h{w}'])<1e-12
        comparisons.append(dict(history_items=w,time_change_vs_legacy_pct=100*(sdpa['step_ms']/old['step_ms']-1),
            time_change_vs_modern_eager_pct=100*(sdpa['step_ms']/modern['step_ms']-1),
            memory_change_vs_legacy_pct=100*(sdpa['peak_allocated_GiB']/old['peak_allocated_GiB']-1),
            end_to_end_change_vs_legacy_pct=100*(sdpa['end_to_end_ms']/old['end_to_end_ms']-1)))
    report=dict(source_code_sha=manifest['git_sha'],run_duration_seconds=manifest['duration_seconds'],
        child_section_seconds=metrics['wall_seconds'],rows=rows,comparisons=comparisons,
        parity={b:v['parity']for b,v in data.items()},
        limitations=['One sequential run on one physical T4; no randomized runtime order or independent timing repetitions.',
            '50 optimizer updates timed per configuration; repeated fixed real minibatch, not a full epoch.',
            'Numerical parity checked in FP32/eval mode with backward; FP16/dropout .1 checked for short-run stability only.',
            'Profiler totals overlap parent/child operators; do not sum inclusive operator CUDA times.',
            'No downstream quality, convergence, full-epoch or ML1M speed claims.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    inventory=[dict(path=p.relative_to(root).as_posix(),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(root.rglob('*'))if p.is_file()]
    (output/'download-manifest.json').write_text(json.dumps(inventory,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));colors=['#7a8794','#da9757','#397ea5'];labels=['Eager 4.47','Eager 5.22.1','SDPA 5.22.1']
    for i,backend in enumerate(['legacy_eager','eager','sdpa']):
        values=[next(r for r in rows if r['backend']==backend and r['history_items']==w)for w in [20,50]];x=np.arange(2)+(i-1)*.24
        for ax,key in zip(axes,['step_ms','peak_allocated_GiB']):
            bars=ax.bar(x,[r[key]for r in values],width=.22,color=colors[i],label=labels[i]);ax.bar_label(bars,fmt='%.2f'if key.endswith('GiB')else'%.1f',fontsize=8,padding=3)
    for ax in axes:
        ax.set_xticks([0,1],['20 items','50 items']);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    axes[0].set_ylabel('Full optimizer step (ms)');axes[1].set_ylabel('Peak allocated CUDA memory (GiB)');axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('TIGER on T4: native SDPA pilot',fontsize=13)
    fig.text(.5,.015,'FP16, dropout 0.1, batch 256 / microbatch 128; 50 timed steps, one sequential run.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(output/'t5-sdpa-pilot.png',dpi=180);fig.savefig(output/'t5-sdpa-pilot.pdf');plt.close(fig)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();d=analyze(a.root,a.output);print(json.dumps(d['comparisons'],indent=2))
