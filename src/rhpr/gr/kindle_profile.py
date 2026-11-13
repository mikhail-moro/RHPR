"""Bounded disposable profiling of the exact Kindle mini-TIGER checkpoint."""
from __future__ import annotations
import argparse,gc,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_write


def setup_model(p,device="cuda:0"):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.pair_unique4 import attach4,InputIndex4
    from rhpr.gr.pair_sid import preencode
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.sid import SidIndex
    c=p['config'];torch.set_num_threads(1)
    sid=np.load(Path(p['parent'])/'frozen-indexes/items-42.npy',allow_pickle=False)
    index=InputIndex4(SidIndex.from_full_sids(sid,width=256))
    model=attach4(build_grid_generator(sid,c['generator'],42,device,p['source']))
    state=torch.load(Path(p['parent'])/'trials/cosine_003/generator-best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);del state
    examples={}
    for split in ('train','validation'):
        h,u,t=cases(p['data'],split);examples[split]=(preencode(h,sid),u,t)
    def builder(h,u,t,sids,device):return index.model_batch(h,u,t,device)
    return model,index,examples,builder


def train_profile(p,compile_only=False):
    import torch
    from rhpr.gr.profiling import benchmark_training,detailed_training_profile,ProfileDeadline
    model,index,ex,builder=setup_model(p);c=p['config'];out=Path(p['output'])
    deadline=ProfileDeadline(c['profile']['max_seconds']);indices=np.random.default_rng(42).permutation(len(ex['train'][2]))[:2048]
    spec=dict(warmup=8,iterations=20,seed=42);rows=[]
    if compile_only:
        started=time.monotonic();model=torch.compile(model,mode='reduce-overhead',fullgraph=False)
        row=benchmark_training(model,ex['train'],index.sids,indices,512,'fp16','gpu_resident',c['training'],spec,deadline,'cuda:0',batch_builder=builder)
        row.update(case_wall_seconds=time.monotonic()-started,compile_mode='reduce-overhead')
        from torch._dynamo.utils import counters
        row['dynamo_counters']={str(k):{str(a):int(b)for a,b in v.items()}for k,v in counters.items()}
        _write(out/'compile-profile.json',row);return row
    for size in (256,512,1024,2048):
        for mode in ('full_pipeline','gpu_resident'):
            _event('kindle_profile_case_start',batch_size=size,mode=mode)
            begin=time.time()
            try:row=benchmark_training(model,ex['train'],index.sids,indices,size,'fp16',mode,c['training'],spec,deadline,'cuda:0',batch_builder=builder)
            except Exception as e:
                row=dict(status='failed',physical_batch_size=size,mode=mode,exception_type=type(e).__name__,message=str(e));gc.collect();torch.cuda.empty_cache()
            row.update(started_unix=begin,finished_unix=time.time());rows.append(row);_write(out/'train-profile.json',dict(cases=rows))
            _event('kindle_profile_case_complete',**row)
    operator=detailed_training_profile(model,ex['train'],index.sids,indices,512,'fp16',c['training'],'cuda:0',out,batch_builder=builder,export_trace=True)
    _write(out/'operator-profile.json',operator)
    del model;gc.collect();torch.cuda.empty_cache()
    # Keep the compiler and its subprocesses out of the core benchmark and bound
    # its cold cost independently. A timeout is a measurement, never a fallback.
    log=(out/'compile-worker.log').open('w');begin=time.time()
    process=subprocess.Popen([sys.executable,'-u','-m','rhpr.gr.kindle_profile','--plan',p['plan_path'],'--compile'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=process.wait(timeout=120)
        if code!=0:_write(out/'compile-profile.json',dict(status='failed',exit_code=code))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait();_write(out/'compile-profile.json',dict(status='cold_compile_timeout',timeout_seconds=120))
    finally:log.close()
    compile_report=json.loads((out/'compile-profile.json').read_text());compile_report.update(started_unix=begin,finished_unix=time.time());_write(out/'compile-profile.json',compile_report)
    if not any(x.get('status')=='ok'and x.get('physical_batch_size')==512 and x.get('mode')=='full_pipeline'for x in rows):raise RuntimeError('baseline training profile failed')
    return dict(seed=42,cases=len(rows),quality_training=False)


def eval_profile(p):
    from rhpr.gr.profiling import benchmark_evaluation,prediction_agreement,ProfileDeadline
    model,index,ex,builder=setup_model(p);out=Path(p['output']);deadline=ProfileDeadline(p['config']['profile']['max_seconds'])
    rows=[];reference=None
    for size in (32,64,128,256):
        _event('kindle_profile_eval_start',batch_size=size);begin=time.time();measurements=[];ids=None
        try:
            for repeat in range(2):
                row,ids=benchmark_evaluation(model,ex['validation'],index,size,'fp32','cuda:0',deadline,warmup=1,batch_builder=builder);measurements.append(row)
            if reference is None:reference=ids.copy()
            row=dict(batch_size=size,status='ok',measurements=measurements,agreement=prediction_agreement(reference,ids),started_unix=begin,finished_unix=time.time())
            np.savez_compressed(out/f'validation-batch-{size}.npz',semantic_ids=ids,user_ids=ex['validation'][1],target_item_ids=ex['validation'][2])
        except Exception as e:row=dict(batch_size=size,status='failed',exception_type=type(e).__name__,message=str(e),started_unix=begin,finished_unix=time.time())
        rows.append(row);_write(out/'evaluation-profile.json',dict(cases=rows));_event('kindle_profile_eval_complete',**row)
    if rows[0]['status']!='ok':raise RuntimeError('baseline validation profile failed')
    return dict(seed=42,cases=len(rows),test_evaluated=False)


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());p['plan_path']=str(plan_path)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5)or transformers.__version__!='5.22.1':raise RuntimeError('requires isolated T4/HF5.22.1')
    trial=p['assignments'][slot]['trials'][0]['id'];status=dict(results={},failures=[])
    try:
        _event('kindle_profile_trial_start',trial=trial);row=train_profile(p)if slot==0 else eval_profile(p);status['results'][trial]=row
    except Exception as e:
        (Path(p['output'])/f'worker-{slot}-failure.log').write_text(traceback.format_exc());status['failures'].append(dict(trial=trial,message=str(e),exception_type=type(e).__name__))
    _write(Path(p['output'])/'workers'/f'worker-{slot}.json',status)


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.data import file_sha256
    from rhpr.offline_runtime import resolve_dataset_directory
    if (c['kind']!='gr_kindle_runtime_profile'or c['evaluation']['test_after_selection']or c['profile']['max_seconds']!=420
            or c['training']['batch_size']!=512 or c['training']['precision']!='fp16'
            or c['generator']['attn_implementation']!='sdpa'):raise ValueError('invalid bounded profile')
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):raise RuntimeError('requires two T4s')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    parent=locate_parent(c['parent']);old=json.loads((parent/'run-manifest.json').read_text())['config']
    if old['generator']!=c['generator']or old['data']!=c['data']:raise ValueError('profile changes model or data')
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if name!='test-cases.npz'and file_sha256(data/name)!=digest:raise ValueError('data pin differs')
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid');overlay=scratch/'modern-site';overlay.mkdir(exist_ok=True)
    with (output/'modern-install.log').open('w')as f:subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),parent=str(parent),assignments=[dict(slot=i,device=f'cuda:{i}',trials=[c['trials'][i]])for i in (0,1)])
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    stream=(output/'gpu-utilization.csv').open('w');sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits','--loop-ms=200'],stdout=stream,stderr=subprocess.STDOUT)
    try:summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_profile',event_prefix='kindle_profile')
    finally:sampler.terminate();sampler.wait(timeout=10);stream.close()
    _write(output/'study-summary.json',summary)
    return dict(experiment=c['name'],profile_complete=True,quality_training=False,test_evaluated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int);p.add_argument('--compile',action='store_true');a=p.parse_args()
    if a.compile:
        plan=json.loads(Path(a.plan).read_text());train_profile(plan,True)
    else:worker(a.plan,a.slot)
