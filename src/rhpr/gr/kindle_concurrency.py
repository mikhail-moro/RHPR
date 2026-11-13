"""Bounded one-versus-two compiled trials on one T4; optional verified MPS."""
from __future__ import annotations
import argparse,json,os,re,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_write
from rhpr.gr.study_parallel import atomic_json,worker_environment


def validate(c):
    if (c['kind']!='gr_kindle_concurrency' or c['profile']!={'max_seconds':600,'steps':100,'repeats':3}
            or c['evaluation']['test_after_selection'] or c['training']['batch_size']!=512
            or c['training']['physical_batch_size']!=512 or c['training']['precision']!='fp16'
            or c['training']['evaluation_precision']!='fp32' or c['execution']['cpu_threads_per_worker']!=1):
        raise ValueError('invalid concurrency budget or fixed workload')


def aggregate(rows):
    if not rows or any(x.get('status')!='ok' for x in rows):raise ValueError('incomplete concurrency round')
    if any(x['applied_updates']!=x['attempted_updates']for x in rows):raise ValueError('AMP-skipped updates invalidate throughput')
    start=min(x['started_unix']for x in rows);end=max(x['finished_unix']for x in rows)
    if end<=start:raise ValueError('invalid timing window')
    return dict(workers=len(rows),updates=sum(x['applied_updates']for x in rows),seconds=end-start,
        updates_per_second=sum(x['applied_updates']for x in rows)/(end-start),
        start_skew_seconds=max(x['started_unix']for x in rows)-start,started_unix=start,finished_unix=end,workers_detail=rows)


def paths(p,worker,sequence):
    root=Path(p['ipc']);return root/f'command-{worker}-{sequence}.json',root/f'result-{worker}-{sequence}.json'


def worker(plan_path,identifier):
    import torch,transformers
    from rhpr.gr.kindle_profile import setup_model
    from rhpr.gr.profiling import _step
    from rhpr.gr.experiment import _evaluate
    p=json.loads(Path(plan_path).read_text());out=Path(p['ipc']);out.mkdir(parents=True,exist_ok=True)
    try:
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5)or transformers.__version__!='5.22.1':raise RuntimeError('requires isolated T4/HF5.22.1')
        model,index,examples,builder=setup_model(p);initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        selected=np.random.default_rng(42).permutation(len(examples['train'][2]))[:512]
        batch=index.model_batch(*(x[selected]for x in examples['train']),'cuda:0')
        compiled=torch.compile(model,mode='reduce-overhead',fullgraph=False,dynamic=False)
        optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        provider=lambda start,end:{k:v[start:end]for k,v in batch.items()}
        atomic_json(out/f'ready-{identifier}.json',dict(status='ok',pid=os.getpid()))
        sequence=0
        while True:
            command_path,result_path=paths(p,identifier,sequence)
            if not command_path.exists():time.sleep(.01);continue
            command=json.loads(command_path.read_text());kind=command['kind'];started=time.monotonic()
            if kind=='quit':break
            try:
                if kind=='prepare':
                    model.load_state_dict(initial);optimizer.state.clear();torch.manual_seed(command['seed']);torch.cuda.manual_seed_all(command['seed']);model.train()
                    for _ in range(8):
                        m=_step(compiled,provider,512,512,optimizer,scaler,'fp16','cuda:0',1.)
                        if not m['optimizer_update_applied']:raise FloatingPointError('warm-up AMP update skipped')
                    torch.cuda.synchronize();row=dict(status='ok',seconds=time.monotonic()-started)
                else:
                    if kind not in ('train','validation'):raise ValueError('unknown concurrency command')
                    if kind=='validation':model.load_state_dict(initial)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    atomic_json(out/f'armed-{identifier}-{sequence}.json',dict(status='armed'))
                    gate=Path(command['gate'])
                    while not gate.exists():time.sleep(.005)
                    start_at=json.loads(gate.read_text())['start_unix']
                    while time.time()<start_at:time.sleep(.001)
                    begin=time.time();times=[];applied=0
                    if kind=='train':
                        model.train()
                        for _ in range(command['steps']):
                            before=time.perf_counter();m=_step(compiled,provider,512,512,optimizer,scaler,'fp16','cuda:0',1.);torch.cuda.synchronize();times.append(time.perf_counter()-before);applied+=m['optimizer_update_applied']
                        row=dict(status='ok',attempted_updates=command['steps'],applied_updates=applied,step_seconds=times,last_loss=m['loss'])
                    else:
                        values,ids,scores=_evaluate(model,examples['validation'],index,256,'cuda:0','fp32')
                        np.savez_compressed(out/f'validation-{identifier}.npz',semantic_ids=ids,user_ids=examples['validation'][1],target_item_ids=examples['validation'][2])
                        row=dict(status='ok',examples=2625,validation_metrics=values)
                    torch.cuda.synchronize();row.update(started_unix=begin,finished_unix=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(result_path,row)
            except Exception as e:
                atomic_json(result_path,dict(status='failed',exception_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
            sequence+=1
    except Exception as e:
        atomic_json(out/f'fatal-{identifier}.json',dict(status='failed',message=str(e),traceback=traceback.format_exc()));raise


class Peer:
    def __init__(self,p,identifier,environment):
        self.p=p;self.identifier=identifier;self.sequence=0
        self.log=(Path(p['ipc'])/f'worker-{identifier}.log').open('w')
        self.process=subprocess.Popen([sys.executable,'-u','-m','rhpr.gr.kindle_concurrency','--plan',p['plan_path'],'--worker',str(identifier)],env=environment,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
    def send(self,command):
        self.pending=self.sequence;self.sequence+=1;path,_=paths(self.p,self.identifier,self.pending);atomic_json(path,command)
    def result(self):return paths(self.p,self.identifier,self.pending)[1]
    def armed(self):return Path(self.p['ipc'])/f'armed-{self.identifier}-{self.pending}.json'
    def close(self):
        if self.process.poll()is None:
            try:self.send(dict(kind='quit'));self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        self.log.close()


def wait_files(peers,files,deadline,timeout):
    limit=min(deadline,time.monotonic()+timeout)
    while not all(p.exists()for p in files):
        for peer in peers:
            fatal=Path(peer.p['ipc'])/f'fatal-{peer.identifier}.json'
            if fatal.exists():raise RuntimeError(fatal.read_text())
            if peer.process.poll()is not None:raise RuntimeError('benchmark worker exited; inspect worker log')
        if time.monotonic()>limit:raise TimeoutError('bounded concurrency command timed out')
        time.sleep(.02)


def prepare(peers,round_id,deadline):
    _event('concurrency_warmup_start',workers=len(peers),round=round_id)
    for peer in peers:peer.send(dict(kind='prepare',seed=42+round_id+peer.identifier))
    wait_files(peers,[peer.result()for peer in peers],deadline,150)
    for peer in peers:
        row=json.loads(peer.result().read_text())
        if row['status']!='ok':raise RuntimeError(str(row))


def measure(peers,kind,tag,deadline):
    gate=Path(peers[0].p['ipc'])/f'gate-{tag}.json'
    for peer in peers:peer.send(dict(kind=kind,steps=100,gate=str(gate)))
    wait_files(peers,[peer.armed()for peer in peers],deadline,60)
    atomic_json(gate,dict(start_unix=time.time()+.2))
    wait_files(peers,[peer.result()for peer in peers],deadline,90)
    rows=[json.loads(peer.result().read_text())for peer in peers]
    if kind=='train':return aggregate(rows)
    return dict(workers_detail=rows,all_fit=all(x['status']=='ok'for x in rows))


def mps_clients(binary,environment,peers):
    def query(command):return subprocess.run([binary],input=command+'\n',text=True,capture_output=True,env=environment,timeout=10)
    servers=query('get_server_list');pids=[int(x)for x in re.findall(r'^\s*(\d+)\s*$',servers.stdout,re.M)];clients=set()
    for pid in pids:
        reply=query(f'get_client_list {pid}');clients.update(int(x)for x in re.findall(r'^\s*(\d+)\s*$',reply.stdout,re.M))
    return dict(verified=all(peer.process.pid in clients for peer in peers),server_pids=pids,client_pids=sorted(clients),expected_client_pids=[peer.process.pid for peer in peers])


def mode(p,name,environment,deadline,binary=None):
    peers=[];report=dict(status='running',single=[],paired=[],single_after=[]);path=Path(p['output'])/f'{name}-results.json'
    p={**p,'ipc':str(Path(p['output'])/name)};Path(p['ipc']).mkdir(exist_ok=True);p['plan_path']=str(Path(p['ipc'])/'plan.json');_write(Path(p['plan_path']),p)
    try:
        peers.append(Peer(p,0,environment));wait_files(peers,[Path(p['ipc'])/'ready-0.json'],deadline,90)
        for i in range(3):
            prepare(peers,i,deadline);row=measure(peers,'train',f'single-{i}',deadline);report['single'].append(row);_write(path,report)
            _event('concurrency_round_complete',mode=name,round_kind='single',repeat=i,updates_per_second=row['updates_per_second'])
        if binary:
            report['mps_single_verification']=mps_clients(binary,environment,peers)
            if not report['mps_single_verification']['verified']:raise RuntimeError('MPS client registration unverified')
        peers.append(Peer(p,1,environment));wait_files(peers,[Path(p['ipc'])/'ready-1.json'],deadline,90)
        for i in range(3):
            prepare(peers,i+10,deadline)
            if binary:
                report['mps_pair_verification']=mps_clients(binary,environment,peers)
                if not report['mps_pair_verification']['verified']:raise RuntimeError('two MPS clients unverified')
            row=measure(peers,'train',f'pair-{i}',deadline);report['paired'].append(row);_write(path,report)
            _event('concurrency_round_complete',mode=name,round_kind='paired',repeat=i,updates_per_second=row['updates_per_second'])
        for i in range(3):
            prepare(peers[:1],i+20,deadline);row=measure(peers[:1],'train',f'after-{i}',deadline);report['single_after'].append(row);_write(path,report)
            _event('concurrency_round_complete',mode=name,round_kind='single_after',repeat=i,updates_per_second=row['updates_per_second'])
        report['concurrent_validation']=measure(peers,'validation','validation',deadline)
        base=float(np.median([x['updates_per_second']for x in report['single']]))
        paired=float(np.median([x['updates_per_second']for x in report['paired']]))
        report.update(status='ok',single_updates_per_second_median=base,paired_updates_per_second_median=paired,aggregate_speedup=paired/base)
    except Exception as e:report.update(status='failed',exception_type=type(e).__name__,message=str(e))
    finally:
        for peer in peers:peer.close()
        _write(path,report)
    return report


def run(c,output):
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.data import file_sha256
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);parent=locate_parent(c['parent'])
    old=json.loads((parent/'run-manifest.json').read_text())['config']
    if c['generator']!=old['generator']or c['data']!=old['data']:raise ValueError('model/data changed')
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if name!='test-cases.npz'and file_sha256(data/name)!=digest:raise ValueError('data pin differs')
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid');overlay=scratch/'modern-site';overlay.mkdir(exist_ok=True)
    with (output/'modern-install.log').open('w')as f:subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    sys.path.insert(0,str(overlay))
    hardware=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    rows=[x.strip().split(', ')for x in hardware.strip().splitlines()];visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
    gpu=next((x for x in rows if x[0]==visible or x[1]==visible),rows[0]);
    if 'T4'not in gpu[2]:raise RuntimeError('requires a T4')
    _write(output/'hardware.json',dict(gpu_index=gpu[0],gpu_name=gpu[2],vram_mib=float(gpu[3]),cpu_threads_per_worker=1))
    p=dict(config=c,output=str(output),source=str(source),data=str(data),parent=str(parent))
    env=worker_environment(c,dict(device='cuda:0'));env['CUDA_VISIBLE_DEVICES']=gpu[1];env['TORCHINDUCTOR_CACHE_DIR']=str(scratch/'compile-cache')
    for key in list(env):
        if key.startswith('CUDA_MPS_'):env.pop(key)
    deadline=time.monotonic()+600
    stream=(output/'gpu-utilization.csv').open('w');sampler=subprocess.Popen(['nvidia-smi',f'--id={gpu[0]}','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits','--loop-ms=200'],stdout=stream,stderr=subprocess.STDOUT)
    results={};binary=shutil.which('nvidia-cuda-mps-control');mps=dict(binary_available=bool(binary),attempted=False)
    _event('concurrency_capability',mps_binary_available=bool(binary),gpu_name=gpu[2])
    try:
        _event('concurrency_mode_start',mode='ordinary');results['ordinary']=mode(p,'ordinary',env,deadline);_write(output/'study-summary.json',dict(results=results,mps=mps))
        if binary and time.monotonic()+180<deadline:
            mpsenv={**env,'CUDA_MPS_PIPE_DIRECTORY':str(scratch/'mps-pipe'),'CUDA_MPS_LOG_DIRECTORY':str(scratch/'mps-log')}
            for key in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY'):Path(mpsenv[key]).mkdir(exist_ok=True)
            start=subprocess.run([binary,'-d'],env=mpsenv,capture_output=True,text=True,timeout=15);mps.update(attempted=True,start_returncode=start.returncode,start_message=(start.stdout+start.stderr)[-2000:])
            if start.returncode==0:
                try:_event('concurrency_mode_start',mode='mps');results['mps']=mode(p,'mps',mpsenv,deadline,binary)
                finally:subprocess.run([binary],input='quit\n',text=True,env=mpsenv,capture_output=True,timeout=15)
        elif binary:mps['not_attempted_reason']='remaining bounded budget too short'
    finally:
        sampler.terminate();sampler.wait(timeout=10);stream.close();_write(output/'study-summary.json',dict(results=results,mps=mps,test_evaluated=False))
    if results['ordinary']['status']!='ok':raise RuntimeError('ordinary concurrency benchmark incomplete; partial measurements saved')
    metrics=dict(experiment=c['name'],profile_complete=True,test_evaluated=False,quality_training=False,ordinary_aggregate_speedup=results['ordinary']['aggregate_speedup'],ordinary_concurrent_validation_fits=results['ordinary']['concurrent_validation']['all_fit'],mps_binary_available=bool(binary),mps_verified=results.get('mps',{}).get('status')=='ok')
    if metrics['mps_verified']:metrics['mps_aggregate_speedup']=results['mps']['aggregate_speedup']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--worker',type=int,required=True);a=p.parse_args();worker(a.plan,a.worker)
