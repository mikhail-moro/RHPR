import contextlib,copy,io,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rhpr.gr.kindle_concurrency import aggregate,validate,measure,paths,mps_clients,mode
from rhpr.gr.study_parallel import atomic_json

class ConcurrencyProbeTest(unittest.TestCase):
    def test_complete_mode_runs_all_rounds_with_real_event_logger(self):
        class StubPeer:
            def __init__(self,*args):self.process=SimpleNamespace(pid=123)
            def close(self):pass
        def measured(peers,kind,tag,deadline):
            if kind=='validation':return dict(all_fit=True,workers_detail=[])
            return dict(updates_per_second=10.*len(peers))
        with tempfile.TemporaryDirectory()as tmp,contextlib.redirect_stdout(io.StringIO()),\
                patch('rhpr.gr.kindle_concurrency.Peer',StubPeer),\
                patch('rhpr.gr.kindle_concurrency.prepare'),\
                patch('rhpr.gr.kindle_concurrency.wait_files'),\
                patch('rhpr.gr.kindle_concurrency.measure',side_effect=measured):
            row=mode(dict(output=tmp),'ordinary',{},time.monotonic()+60)
            self.assertEqual(row['status'],'ok',row)
            self.assertEqual([len(row[k])for k in ('single','paired','single_after')],[3,3,3])
            self.assertEqual(row['aggregate_speedup'],2.)

    def test_aggregate_counts_both_workers_over_common_wall_window(self):
        rows=[dict(status='ok',applied_updates=100,attempted_updates=100,started_unix=10.,finished_unix=20.),dict(status='ok',applied_updates=100,attempted_updates=100,started_unix=10.1,finished_unix=20.2)]
        x=aggregate(rows);self.assertEqual(x['updates'],200);self.assertAlmostEqual(x['seconds'],10.2);self.assertAlmostEqual(x['updates_per_second'],200/10.2)
        rows[1]['applied_updates']=99
        with self.assertRaises(ValueError):aggregate(rows)

    def test_mps_must_register_every_expected_client(self):
        peers=[SimpleNamespace(process=SimpleNamespace(pid=111)),SimpleNamespace(process=SimpleNamespace(pid=222))]
        with patch('rhpr.gr.kindle_concurrency.subprocess.run',side_effect=[SimpleNamespace(stdout='999\n'),SimpleNamespace(stdout='111\n222\n')]):self.assertTrue(mps_clients('mps',{},peers)['verified'])
        with patch('rhpr.gr.kindle_concurrency.subprocess.run',side_effect=[SimpleNamespace(stdout='999\n'),SimpleNamespace(stdout='111\n')]):self.assertFalse(mps_clients('mps',{},peers)['verified'])

    def test_config_cannot_spend_extra_steps_or_enable_test(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/kindle_concurrency_probe.json').read_text());validate(c)
        for key in ('test','steps','batch'):
            b=copy.deepcopy(c)
            if key=='test':b['evaluation']['test_after_selection']=True
            if key=='steps':b['profile']['steps']=1000
            if key=='batch':b['training']['batch_size']=1024
            with self.assertRaises(ValueError):validate(b)

    def test_two_processes_arm_before_common_gate_and_results_are_complete(self):
        script='''import json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);i=int(sys.argv[2]);command=r/f"command-{i}-0.json"
while not command.exists():time.sleep(.005)
c=json.loads(command.read_text());(r/f"armed-{i}-0.json").write_text('{}');gate=Path(c['gate'])
while not gate.exists():time.sleep(.005)
t=json.loads(gate.read_text())['start_unix']
while time.time()<t:time.sleep(.001)
a=time.time();time.sleep(.05);row=dict(status='ok',attempted_updates=100,applied_updates=100,started_unix=a,finished_unix=time.time())
p=r/f"result-{i}-0.json";tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(p);time.sleep(2)
'''
        with tempfile.TemporaryDirectory()as tmp:
            class FakePeer:
                def __init__(self,i):self.identifier=i;self.p={'ipc':tmp};self.pending=0;self.process=subprocess.Popen([sys.executable,'-c',script,tmp,str(i)])
                def send(self,c):atomic_json(paths(self.p,self.identifier,0)[0],c)
                def result(self):return paths(self.p,self.identifier,0)[1]
                def armed(self):return Path(tmp)/f'armed-{self.identifier}-0.json'
            peers=[FakePeer(0),FakePeer(1)]
            try:
                row=measure(peers,'train','test',time.monotonic()+10);self.assertEqual(row['workers'],2);self.assertEqual(row['updates'],200)
            finally:
                for peer in peers:peer.process.terminate();peer.process.wait()
