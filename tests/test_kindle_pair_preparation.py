import contextlib,gzip,importlib.util,io,json,tempfile,unittest
from pathlib import Path
import numpy as np

class KindlePreparationTest(unittest.TestCase):
    def test_dense_sample_preserves_eligible_user_histories(self):
        spec=importlib.util.spec_from_file_location('kindle_dense',Path(__file__).resolve().parents[1]/'scripts/prepare_kindle_dense_pair_pilot.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            with gzip.open(root/'ratings.csv.gz','wt')as f:
                f.write('user_id,parent_asin,rating,timestamp\n')
                for u in range(30):
                    for i in range(30):f.write(f'u{u},b{(u+i)%50:02d},5,{i+100}\n')
            catalog=[dict(parent_asin=f'b{i:02d}',title=f'Book{i}',categories=['Fiction'])for i in range(50)]
            (root/'catalog.json').write_text(json.dumps(catalog))
            with contextlib.redirect_stdout(io.StringIO()):module.prepare(root,root/'out',root/'catalog.json',target=300,top_items=50)
            report=json.loads((root/'out/data-report.json').read_text());self.assertEqual(report['events'],300);self.assertEqual(report['users'],10)
            audit=json.loads((root/'out/user-audit.json').read_text());ids=json.loads((root/'out/source-ids.json').read_text())
            for split in ('train','validation','test'):
                z=np.load(root/'out'/(split+'-cases.npz'))
                for h,uid,t in zip(z['contexts'],z['user_ids'],z['targets']):
                    original=int(audit[int(uid)-1]['source_user_id'][1:]);seq=[f'b{(original+i)%50:02d}'for i in range(30)];at=seq.index(ids[int(t)])
                    self.assertEqual([ids[int(i)]for i in h if i>=0],seq[max(0,at-20):at])
                    if split=='train':self.assertLess(at,28)
                    if split=='validation':self.assertEqual(at,28)
                    if split=='test':self.assertEqual(at,29)

    def test_whole_positive_histories_and_leave_last_two(self):
        spec=importlib.util.spec_from_file_location('kindle_prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_kindle_pair_pilot.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);source={}
            with gzip.open(root/'ratings.csv.gz','wt')as f:
                f.write('user_id,parent_asin,rating,timestamp\n')
                for u in range(30):
                    rows=[]
                    for i in range(8):
                        asin=f'b{u:03d}_{i}';rating=3 if i==1 else 5;t=100+i
                        f.write(f'u{u},{asin},{rating},{t}\n')
                        if rating>=4 and i!=2:rows.append((t,asin))
                    source[f'u{u}']=rows
            with gzip.open(root/'metadata.jsonl.gz','wt')as f:
                for u in range(30):
                    for i in range(8):
                        f.write(json.dumps(dict(parent_asin=f'b{u:03d}_{i}',title=''if i==2 else f'Book{i}',categories=[]))+'\n')
            with contextlib.redirect_stdout(io.StringIO()):module.prepare(root,root/'a',target=48,seed=42);module.prepare(root,root/'b',target=48,seed=42)
            audit=json.loads((root/'a/user-audit.json').read_text());ids=json.loads((root/'a/source-ids.json').read_text())
            self.assertEqual(sum(x['events']for x in audit),48)
            for split in ('train','validation','test'):
                a=np.load(root/'a'/(split+'-cases.npz'));b=np.load(root/'b'/(split+'-cases.npz'))
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                for h,uid,t in zip(a['contexts'],a['user_ids'],a['targets']):
                    row=audit[int(uid)-1];seq=[asin for time,asin in source[row['source_user_id']]];target=ids[int(t)];at=seq.index(target)
                    self.assertEqual([ids[int(i)]for i in h if i>=0],seq[max(0,at-20):at])
                    if split=='validation':self.assertEqual(at,len(seq)-2)
                    if split=='test':self.assertEqual(at,len(seq)-1)
                    if split=='train':self.assertLess(at,len(seq)-2)
