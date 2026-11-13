"""Deterministic ML100K leave-last-two-out data for the pair SID study."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np

def prepare(archive, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        rows=[tuple(map(int,line.split('\t'))) for line in z.read('ml-100k/u.data').decode().splitlines()]
        genres=[line.split('|')[0] for line in z.read('ml-100k/u.genre').decode().splitlines() if line.strip()]
        items=[line.split('|') for line in z.read('ml-100k/u.item').decode('latin1').splitlines()]
    assert len(rows)==100000 and len(items)==1682 and len(genres)==19
    items.sort(key=lambda x:int(x[0]));assert [int(x[0]) for x in items]==list(range(1,1683))
    texts=[]
    for x in items:
        selected=[g for g,v in zip(genres,x[5:]) if int(v)]
        texts.append('Title: '+x[1]+'. Genres: '+', '.join(selected))
    a=np.array(rows,dtype=np.int64);a=a[np.lexsort((a[:,1],a[:,3],a[:,0]))]
    groups=np.split(a,np.flatnonzero(np.diff(a[:,0]))+1);cases={k:[[],[],[]] for k in ('train','validation','test')};ties=0
    for g in groups:
        if len(g)<5:continue
        uid=int(g[0,0]);seq=g[:,1]-1;ties+=int(g[-1,3]==g[-2,3])
        for split,positions in [('train',range(1,len(g)-2)),('validation',[len(g)-2]),('test',[len(g)-1])]:
            for i in positions:
                hist=seq[max(0,i-20):i];row=np.full(20,-1,dtype=np.int64);row[-len(hist):]=hist
                cases[split][0].append(row);cases[split][1].append(uid);cases[split][2].append(seq[i])
    report={'users':len(cases['validation'][1]),'items':len(items),'train_examples':len(cases['train'][1]),'val_test_same_timestamp':ties,'tie_break':'timestamp then original MovieLens item ID','text':'MovieLens title (includes year) and genre flags','min_interactions':5,'train':'all noninitial prefixes before val','test_includes_val':True,'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest()}
    for name,values in cases.items():
        h,u,t=(np.asarray(v,dtype=np.int64) for v in values);np.savez_compressed(out/(name+'-cases.npz'),contexts=h,user_ids=u,targets=t)
    (out/'texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n')
    (out/'data-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();prepare(a.archive,a.output)
