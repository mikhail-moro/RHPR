"""Fit frozen ALS on distinct edges appearing ONLY in admitted TRAIN examples."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rhpr.gr.sid_graph_search import fit_als_signal
from rhpr.gr.movie_data import case_digest


def train_users(contexts,user_ids,targets):
    seen={}
    for h,u,t in zip(contexts,user_ids,targets):
        row=seen.setdefault(int(u),set());row.update(int(i)for i in h if i>=0);row.add(int(t))
    return [SimpleNamespace(train=np.asarray(sorted(seen[u]),dtype=np.int64))for u in sorted(seen)]


def prepare(data,out):
    out.mkdir(parents=True,exist_ok=True)
    with np.load(data/'train-cases.npz')as z:
        cases=tuple(z[k]for k in ['contexts','user_ids','targets'])
    report=json.loads((data/'split-report.json').read_text());assert case_digest(cases)==report['case_sha256']['train']
    spec=dict(factors=128,regularization=.1,alpha=20,iterations=20,seed=42,cpu_threads=1)
    factors,_,frequency,diagnostics=fit_als_signal(train_users(*cases),2874,spec)
    observed=frequency>0;factors[~observed]=0
    rng=np.random.default_rng(42);eligible=np.flatnonzero(observed);permutation=np.arange(len(factors))
    # Derange supported rows only; retain missing-data masks and vector multiset.
    while True:
        shuffled=rng.permutation(eligible)
        if (shuffled!=eligible).all():break
    permutation[eligible]=shuffled
    np.save(out/'als.npy',factors,allow_pickle=False);np.save(out/'shuffled.npy',factors[permutation],allow_pickle=False)
    np.save(out/'observed.npy',observed,allow_pickle=False);np.save(out/'permutation.npy',permutation,allow_pickle=False)
    np.save(out/'frequency.npy',frequency,allow_pickle=False)
    (out/'source-ids.json').write_bytes((data/'source-ids.json').read_bytes())
    diagnostics.update(train_case_sha256=case_digest(cases),users=2786,items=2874,observed_items=int(observed.sum()),no_val_test_read=True,source='union of TRAIN contexts and targets, binary unique user-item edges',all_ratings_are_interactions=True,shuffle_seed=42,shuffle_supported_only=True,fixed_points_supported=0)
    assert diagnostics['binary_interactions']==303199 and observed.sum()==2777
    (out/'als-report.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(files=[dict(path=p.name,size_bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.iterdir())if p.name not in {'resources-manifest.json','dataset-metadata.json'}])
    (out/'resources-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'dataset-metadata.json').write_text(json.dumps(dict(title='RHPR ML1M TRAIN-only ALS input',id='carasik/rhpr-ml1m-als-input',licenses=[dict(name='other')]),indent=2))
    print(json.dumps(dict(manifest_sha256=sha(out/'resources-manifest.json'),**diagnostics),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.data,a.output)
