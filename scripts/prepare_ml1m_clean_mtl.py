"""Build clean ML1M cases and public multimodal inputs; no model training."""
import argparse,collections,csv,gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from rhpr.gr.movie_data import case_digest
from scripts.prepare_movielens_text_ladder import field_texts

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build_clean_cases(a,ids):
    index={i:j for j,i in enumerate(ids)}
    a=a[np.lexsort((a[:,1],a[:,3],a[:,0]))];lookup=np.zeros(int(a[:,1].max())+1,dtype=bool);lookup[ids]=True
    cases={k:[[],[],[]]for k in ['train','validation','test']};users=[]
    for g in np.split(a,np.flatnonzero(np.diff(a[:,0]))+1):
        times,starts,counts=np.unique(g[:,3],return_index=True,return_counts=True)
        valid=lookup[g[:,1]];group_valid=np.logical_and.reduceat(valid,starts);single=counts==1
        row_group=np.repeat(np.arange(len(times)),counts);bad=np.r_[0,np.cumsum(~valid)]
        hstart=np.maximum(starts-20,0);boundary=starts[row_group[hstart]]
        clean=(starts>0)&((bad[starts]-bad[boundary])==0)
        eligible=group_valid&single&clean&(starts>=20)
        candidates=np.flatnonzero(eligible[:-1]&eligible[1:])
        if not len(candidates):continue
        vg=int(candidates[-1]);tg=vg+1
        if vg<2:continue
        training_groups=np.flatnonzero(group_valid[:vg]&clean[:vg])
        if not len(training_groups):continue
        uid=int(g[0,0]);ntrain=0
        for split,groups in [('train',training_groups),('validation',[vg]),('test',[tg])]:
            for j in groups:
                begin=int(starts[j]);past=g[max(0,begin-20):begin]
                assert lookup[past[:,1]].all()and group_valid[j]
                h=[-1]*(20-len(past))+[index[int(i)]for i in past[:,1]]
                for row in g[begin:begin+int(counts[j])]:
                    cases[split][0].append(h);cases[split][1].append(uid);cases[split][2].append(index[int(row[1])])
                    if split=='train':ntrain+=1
        assert starts[tg]==starts[vg]+1 and counts[vg]==counts[tg]==1
        users.append(dict(user_id=uid,validation_time=int(times[vg]),test_time=int(times[tg]),train_cases=ntrain))
    return cases,users


def prepare(movielens,masks,fields,plots,posters,output,expected_audit=None):
    output.mkdir(parents=True,exist_ok=True)
    mask=pd.read_parquet(masks);ids=sorted(mask.loc[mask['filter'].eq('tmdb_rich4_and_zenodo_plot_vit'),'movie_id'].astype(int));assert len(ids)==2874
    index={i:j for j,i in enumerate(ids)}
    with zipfile.ZipFile(movielens)as z:a=np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::',',')),delimiter=',',dtype=np.int64)
    assert a.shape==(1000209,4)
    cases,users=build_clean_cases(a,ids)
    assert len(users)==2786 and len(cases['train'][2])==151166
    if expected_audit:
        expected={int(r['user_id']):(int(r['validation_time']),int(r['test_time']),int(r['train_cases']))for r in csv.DictReader(expected_audit.open())}
        assert {r['user_id']:(r['validation_time'],r['test_time'],r['train_cases'])for r in users}==expected
    hashes={}
    for split,v in cases.items():
        arrays=tuple(np.asarray(x,dtype=np.int64)for x in v);assert arrays[0].shape==(len(arrays[2]),20)
        if split!='train':assert len(arrays[2])==2786 and (arrays[0]>=0).all()
        hashes[split]=case_digest(arrays)
        np.savez_compressed(output/(split+'-cases.npz'),contexts=arrays[0],user_ids=arrays[1],targets=arrays[2])
    vh,vu,vt=map(lambda x:np.asarray(x,dtype=np.int64),cases['validation']);th,tu,tt=map(lambda x:np.asarray(x,dtype=np.int64),cases['test'])
    np.testing.assert_array_equal(vu,tu);np.testing.assert_array_equal(th[:,:-1],vh[:,1:]);np.testing.assert_array_equal(th[:,-1],vt)
    with gzip.open(fields,'rt')as f:catalog={r['item_id']:r for r in json.load(f)}
    texts=[field_texts(catalog[i]['fields'])[4]for i in ids]
    assert sha(plots)=='4676223139ff78710f4451bcca0cc389cca03303db6f4980d96eefc1f90925a3'
    with plots.open(encoding='utf-8',newline='')as f:plot_lookup={int(r['item']):r['description'].strip()for r in csv.DictReader(f)}
    descriptions=[plot_lookup[i]for i in ids];assert all(t and t.lower()not in {'none','null','nan'}for t in descriptions)
    assert hashlib.md5(posters.read_bytes()).hexdigest()=='8f184920d99edd58d1f2063450f4c0c6'
    with zipfile.ZipFile(posters)as z:poster_lookup=json.loads(z.read('json/vit_cls.json'))
    image=np.asarray([poster_lookup[str(i)]for i in ids],dtype=np.float32);assert image.shape==(2874,768)and np.isfinite(image).all()
    np.save(output/'poster.npy',image,allow_pickle=False)
    (output/'source-ids.json').write_text(json.dumps(ids)+'\n');(output/'metadata-texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n');(output/'plot-texts.json').write_text(json.dumps(descriptions,ensure_ascii=False)+'\n')
    (output/'holdout-audit.json').write_text(json.dumps(users,indent=2)+'\n')
    report=dict(items=2874,users=2786,train_examples=151166,validation_examples=2786,test_examples=2786,case_sha256=hashes,
        split='raw_contiguous_20_val_test',training='all earlier raw clean up-to20windows; natural original short prefixes only',
        target_selection='latest adjacent original raw singleton val/test with full20cleancontext and at least1earlier admitted training example',
        timestamp_groups='any uncovered member invalidates whole group, including touched left boundary group',all_ratings_retained=True,
        source_files=[dict(name=p.name,sha256=sha(p))for p in [movielens,masks,fields,plots,posters]],
        metadata_fields=['title_with_year','genres','explicit_year','runtime','tmdb_keywords'],plot_field='Zenodo description ONLY',no_interactions_used_for_features=True)
    (output/'split-report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=dict(files=[dict(path=p.name,size_bytes=p.stat().st_size,sha256=sha(p))for p in sorted(output.iterdir())if p.is_file()and p.name not in {'resources-manifest.json','dataset-metadata.json'}])
    (output/'resources-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'dataset-metadata.json').write_text(json.dumps(dict(title='RHPR ML1M clean MTL input',id='carasik/rhpr-ml1m-clean-mtl-input',licenses=[dict(name='other')]),indent=2))
    print(json.dumps(dict(**report,manifest_sha256=sha(output/'resources-manifest.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['movielens','masks','fields','plots','posters','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-audit',type=Path);a=p.parse_args();prepare(a.movielens,a.masks,a.fields,a.plots,a.posters,a.output,a.expected_audit)
