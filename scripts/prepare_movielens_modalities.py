"""Align frozen movie modalities to the exact existing ML100K catalogue."""
import argparse,csv,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np


def prepare(catalogue, text_embedding, zenodo_json, plot_csv, people_dir, output, plot_encoder, plot_revision):
    rows=json.loads(gzip.decompress(catalogue.read_bytes()));ids=[r['item_id'] for r in rows]
    if len(ids)!=1273 or len(set(ids))!=1273:raise ValueError('wrong canonical catalogue')
    if ids!=json.loads((text_embedding.parent/'sid-catalogue.json').read_text())['original_item_ids']:
        raise ValueError('text embedding row order differs from canonical catalogue')
    report=json.loads((text_embedding.parent/'representation.json').read_text())
    if (hashlib.sha256(text_embedding.read_bytes()).hexdigest()!=report['embedding_sha256']
            or report['stage']!=5 or not report['frozen']
            or report['fields']!=['title_with_year','genres','explicit_year','runtime','tmdb_keywords']):
        raise ValueError('base text embeddings are not the verified stage5 source')
    if hashlib.md5(zenodo_json.read_bytes()).hexdigest()!='8f184920d99edd58d1f2063450f4c0c6':raise ValueError('Zenodo archive changed')
    if hashlib.md5(plot_csv.read_bytes()).hexdigest()!='a98afbff73f503ce705613ed7e6efe9c':raise ValueError('plot source changed')
    with zipfile.ZipFile(zenodo_json) as z:posters=json.loads(z.read('json/vit_cls.json'))
    with plot_csv.open(encoding='utf-8',newline='') as f:
        plots={int(r['item']):r['description'].strip() for r in csv.DictReader(f)}
    text=np.load(text_embedding,allow_pickle=False);people=np.load(people_dir/'people.npy',allow_pickle=False)
    if text.shape!=(1273,768) or people.shape!=(1273,128):raise ValueError('wrong frozen feature dimensions')
    people_ids=json.loads((people_dir/'source-ids.json').read_text())
    if people_ids!=ids:raise ValueError('people catalogue order differs')
    poster=np.stack([posters[str(r['ml1m_id'])] for r in rows]).astype(np.float32)
    plot_texts=[plots.get(r['ml1m_id'],'') for r in rows]
    plot_texts=['' if t.lower() in {'nan','none','null'} else t for t in plot_texts]
    output.mkdir(parents=True,exist_ok=True)
    shutil.copytree(plot_encoder,output/'plot-encoder',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('.cache','__pycache__'))
    np.savez_compressed(output/'base-features.npz',text=text,poster=poster,people=people,
        text_observed=np.ones(1273,dtype=bool),poster_observed=np.ones(1273,dtype=bool),people_observed=np.ones(1273,dtype=bool))
    (output/'source-ids.json').write_text(json.dumps(ids)+'\n')
    (output/'plot-texts.json').write_text(json.dumps(plot_texts,ensure_ascii=False)+'\n')
    shutil.copy2(people_dir/'people-vectors.npz',output/'people-vectors.npz')
    provenance={'items':1273,'source_ids':'canonical ML100K IDs; external features joined through explicit ml1m_id bridge',
        'text':'previous frozen Sentence-T5 stage5: title including year,genres,explicit year,runtime,TMDBkeywords; no names',
        'poster':'Zenodo15403972 json/vit_cls.json,768D; keyed original ML1M IDs, not row-position NPY arrays',
        'plot':'Zenodo ml1m_text.tsv CSV description field ONLY; genres/tags/reviews columns excluded; encoded by pinned all-mpnet-base-v2 at run preparation',
        'plot_encoder':{'model':'sentence-transformers/all-mpnet-base-v2','revision':plot_revision,'max_length':384},
        'plot_observed':sum(bool(t) for t in plot_texts),'people':json.loads((people_dir/'people-provenance.json').read_text()),
        'source_files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [catalogue,text_embedding,zenodo_json,plot_csv,people_dir/'people.npy']],
        'no_user_interactions_used_for_features':True}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for p in output.glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            if any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError('nonfinite features')
    manifest={'files':[{'path':p.relative_to(output).as_posix(),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['resources-manifest.json','dataset-metadata.json']]}
    (output/'resources-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'items':1273,'plot_observed':provenance['plot_observed'],'manifest_sha256':hashlib.sha256((output/'resources-manifest.json').read_bytes()).hexdigest()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['catalogue','text-embedding','zenodo-json','plot-csv','people-dir','output','plot-encoder']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plot-revision',required=True)
    a=p.parse_args();prepare(a.catalogue,a.text_embedding,a.zenodo_json,a.plot_csv,a.people_dir,a.output,a.plot_encoder,a.plot_revision)
