"""ML100K ratings + exact title/year bridge to verified movie metadata.

No fuzzy matching, raw-ID reuse, encoder fitting or experiment dispatch.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote_plus, urlparse
import zipfile

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.movie_data import split_ratings, load_temporal_users, temporal_examples, case_digest
from scripts.prepare_movielens_text_ladder import field_texts, FIELDS

OFFICIAL_SHA='50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229'


def normalized_title(value):
    value=html.unescape(value)
    m=re.fullmatch(r'(.*), (The|A|An|Der|Das|Die|Le|La|Les|Il|El)',value)
    if m:value=m[2]+' '+m[1]
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',value).encode('ascii','ignore').decode().casefold())


def title_aliases(title):
    values=[title,re.sub(r'\([^()]*\)','',title).strip(),*re.findall(r'\(([^()]*)\)',title)]
    return {normalized_title(re.sub(r'^a\.?k\.?a\.?\s*','',v,flags=re.I)) for v in values if normalized_title(v)}


def build_bridge(items, audit):
    aliases=collections.defaultdict(set);imdb=collections.defaultdict(set)
    for mid,r in audit.items():
        m=re.search(r'\s*\((\d{4})\)$',r['title'])
        if m:
            for v in title_aliases(r['title'][:m.start()]):aliases[v,int(m[1])].add(mid)
        if r.get('metadata_present'):
            for v in title_aliases(r['tmdb_title']):aliases[v,r.get('tmdb_year')].add(mid)
            imdb[r['imdb_id']].add(mid)
    matches,unmatched,ambiguous=[],[],[]
    for row in items:
        raw_id,title,url=int(row[0]),row[1],row[4]
        direct=re.search(r'(?:tt|imdb-title-)(\d+)',url)
        candidates=imdb.get('tt'+direct[1].zfill(7),set()) if direct else set()
        method='explicit_imdb_id'
        if not candidates:
            method='exact_title_year_alias'
            candidates=set()
            for value in [title,unquote_plus(urlparse(url).query,encoding='latin-1')]:
                m=re.search(r'\((\d{4})\)',value)
                if m:
                    for v in title_aliases(value[:m.start()].strip()):candidates.update(aliases.get((v,int(m[1])),set()))
        if len(candidates)==1:
            matches.append({'ml100k_id':raw_id,'ml1m_id':next(iter(candidates)),'title':title,'method':method})
        elif candidates:ambiguous.append({'ml100k_id':raw_id,'title':title,'candidate_ml1m_ids':sorted(candidates)})
        else:unmatched.append({'ml100k_id':raw_id,'title':title})
    return matches,unmatched,ambiguous


def canonical_ratings(ratings, bridge):
    """Merge verified aliases; retain latest user-film rating, never prefer score."""
    groups=collections.defaultdict(list)
    for raw,mid in bridge.items():groups[mid].append(raw)
    canonical={raw:min(groups[mid]) for raw,mid in bridge.items()}
    kept=ratings[np.isin(ratings[:,1],list(bridge))].copy()
    before=len(kept);original_ids=kept[:,1].copy()
    kept[:,1]=[canonical[int(i)] for i in kept[:,1]]
    order=np.lexsort((original_ids,kept[:,3],kept[:,1],kept[:,0]));kept=kept[order]
    last=np.r_[np.any(kept[1:,:2]!=kept[:-1,:2],axis=1),True]
    return kept[last],canonical,before-int(last.sum())


def prepare(args):
    from transformers import AutoTokenizer
    import pandas as pd
    if file_sha256(args.movielens)!=OFFICIAL_SHA:raise ValueError('ML100K source archive changed')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.catalogue_audit,'rt') as f:audit={r['movie_id']:r for r in map(json.loads,f)}
    with gzip.open(args.fields,'rt') as f:fields={r['item_id']:r for r in json.load(f)}
    with zipfile.ZipFile(args.movielens) as z:
        items=[r.split('|') for r in z.read('ml-100k/u.item').decode('latin-1').splitlines()]
        ratings=np.loadtxt(io.StringIO(z.read('ml-100k/u.data').decode()),dtype=np.int64)
        genres={int(r.split('|')[1]):r.split('|')[0] for r in z.read('ml-100k/u.genre').decode().splitlines() if r.strip()}
    assert ratings.shape==(100000,4) and len(items)==1682 and len(np.unique(ratings[:,0]))==943
    matches,unmatched,ambiguous=build_bridge(items,audit)
    retained={r['ml100k_id']:r['ml1m_id'] for r in matches if r['ml1m_id'] in fields}
    clean,canonical,duplicate_ratings=canonical_ratings(ratings,retained)
    ids=sorted(set(canonical.values()));lookup={int(r[0]):r for r in items}
    mapped={canonical[raw]:mid for raw,mid in retained.items()}
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    catalogues=[[] for _ in FIELDS];catalogue=[];lengths=[]
    for cid in ids:
        original=lookup[cid];mid=mapped[cid];source=fields[mid]
        f=copy.deepcopy(source['fields']);f['title']=original[1].strip()
        year=re.search(r'\((\d{4})\)',f['title'])
        if not year:raise ValueError('retained ML100K title lacks year')
        f['year']=int(year[1]);f['genres']=[genres[i] for i,x in enumerate(original[5:]) if x=='1']
        catalogue.append({'item_id':cid,'ml1m_id':mid,'tmdb_id':source['tmdb_id'],'imdb_id':source['imdb_id'],'fields':f})
        for stage,text in enumerate(field_texts(f)):
            n=len(tokenizer.encode(text,truncation=False))
            if n>2048:raise ValueError('text exceeds common encoder budget')
            catalogues[stage].append({'item_id':cid,'text':text})
            lengths.append({'movie_id':cid,'stage':stage+1,'tokens':n})
    for i,rows in enumerate(catalogues,1):(out/f'catalog-stage-{i:02d}.json').write_text(json.dumps(rows,ensure_ascii=False))
    (out/'catalogue-fields.json').write_text(json.dumps(catalogue,ensure_ascii=False))
    rows,excluded,embargo=split_ratings(clean,seed=42)
    (out/'sequences.json').write_text(json.dumps(rows,separators=(',',':')))
    users=load_temporal_users(out/'sequences.json',ids);hashes={};counts={}
    for split in ['train','validation','test']:
        examples=temporal_examples(users,20,split);hashes[split]=case_digest(examples);counts[split]=len(examples[2])
    report={'dataset':'MovieLens100K','items':len(ids),'users':len(users),'original_items':1682,'original_events':100000,
        'retained_raw_item_ids':len(retained),'retained_events_before_canonical_dedup':int(np.isin(ratings[:,1],list(retained)).sum()),
        'retained_events_before_split':len(clean),'removed_duplicate_user_movie_ratings':duplicate_ratings,
        'train_events':sum(len(u.train) for u in users),'train_examples':counts['train'],'evaluation_examples':counts['validation'],
        'case_sha256':hashes,'excluded_users':excluded,'embargoed_group_events':len(embargo),
        'bridge':{'matched_raw_ids':len(matches),'unmatched_raw_ids':len(unmatched),'ambiguous_raw_ids':len(ambiguous),
                  'method':'explicit IMDb ID when present, otherwise unique exact title/year aliases across GroupLens/TMDB; no fuzzy match or raw-ID reuse'},
        'merge_policy':'same verified external movie -> minimum ML100K item ID; latest user-film rating, source-ID tiebreak at identical times',
        'seed':42,'history_items':20,'timestamp_semantics':'rating submission, not viewing',
        'target_selection':'SHA256(seed,user_id,timestamp,movie_id), independent of rating/file order',
        'training_context':'strictly earlier timestamp groups','test_context':'train plus selected earlier validation item',
        'ratings':'all values1..5 retained','candidate_universe':f'all{len(ids)}retainedmovies; no seen mask or sampled negatives',
        'title_policy':'original canonical ML100K title INCLUDING year; explicit Year stage repeats it','overview_included':False}
    (out/'split-report.json').write_text(json.dumps(report,indent=2));(out/'embargoed-events.json').write_text(json.dumps(embargo))
    mapping={'matches':matches,'unmatched':unmatched,'ambiguous':ambiguous,'canonical_ml100k_ids':canonical}
    (out/'mapping-audit.json').write_text(json.dumps(mapping,indent=2));pd.DataFrame(lengths).to_csv(out/'token-lengths.csv',index=False)
    (out/'preflight.json').write_text(json.dumps({'max_tokens':max(r['tokens'] for r in lengths),'split':report},indent=2))
    (out/'sources.json').write_text(json.dumps([{'path':str(p),'sha256':file_sha256(p)} for p in [args.movielens,args.fields,args.catalogue_audit]],indent=2))
    manifest={'files':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['resources-manifest.json','dataset-metadata.json']]}
    (out/'resources-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'report':report,'manifest_sha256':file_sha256(out/'resources-manifest.json')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--movielens',type=Path,default=Path('/tmp/ml100k/ml-100k.zip'))
    p.add_argument('--catalogue-audit',type=Path,default=Path('/workspace/artifacts/movielens-semantics/catalogue-audit.jsonl.gz'))
    p.add_argument('--fields',type=Path,default=Path('/workspace/artifacts/movielens-text-preflight/catalogue-fields.json.gz'))
    p.add_argument('--tokenizer',type=Path,default=Path('/workspace/scratch/gr-beauty-input/sentence-t5-base'))
    p.add_argument('--output',type=Path,default=Path('/tmp/ml100k/input'))
    a=p.parse_args();prepare(a)
