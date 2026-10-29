"""CPU audit of ML1M + pinned TMDB snapshot; no training and no fuzzy auto-joins."""
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,default=Path('/tmp/rhpr-movie-audit'))
p.add_argument('--output',type=Path,default=Path('/tmp/rhpr-movie-audit/analysis'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)

def csv_rows(name):
    path=a.input/name
    if path.exists():
        with path.open(encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
    else:
        with zipfile.ZipFile(str(path)+'.zip') as z:
            with z.open(name) as b:
                with io.TextIOWrapper(b,encoding='utf-8',newline='') as f:yield from csv.DictReader(f)

def valid(x):return bool(x and x.strip() and x.strip().lower() not in ['nan','none','null'])
def array(x):return ast.literal_eval(x) if valid(x) else []
def norm(x):
    article=re.match(r'^(.*), (The|A|An)$',x)
    if article:x=article.group(2)+' '+article.group(1)
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode().lower())
def title_variants(x):
    values=[x,re.sub(r'\s*\([^()]*\)','',x).strip(),*re.findall(r'\(([^()]*)\)',x)]
    normalized={norm(re.sub(r'^a\.?k\.?a\.?\s*','',v,flags=re.I)) for v in values if v.strip()}
    return normalized-{''}
def quantiles(x):
    x=np.asarray(x);return dict(n=len(x),min=float(x.min()) if len(x) else None,median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,max=float(x.max()) if len(x) else None)

with zipfile.ZipFile(a.input/'ml-1m.zip') as z:
    raw=z.read('ml-1m/movies.dat').decode('latin-1');movies={}
    for line in raw.splitlines():
        i,title,genres=line.split('::');movies[int(i)]=dict(movie_id=int(i),title=title,genres=genres.split('|'))
    ratings=np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::',',')),delimiter=',',dtype=np.int64)
used=set(ratings[:,1].tolist());links={int(r['movieId']):r for r in csv_rows('links.csv')}
meta=collections.defaultdict(list);malformed=0
for row in csv_rows('movies_metadata.csv'):
    if not row['id'].isdigit():malformed+=1;continue
    meta[int(row['id'])].append(row)
meta_dup={k:len(v) for k,v in meta.items() if len(v)>1}
records=[];issues=[];wanted=set()
for i,m in movies.items():
    link=links.get(i);v={**m,'rated':i in used,'link_present':link is not None,'metadata_present':False,'features':{}}
    if link:
        imdb='tt'+link['imdbId'].zfill(7);tmdb=int(float(link['tmdbId'])) if valid(link['tmdbId']) else None
        v.update(imdb_id=imdb,tmdb_id=tmdb)
        rows=meta.get(tmdb,[]);consistent=[r for r in rows if r['imdb_id']==imdb]
        if consistent:
            # Deterministic selection among snapshots of the same confirmed external ID.
            row=max(consistent,key=lambda x:(sum(valid(x.get(k)) for k in ['overview','poster_path','genres','runtime']),len(x.get('overview','')),json.dumps(x,sort_keys=True)))
            v['metadata_present']=True;wanted.add(tmdb)
            ml_year=int(m['title'][-5:-1]) if re.search(r'\(\d{4}\)$',m['title']) else None
            release_year=int(row['release_date'][:4]) if re.match(r'^\d{4}',row['release_date']) else None
            ml_name=re.sub(r' \(\d{4}\)$','',m['title']);same_title=bool(title_variants(ml_name)&(title_variants(row['title'])|title_variants(row['original_title'])))
            v.update(tmdb_title=row['title'],ml_year=ml_year,tmdb_year=release_year,title_normalized_match=same_title,
                     year_difference=release_year-ml_year if release_year and ml_year else None)
            v['features']=dict(overview=valid(row['overview']),poster_path=valid(row['poster_path']),genres=bool(array(row['genres'])),
                runtime=valid(row['runtime']) and float(row['runtime'])>0,language=valid(row['original_language']),
                keywords=False,cast=False,director=False)
            v.update(overview_words=len(row['overview'].split()),poster_path=row['poster_path'],keyword_count=0,cast_count=0,director_count=0)
            if not same_title or (release_year and ml_year and abs(release_year-ml_year)>1):
                issues.append(dict(kind='title_or_year_review',movie_id=i,ml_title=m['title'],tmdb_title=row['title'],original_title=row['original_title'],release_date=row['release_date'],ids_agree=True))
        elif rows:issues.append(dict(kind='conflicting_imdb_tmdb_ids',movie_id=i,expected_imdb=imdb,metadata_imdb=[r['imdb_id'] for r in rows]))
    records.append(v)
keywords={};credits={};duplicates=collections.Counter()
for row in csv_rows('keywords.csv'):
    if row['id'].isdigit() and int(row['id']) in wanted:
        i=int(row['id']);v=array(row['keywords']);duplicates['keywords']+=int(i in keywords)
        keywords[i]=max(keywords.get(i,[]),v,key=len)
for row in csv_rows('credits.csv'):
    if row['id'].isdigit() and int(row['id']) in wanted:
        i=int(row['id']);v=dict(cast=array(row['cast']),crew=array(row['crew']));duplicates['credits']+=int(i in credits)
        if i not in credits or len(v['cast'])+len(v['crew'])>len(credits[i]['cast'])+len(credits[i]['crew']):credits[i]=v
for v in records:
    if not v['metadata_present']:continue
    i=v['tmdb_id'];kw=keywords.get(i,[]);cr=credits.get(i,dict(cast=[],crew=[]))
    directors=[x for x in cr['crew'] if x['job']=='Director']
    v.update(keyword_count=len({x['id'] for x in kw}),cast_count=len({x['id'] for x in cr['cast']}),director_count=len({x['id'] for x in directors}))
    v['features'].update(keywords=bool(kw),cast=bool(cr['cast']),director=bool(directors))

coverage={}
for name,rows in [('catalogue',records),('rated_movies',[v for v in records if v['rated']])]:
    keys=['overview','keywords','cast','director','poster_path','genres','runtime','language']
    coverage[name]=dict(items=len(rows),links=sum(v['link_present'] for v in rows),metadata=sum(v['metadata_present'] for v in rows),
        field_counts={k:sum(v['features'].get(k,False) for v in rows) for k in keys},
        complete_rich_text=sum(all(v['features'].get(k,False) for k in ['overview','keywords','cast','director']) for v in rows),
        title_year_exact_or_plusminus1=sum(v.get('title_normalized_match',False) and abs(v.get('year_difference') or 0)<=1 for v in rows),
        field_count_distributions={k:quantiles([v[k] for v in rows if v['metadata_present']]) for k in ['overview_words','keyword_count','cast_count','director_count']})
record_by_id={v['movie_id']:v for v in records}
event_coverage={k:float(np.mean([record_by_id[i]['features'].get(k,False) for i in ratings[:,1]])) for k in ['overview','keywords','cast','director','poster_path']}
order=np.lexsort((np.arange(len(ratings)),ratings[:,3],ratings[:,0]));ordered=ratings[order]
same=(ordered[1:,0]==ordered[:-1,0])&(ordered[1:,3]==ordered[:-1,3]);tied=np.r_[False,same]|np.r_[same,False]
starts=np.flatnonzero(np.r_[True,~same]);group_lengths=np.diff(np.r_[starts,len(ordered)])
_,timestamp_group_counts=np.unique(ordered[starts,0],return_counts=True)
last_rows=np.flatnonzero(np.r_[ordered[:-1,0]!=ordered[1:,0],True])
assert len(ratings)==1000209 and len(used)==3706 and len(movies)==3883
users,counts_user=np.unique(ratings[:,0],return_counts=True);items,counts_item=np.unique(ratings[:,1],return_counts=True)
utc=lambda x:np.datetime_as_string(np.datetime64(int(x),'s'))+'Z'
summary=dict(dataset=dict(catalogue_items=len(movies),rated_items=len(used),users=len(users),events=len(ratings),
    rating_counts={str(k):int((ratings[:,2]==k).sum()) for k in range(1,6)},history_length=quantiles(counts_user),
    item_frequency=quantiles(counts_item),min_timestamp=utc(ratings[:,3].min()),max_timestamp=utc(ratings[:,3].max()),
    events_in_same_user_same_second_group=int(tied.sum()),fraction_events_in_timestamp_ties=float(tied.mean()),
    adjacent_timestamp_tie_fraction=float(same.sum()/(len(ratings)-len(users)))),coverage=coverage,event_coverage=event_coverage,
    malformed_metadata_rows=malformed,duplicate_metadata_ids=len(meta_dup),duplicate_selected_aux_rows=dict(duplicates),
    join_issue_counts=dict(collections.Counter(x['kind'] for x in issues)),
    important='Coverage uses existing MLmovieId->IMDb/TMDB links and agreement between both external IDs; title/year disagreements remain flagged for review. No fuzzy title joins, no interaction filtering, no feature imputation, no training.')
summary['dataset'].update(distinct_timestamp_groups_per_user=quantiles(timestamp_group_counts),users_with_fewer_than3_timestamp_groups=int((timestamp_group_counts<3).sum()),
    users_last_two_ratings_same_second=int((ordered[last_rows,3]==ordered[last_rows-1,3]).sum()))
inventory=[]
for f in sorted(a.input.iterdir()):
    if f.is_file():inventory.append(dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
with (a.output/'catalogue-audit.jsonl').open('w') as f:
    for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
for name,value in [('summary',summary),('join-issues',issues),('source-files',inventory)]:
    (a.output/(name+'.json')).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
