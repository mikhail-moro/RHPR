"""Saved-score sensitivity where the actual visible movie list is unchanged."""
import csv,gzip,json
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import load_temporal_users,temporal_examples
O=Path('/tmp/context50-split-gap');P=Path('/tmp/movie-actor-full-completed');ids=json.loads((P/'trials/none_seed42/sid-catalogue.json').read_text())['original_item_ids'];users=load_temporal_users('/tmp/context50-split-gap-inputs/sequences.json',ids);N=len(users);ix={u.user_id:j for j,u in enumerate(users)};scores={};rows=[]
cat=json.loads(gzip.decompress(Path('/tmp/context50-split-gap-inputs/catalogue-fields.json.gz').read_bytes()))
with gzip.open('/tmp/context50-three-seed-analysis/cases.csv.gz','rt')as f:
 for r in csv.DictReader(f):
  h=int(r['history_items'])
  if h not in [40,50]:continue
  k=h,r['split'],r['candidate'],int(r['seed']);scores.setdefault(k,np.zeros(N))[ix[int(r['user_id'])]]=float(r['ndcg10'])
for split in ['validation','test']:
 a=temporal_examples(users,40,split);b=temporal_examples(users,50,split);same=np.array([np.array_equal(x[x>=0],y[y>=0])for x,y in zip(a[0],b[0])]);comedy=np.array(['Comedy'in cat[t]['fields']['genres']for t in a[2]])
 for subset,m in [('same_visible_movies',same),('additional_old_movies',~same),('Comedy_same_movies',same&comedy),('Comedy_added_movies',(~same)&comedy)]:
  record=dict(split=split,subset=subset,n=int(m.sum()))
  for h in [40,50]:
   base=np.stack([scores[h,split,'none',s]for s in [42,101,202]]);g=np.stack([scores[h,split,'actor_w01',s]for s in [42,101,202]])
   record[f'L{h}']=dict(control=float(base[:,m].mean()),graph=float(g[:,m].mean()),relative=float(g[:,m].mean()/base[:,m].mean()-1),delta_by_seed=(g-base)[:,m].mean(1).tolist())
  rows.append(record)
r=dict(rows=rows,scope='Same unpadded movie IDs for these users. Different fitted models and maximum padding width remain; this does not isolate padding or training causally.')
(O/'same-visible-history.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

# Follow-up genre checks use the same users/targets as the original windows.
genres=[]
for h in [40,50]:
 for split in ['validation','test']:
  ex=temporal_examples(users,h,split)
  for genre in ['Comedy','Mystery']:
   m=np.array([genre in cat[t]['fields']['genres']for t in ex[2]]);b=np.stack([scores[h,split,'none',ss]for ss in [42,101,202]]);g=np.stack([scores[h,split,'actor_w01',ss]for ss in [42,101,202]])
   genres.append(dict(history_items=h,split=split,genre=genre,n=int(m.sum()),control=float(b[:,m].mean()),graph=float(g[:,m].mean()),delta_by_seed=(g-b)[:,m].mean(1).tolist()))
(O/'genre-window-robustness.json').write_text(json.dumps(genres,indent=2)+'\n')
