"""Weighted DeepWalk of person-film coappearance, using gensim SGNS on CPU."""
import argparse,collections,gzip,json,os,random,time
from pathlib import Path
import numpy as np


class PeopleWalks:
    def __init__(self, movies, seed=42, walks=10, length=40):
        self.movies=movies;self.seed=seed;self.walks=walks;self.length=length
        self.incident=collections.defaultdict(list)
        for i,people in enumerate(movies):
            for p in people:self.incident[p].append(i)
        self.people=sorted(self.incident)
    def __iter__(self):
        gen=random.Random(self.seed)
        for _ in range(self.walks):
            order=self.people.copy();gen.shuffle(order)
            for p in order:
                path=[str(p)]
                for _ in range(self.length-1):
                    movie=gen.choice(self.incident[p]);vs=self.movies[movie]
                    if len(vs)>1:
                        j=gen.randrange(len(vs)-1);offset=vs.index(p);p=vs[j+(j>=offset)]
                    path.append(str(p))
                yield path


def prepare(catalogue,output):
    from gensim.models import Word2Vec
    if os.environ.get('PYTHONHASHSEED')!='42':raise ValueError('run with PYTHONHASHSEED=42 for reproducible SGNS initialization')
    rows=json.loads(gzip.decompress(catalogue.read_bytes()))
    movies=[sorted({x['person_id'] for x in r['cast']+r['directors']}) for r in rows]
    if any(not people for people in movies):raise ValueError('film without people')
    walks=PeopleWalks(movies);before=time.monotonic()
    model=Word2Vec(sentences=walks,vector_size=64,window=5,min_count=1,sg=1,negative=5,
        sample=0,workers=1,seed=42,epochs=3)
    output.mkdir(parents=True,exist_ok=True)
    pooled=[]
    for row in rows:
        pooled.append(np.concatenate([np.stack([model.wv[str(p['person_id'])] for p in row[field]]).mean(0)
            for field in ['cast','directors']]))
    np.save(output/'people.npy',np.asarray(pooled,dtype=np.float32),allow_pickle=False)
    np.savez_compressed(output/'people-vectors.npz',person_ids=np.array(walks.people),vectors=np.stack([model.wv[str(p)] for p in walks.people]))
    (output/'source-ids.json').write_text(json.dumps([r['item_id'] for r in rows])+'\n')
    report={'method':'weighted DeepWalk, SGNS via gensim4.4.0','walk_transition':'uniform incident film, then uniform other participant; W_ij=sum1/(n_film-1)',
        'people':len(walks.people),'movies':len(rows),'person_dimension':64,'item_dimension':128,
        'role_pooling':'separate cast/director means, one shared person table','seed':42,'walks_per_node':10,
        'walk_length':40,'window':5,'negative':5,'sgns_epochs':3,'workers':1,'sample':0,
        'no_interactions_used':True,'catalogue_scope':'same1273films','seconds':time.monotonic()-before}
    (output/'people-provenance.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.catalogue,a.output)
