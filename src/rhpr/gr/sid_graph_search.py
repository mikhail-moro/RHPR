"""TRAIN-only ALS signal and confidence-guided fixed-tree local search."""
from __future__ import annotations
import hashlib
import time
import numpy as np
from scipy import sparse
from rhpr.gr.sid_graph_swap import PairTerm,SwapSearch,lcp_similarity


def fit_als_signal(users, items, spec):
    """Use implicit's CPU ALS; rows are users, columns are catalogue items."""
    import implicit
    from implicit.cpu.als import AlternatingLeastSquares
    row,col=[],[]
    for uid,user in enumerate(users):
        ids=np.unique(user.train)
        if np.any(ids<0) or np.any(ids>=items):raise ValueError('TRAIN item outside catalogue')
        row.extend([uid]*len(ids));col.extend(ids)
    matrix=sparse.csr_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(len(users),items))
    frequency=np.asarray((matrix>0).sum(0)).ravel()
    before=time.monotonic()
    model=AlternatingLeastSquares(factors=spec['factors'],regularization=spec['regularization'],
        alpha=spec['alpha'],iterations=spec['iterations'],random_state=spec['seed'],
        num_threads=spec['cpu_threads'],dtype=np.float32)
    model.fit(matrix,show_progress=False)
    factors=np.asarray(model.item_factors,dtype=np.float32)
    if factors.shape!=(items,spec['factors']) or not np.isfinite(factors).all():
        raise RuntimeError('ALS returned invalid item factors')
    digest=hashlib.sha256()
    for a in [matrix.indptr,matrix.indices,matrix.data]:digest.update(a.tobytes())
    report=dict(implementation='implicit.cpu.als.AlternatingLeastSquares',version=implicit.__version__,
        spec=spec,user_item_shape=list(matrix.shape),binary_interactions=matrix.nnz,
        input_csr_sha256=digest.hexdigest(),seconds=time.monotonic()-before,
        train_only=True,item_factor_sha256=hashlib.sha256(factors.tobytes()).hexdigest())
    return factors,np.asarray(model.user_factors),frequency,report


class PriorityProposals:
    """High error * support anchors; semantic-near occupants of a CF friend's branch.

    A friend is not itself the swap partner: exchanging that pair's paths would
    leave their mutual LCP unchanged. The full objective still decides acceptance.
    """
    def __init__(self,graph,embedding,frequency,spec,seed):
        self.g=graph;self.spec=spec;self.rng=np.random.default_rng(seed)
        self.z=np.asarray(embedding,dtype=np.float32)
        self.z=self.z/np.maximum(np.linalg.norm(self.z,axis=1,keepdims=True),1e-30)
        self.frequency=np.asarray(frequency)
        order=np.argsort(graph.source,kind='stable');self.targets=graph.target[order];self.weights=graph.weight[order]
        self.indptr=np.r_[0,np.cumsum(np.bincount(graph.source,minlength=graph.items))]
        self.mass=np.bincount(graph.source,weights=graph.weight,minlength=graph.items)
        self.anchors=None;self.prob=None;self.uniform=0;self.guided=0

    def refresh(self,codes):
        error=np.bincount(self.g.source,weights=self.g.weight*(1-lcp_similarity(codes,self.g.source,self.g.target)),minlength=self.g.items)
        error=np.divide(error,self.mass,out=np.zeros_like(error),where=self.mass>0)
        confidence=self.frequency/(self.frequency+self.spec['support_shrinkage'])
        priority=error*confidence
        eligible=np.flatnonzero((self.mass>0)&(priority>0))
        if not len(eligible):self.anchors=np.array([],dtype=int);return
        tie=self.rng.random(len(eligible));rank=np.lexsort((tie,-priority[eligible]))
        take=max(1,int(np.ceil(len(eligible)*self.spec['anchor_fraction'])))
        self.anchors=eligible[rank[:take]];self.prob=priority[self.anchors]/priority[self.anchors].sum()

    def pair(self,codes,uniform_pair):
        if self.anchors is None:self.refresh(codes)
        if not len(self.anchors) or self.rng.random()<self.spec['uniform_fraction']:
            self.uniform+=1;return tuple(uniform_pair)
        a=int(self.rng.choice(self.anchors,p=self.prob))
        lo,hi=self.indptr[a:a+2];neighbours=self.targets[lo:hi];weight=self.weights[lo:hi]
        valid=codes[neighbours,0]!=codes[a,0];neighbours=neighbours[valid];weight=weight[valid]
        if not len(neighbours):self.uniform+=1;return tuple(uniform_pair)
        friend=int(self.rng.choice(neighbours,p=weight/weight.sum()))
        pool=np.flatnonzero(codes[:,0]==codes[friend,0]);pool=pool[(pool!=a)&(pool!=friend)]
        if not len(pool):self.uniform+=1;return tuple(uniform_pair)
        similarity=self.z[pool]@self.z[a]
        order=np.lexsort((pool,-similarity))[:self.spec['semantic_partner_pool']]
        b=int(self.rng.choice(pool[order]));self.guided+=1
        return a,b


def budgeted_search(initial,cf,semantic,embedding,frequency,spec,mode):
    if mode not in ['uniform','priority']:raise ValueError('unknown proposal mode')
    search=SwapSearch(initial,[PairTerm(cf),PairTerm(semantic)],[1.,spec['semantic_lambda']])
    pairs=np.random.default_rng(spec['uniform_seed']).integers(len(initial),size=(spec['max_proposals'],2))
    same=pairs[:,0]==pairs[:,1];pairs[same,1]=(pairs[same,1]+1)%len(initial)
    proposal=PriorityProposals(cf,embedding,frequency,spec['priority'],spec['priority_seed']) if mode=='priority' else None
    moves=[];snapshots=[(0,0,search.codes.copy(),search.owner.copy())];started=time.monotonic()
    for position,pair in enumerate(pairs,1):
        if proposal is not None:
            if (position-1)%spec['priority']['refresh_proposals']==0:proposal.refresh(search.codes)
            a,b=proposal.pair(search.codes,pair)
        else:a,b=pair
        accepted,delta=search.try_swap(a,b)
        if accepted:
            moves.append([position,int(a),int(b),delta])
            if search.accepted in spec['snapshot_accepted']:
                search.audit(initial);snapshots.append((position,search.accepted,search.codes.copy(),search.owner.copy()))
        if search.accepted>=spec['accepted_cap']:break
    search.audit(initial)
    if snapshots[-1][1]!=search.accepted:snapshots.append((position,search.accepted,search.codes.copy(),search.owner.copy()))
    report=dict(mode=mode,accepted=search.accepted,proposals=position,
        moved_items=int(np.count_nonzero(search.owner!=np.arange(len(initial)))),
        objective=search.score,cf_score=float(search.totals[0]),semantic_score=float(search.totals[1]),
        seconds=time.monotonic()-started,uniform_proposals=position if proposal is None else proposal.uniform,
        guided_proposals=0 if proposal is None else proposal.guided)
    return search.codes,search.owner,np.asarray(moves).reshape(-1,4),snapshots,report


def owner_after_moves(items,moves,budget):
    owner=np.arange(items)
    for _,a,b,_ in moves[:budget]:owner[[int(a),int(b)]]=owner[[int(b),int(a)]]
    return owner


def randomize_changed_paths(priority_owner,seed):
    """Same changed items and available paths as priority; random assignment."""
    initial=np.arange(len(priority_owner));changed=np.flatnonzero(priority_owner!=initial)
    if len(changed)<3:raise ValueError('insufficient changed items for a distinct random control')
    rng=np.random.default_rng(seed)
    for _ in range(1000):
        owner=initial.copy();owner[changed]=rng.permutation(changed)
        if np.all(owner[changed]!=changed) and not np.array_equal(owner,priority_owner):return owner
    raise RuntimeError('cannot generate changed-set matched random derangement')
