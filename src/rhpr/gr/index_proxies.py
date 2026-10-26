"""Exploratory first-prefix diagnostics using catalogue metadata and TRAIN only."""
import numpy as np
from scipy import sparse


class IndexProxies:
    """Fixed user folds and degree-normalized binary item-user incidence.

    Gain is the cross-user last-item count model's improvement over its own
    smoothed prefix unigram, not a neural retrieval score. No validation/test
    event, timestamp or user target is consulted. Price R² is catalogue in-sample
    explained variance and is sensitive to partition granularity.
    """
    def __init__(self, users, price, observed):
        self.price=np.asarray(price,dtype=np.float64)
        self.observed=np.asarray(observed,dtype=bool)
        self.n=len(self.price)
        if self.price.ndim!=1 or self.observed.shape!=self.price.shape or not np.isfinite(self.price[self.observed]).all():
            raise ValueError('invalid catalogue price arrays')
        self.trim=self.observed.copy()
        if self.observed.any():self.trim &= self.price<=np.quantile(self.price[self.observed],.99)
        src=[];dst=[];fold=[];rr=[];cc=[]
        for i,u in enumerate(users):
            train=np.asarray(u.train,dtype=np.int64)
            if np.any((train<0)|(train>=self.n)):raise ValueError('TRAIN item outside catalogue')
            src.extend(train[:-1]);dst.extend(train[1:]);fold.extend([i%3]*(len(train)-1))
            unique=np.unique(train);rr.extend(unique);cc.extend([i]*len(unique))
        self.src=np.asarray(src,dtype=np.int64);self.dst=np.asarray(dst,dtype=np.int64);self.fold=np.asarray(fold,dtype=np.int64)
        x=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(self.n,len(users)))
        self.frequency=np.asarray(x.sum(1)).ravel()
        self.x=sparse.diags(1/np.sqrt(np.maximum(self.frequency,1)))@x

    def first(self,codes):
        codes=np.asarray(codes)
        if codes.shape!=(self.n,) or codes.dtype.kind not in 'iu':raise ValueError('one integer first-code assignment per catalogue item required')
        _,labels=np.unique(codes,return_inverse=True);k=int(labels.max())+1
        def r2(mask):
            y=self.price[mask]
            if len(y)<2:return None
            total=np.sum((y-y.mean())**2)
            if total==0:return None
            a=labels[mask];mass=np.bincount(a,minlength=k);sums=np.bincount(a,weights=y,minlength=k)
            return float(1-np.sum((y-(sums/np.maximum(mass,1))[a])**2)/total)
        m=sparse.csr_matrix((np.ones(self.n),(labels,np.arange(self.n))),shape=(k,self.n))
        v=m@self.x;cnt=np.bincount(labels,weights=self.frequency>0,minlength=k);den=float(np.sum(cnt*(cnt-1)))
        coherence=float((v.multiply(v).sum()-np.sum(self.frequency>0))/den) if den else None
        y=labels[self.dst];gains=[]
        for f in range(3):
            train=self.fold!=f;test=~train
            if not train.any() or not test.any():continue
            prior=(np.bincount(y[train],minlength=k)+.5)/(train.sum()+.5*k)
            keys,mass=np.unique(self.src[train]*k+y[train],return_counts=True)
            q=self.src[test]*k+y[test];pos=np.searchsorted(keys,q);valid=pos<len(keys)
            valid[valid]=keys[pos[valid]]==q[valid]
            matched=np.zeros(len(q));matched[valid]=mass[pos[valid]]
            total=np.bincount(self.src[train],minlength=self.n)
            p=(matched+10*prior[y[test]])/(total[self.src[test]]+10)
            gains.append(np.log2(p)-np.log2(prior[y[test]]))
        return dict(price_r2=r2(self.observed),price_r2_trim99=r2(self.trim),
            train_proxy_gain_bits=float(np.concatenate(gains).mean()) if gains else None,
            train_prefix_coherence=coherence,first_codes_used=k)
