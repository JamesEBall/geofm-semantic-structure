"""Exact-tie expected multi-positive AP/recall. Author: James Edward Ball."""
import numpy as np

def retrieval_metrics(scores, relevance, ks=(1,5,10)):
    scores=np.asarray(scores,dtype=np.float64);rel=np.asarray(relevance)
    if scores.ndim!=2 or 0 in scores.shape or not np.isfinite(scores).all():
        raise ValueError('Require non-empty finite score matrix')
    if rel.shape!=scores.shape or rel.dtype!=np.bool_:
        raise ValueError('Require Boolean relevance matching scores')
    totals=rel.sum(axis=1)
    if np.any(totals==0):raise ValueError('Query without relevant gallery entry')
    ks=tuple(ks)
    if any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or k<1 for k in ks) or len(set(ks))!=len(ks):
        raise ValueError('Require distinct positive integer cutoffs')
    effective={int(k):min(int(k),scores.shape[1]) for k in ks}
    ap=[];recalls={int(k):[] for k in ks}
    for values,truth,total in zip(scores,rel,totals):
        order=np.argsort(-values,kind='stable');v=values[order];y=truth[order]
        starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];ends=np.r_[starts[1:],len(v)]
        numerator=0.;positives=0
        counts={k:0. for k in effective}
        for begin,end in zip(starts,ends):
            n=int(end-begin);r=int(y[begin:end].sum())
            if r:
                j=np.arange(1,n+1,dtype=np.float64)
                previous=(j-1)*(r-1)/(n-1) if n>1 else np.zeros(1)
                numerator+=float(np.sum((r/n)*(positives+1+previous)/(begin+j)))
            for k,cutoff in effective.items():counts[k]+=max(0,min(n,cutoff-int(begin)))*r/n
            positives+=r
        ap.append(numerator/int(total))
        for k in recalls:recalls[k].append(counts[k]/int(total))
    return {'per_query_ap':np.asarray(ap),'map':float(np.mean(ap)),
            'per_query_recall':{k:np.asarray(v) for k,v in recalls.items()},
            'recall':{k:float(np.mean(v)) for k,v in recalls.items()},
            'effective_k':effective,'queries':scores.shape[0],'gallery_size':scores.shape[1]}
