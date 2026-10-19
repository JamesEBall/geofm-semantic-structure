"""Fit-only preprocessing and ridge readout. Author: James Edward Ball."""
from dataclasses import dataclass
import numpy as np

def matrix(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or 0 in a.shape or not np.isfinite(a).all():raise ValueError('Require non-empty finite matrix')
    return a

def owned_readonly(a):
    a=np.array(a,dtype=np.float64,copy=True);a.flags.writeable=False;return a

@dataclass(frozen=True)
class RidgeMap:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    regularisation: float
    fit_rows: int

    def predict(self,x):
        x=matrix(x)
        if x.shape[1]!=len(self.x_mean):raise ValueError('Feature width mismatch')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            out=((x-self.x_mean)/self.x_scale)@self.coefficients+self.y_mean
        if not np.isfinite(out).all():raise ValueError('Non-finite prediction')
        return out

def fit_ridge(x,y,regularisation):
    x=matrix(x);y=matrix(y)
    if len(x)!=len(y) or len(x)<2:raise ValueError('Require matching fit rows and at least two samples')
    if isinstance(regularisation,(bool,np.bool_)) or not np.isscalar(regularisation):raise ValueError('Require positive finite lambda')
    lam=float(regularisation)
    if not np.isfinite(lam) or lam<=0:raise ValueError('Require positive finite lambda')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        mean=x.mean(0);scale=x.std(0,ddof=0);scale=np.where(scale==0,1.,scale);ym=y.mean(0)
        u,s,vt=np.linalg.svd((x-mean)/scale,full_matrices=False)
        coef=(vt.T*(s/(s*s+lam)))@(u.T@(y-ym))
    if not np.isfinite(coef).all():raise ValueError('Non-finite fit')
    return RidgeMap(*(owned_readonly(a) for a in [mean,scale,ym,coef]),lam,len(x))
