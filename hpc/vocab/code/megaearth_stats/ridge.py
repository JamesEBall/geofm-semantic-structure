"""Exact ridge readout from float64 sufficient statistics.

A fit on window set E uses, on data shifted by a reference mean m0 (cancellation control):
  n, s = Σ x', Q = Σ x'x'ᵀ, per-bundle sums Sb (bundle = unique label set) and counts nb.
Then μ = m0 + s/n, Cov = Q − s sᵀ/n, σ = sqrt(diag(Cov)/(n−1)) (σ = 1 for zero-variance columns),
G = D⁻¹ Cov D⁻¹ (= X_sᵀX_s), cross C = D⁻¹(Σ_w x'_w t_wᵀ − s t̄ᵀ), and for every alpha
W_α = U diag(1/(λ+α)) Uᵀ C from one eigh(G). Prediction p(x) = t̄ + ((x−μ)/σ) W_α.
Statistics are additive over windows, so exclusions are exact: stats(E) = stats(E0) − stats(E0∖E).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


@dataclass
class Stats:
    n: int
    s: np.ndarray
    Q: np.ndarray
    Sb: np.ndarray
    nb: np.ndarray

    def __sub__(self, o: "Stats") -> "Stats":
        return Stats(self.n - o.n, self.s - o.s, self.Q - o.Q, self.Sb - o.Sb, self.nb - o.nb)

    def with_bundles(self, Sb: np.ndarray, nb: np.ndarray) -> "Stats":
        return Stats(self.n, self.s, self.Q, Sb, nb)


def accumulate(get_rows, rows: np.ndarray, bundle: np.ndarray, n_bundles: int, m0: np.ndarray,
               chunk: int = 16384, gram: bool = True) -> Stats:
    """Sufficient statistics over `rows` (global window ids) with bundle ids `bundle` (aligned)."""
    rows = np.asarray(rows, dtype=np.int64)
    bundle = np.asarray(bundle, dtype=np.int64)
    D = len(m0)
    s = np.zeros(D)
    Q = np.zeros((D, D)) if gram else None
    Sb = np.zeros((n_bundles, D))
    for a in range(0, len(rows), chunk):
        B = get_rows(rows[a:a + chunk]).astype(np.float64)
        B -= m0
        s += B.sum(0)
        if gram:
            Q += B.T @ B
        b = bundle[a:a + chunk]
        ind = sp.csr_matrix((np.ones(len(b)), (b, np.arange(len(b)))), shape=(n_bundles, len(b)))
        Sb += ind @ B
    nb = np.bincount(bundle, minlength=n_bundles).astype(np.float64)
    return Stats(len(rows), s, Q, Sb, nb)


def stats_for(get_rows, base: Stats | None, base_rows: np.ndarray | None, rows: np.ndarray,
              bundle_of_window: np.ndarray, n_bundles: int, m0: np.ndarray, chunk: int = 16384) -> Stats:
    """Stats over `rows`; when `rows` ⊆ base_rows and the complement is smaller, subtract it from base."""
    rows = np.asarray(rows, dtype=np.int64)
    if base is not None and base_rows is not None:
        removed = np.setdiff1d(base_rows, rows, assume_unique=True)
        if len(removed) + len(rows) == len(base_rows) and len(removed) < len(rows):
            if len(removed) == 0:
                return base
            return base - accumulate(get_rows, removed, bundle_of_window[removed], n_bundles, m0, chunk)
    return accumulate(get_rows, rows, bundle_of_window[rows], n_bundles, m0, chunk)


@dataclass
class Spectrum:
    n: int
    mu: np.ndarray
    sd: np.ndarray
    zero: np.ndarray
    lam: np.ndarray       # descending, clipped at 0
    U: np.ndarray
    s: np.ndarray         # Σ x' (shifted)
    m0: np.ndarray

    def project(self, Xq: np.ndarray) -> np.ndarray:
        Z = (np.asarray(Xq, dtype=np.float64) - self.mu) / self.sd
        Z[:, self.zero] = 0.0
        return Z @ self.U

    def eff_rank(self, alpha: float) -> float:
        return float(np.sum(self.lam / (self.lam + alpha)))


def fit_spectrum(st: Stats, m0: np.ndarray, rel_zero: float = 1e-12) -> Spectrum:
    if st.n < 2:
        raise ValueError(f"ridge fit needs >= 2 windows, got {st.n}")
    s = st.s
    Cov = st.Q - np.outer(s, s) / st.n
    Cov = 0.5 * (Cov + Cov.T)
    d = np.diag(Cov).copy()
    q = np.diag(st.Q)
    zero = (d <= rel_zero * np.maximum(q, 1e-300)) | (d <= 0)
    sd = np.sqrt(np.where(zero, 1.0, d / (st.n - 1)))
    G = Cov / np.outer(sd, sd)
    G[zero, :] = 0.0
    G[:, zero] = 0.0
    lam, U = sla.eigh(G, driver="evd", overwrite_a=True, check_finite=False)
    lam, U = lam[::-1].copy(), np.ascontiguousarray(U[:, ::-1])
    lam = np.clip(lam, 0.0, None)
    return Spectrum(n=st.n, mu=m0 + s / st.n, sd=sd, zero=zero, lam=lam, U=U, s=s, m0=m0)


def bundle_weights(bundles_csr, P: np.ndarray) -> sp.csr_matrix:
    """(n_bundles × K) weights so that T_b = (Bw @ P)[b] = unit(mean of P over the bundle's classes)."""
    indptr, indices = bundles_csr.indptr, bundles_csr.indices
    nbund = len(indptr) - 1
    lens = np.diff(indptr)
    rid = np.repeat(np.arange(nbund), lens)
    M = sp.csr_matrix((1.0 / lens[rid], (rid, indices)), shape=(nbund, P.shape[0]))
    mean = M @ P
    nrm = np.linalg.norm(mean, axis=1)
    inv = np.where(nrm > 1e-12, 1.0 / np.maximum(nrm, 1e-300), 0.0)
    return sp.csr_matrix(sp.diags(inv) @ M)


@dataclass
class Cross:
    C: np.ndarray        # D × d (standardised cross term)
    tbar: np.ndarray     # d


def cross_term(spec: Spectrum, st: Stats, Bw: sp.csr_matrix, P: np.ndarray) -> Cross:
    Sc = Bw.T @ st.Sb                          # K × D: Σ_w a_wc x'_w
    XtT = np.asarray(Sc).T @ P                 # D × d: Σ_w x'_w t_wᵀ
    tbar = np.asarray(Bw.T @ st.nb).ravel() @ P / st.n
    C = (XtT - np.outer(st.s, tbar)) / spec.sd[:, None]
    C[spec.zero] = 0.0
    return Cross(C=C, tbar=tbar)


def gallery_dot_scores(spec: Spectrum, ZU: np.ndarray, cr: Cross, Gal: np.ndarray, alphas, rank: int | None = None) -> list[np.ndarray]:
    """Dot-product scores p(x)·g for each alpha (ranking-equivalent to cosine for unit g)."""
    r = spec.U.shape[1] if rank is None else min(rank, spec.U.shape[1])
    UtCG = spec.U[:, :r].T @ (cr.C @ Gal.T)    # r × k
    off = cr.tbar @ Gal.T
    ZUr = ZU[:, :r]
    return [off + ZUr @ (UtCG / (spec.lam[:r, None] + a)) for a in alphas]


def predict(spec: Spectrum, ZU: np.ndarray, cr: Cross, alpha: float, rank: int | None = None) -> np.ndarray:
    r = spec.U.shape[1] if rank is None else min(rank, spec.U.shape[1])
    UtC = spec.U[:, :r].T @ cr.C
    return cr.tbar + ZU[:, :r] @ (UtC / (spec.lam[:r, None] + alpha))


def weights(spec: Spectrum, cr: Cross, alpha: float, rank: int | None = None) -> np.ndarray:
    """W (D × d) in standardised units: p(x) = t̄ + ((x−μ)/σ) W."""
    r = spec.U.shape[1] if rank is None else min(rank, spec.U.shape[1])
    return spec.U[:, :r] @ ((spec.U[:, :r].T @ cr.C) / (spec.lam[:r, None] + alpha))


def cosine_scores(Pred: np.ndarray, Gal: np.ndarray) -> np.ndarray:
    """cos(p, g) for unit-norm gallery rows; float32 output (the stored, ranked representation)."""
    nrm = np.linalg.norm(Pred, axis=1, keepdims=True)
    out = (Pred @ Gal.T) / np.maximum(nrm, 1e-300)
    out[nrm[:, 0] <= 1e-300] = 0.0
    return out.astype(np.float32)


def direct_ridge(X: np.ndarray, T: np.ndarray, alpha: float):
    """Reference: explicit train standardisation (ddof=1) and solve. Returns (mu, sd, tbar, W)."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    var = (Xc ** 2).sum(0) / (len(X) - 1)
    zero = var <= 1e-24 * np.maximum((X ** 2).mean(0), 1e-300)
    sd = np.sqrt(np.where(zero, 1.0, var))
    Xs = Xc / sd
    Xs[:, zero] = 0.0
    tbar = T.mean(0)
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ (T - tbar))
    return mu, sd, tbar, W
