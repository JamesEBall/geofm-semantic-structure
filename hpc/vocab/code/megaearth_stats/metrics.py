"""Ranking metrics with exact tie conventions, chance levels, aggregation and GZSL gamma sweeps.

Conventions (TEST-PLAN §3):
* RR of a target with a = #candidates scoring strictly higher and t = #candidates tied with it
  (target included) is the expectation under uniform random tie-breaking: (H[a+t] - H[a]) / t.
* top-k is likewise expected: clip(k - a, 0, t) / t.
* Filtered ranking: the other true labels of the window are removed from that pair's candidates.
* Chance for a pair with k' candidates is H[k'] / k'.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import gammaln

_H = np.zeros(1, dtype=np.float64)


def harmonic(n: int) -> np.ndarray:
    """H[0..n] with H[0] = 0 (cached, grows on demand)."""
    global _H
    if len(_H) <= n:
        m = max(int(n) + 1, 2 * len(_H))
        _H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, m, dtype=np.float64))])
    return _H


def rr_from_counts(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    if np.any(t < 1):
        raise ValueError("tie count must include the target (t >= 1)")
    H = harmonic(int((a + t).max(initial=1)))
    return (H[a + t] - H[a]) / t


def topk_from_counts(a: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return np.clip(k - a, 0.0, t) / t


def chance_rr(kprime: np.ndarray) -> np.ndarray:
    kprime = np.asarray(kprime, dtype=np.int64)
    H = harmonic(int(kprime.max(initial=1)))
    return H[kprime] / kprime


def chance_topk(kprime: np.ndarray, k: int) -> np.ndarray:
    kprime = np.asarray(kprime, dtype=np.float64)
    return np.minimum(k, kprime) / kprime


def rank_pairs(S: np.ndarray, prow: np.ndarray, pcol: np.ndarray, true_mask: np.ndarray | None = None,
               allowed: np.ndarray | None = None, chunk: int = 8192) -> dict[str, np.ndarray]:
    """Tie-aware filtered ranking of pairs (query row prow[i], target column pcol[i]) in score matrix S.

    true_mask[r, j]: column j is a true label of query r (removed from competitors, except the target).
    allowed[r, j]: optional per-query gallery restriction (the target must be allowed).
    """
    S = np.asarray(S)
    prow = np.asarray(prow, dtype=np.int64)
    pcol = np.asarray(pcol, dtype=np.int64)
    n = len(prow)
    a = np.zeros(n, dtype=np.int64)
    t = np.zeros(n, dtype=np.int64)
    kp = np.zeros(n, dtype=np.int64)
    if n and not np.all(np.isfinite(S[prow])):
        raise ValueError("non-finite scores")
    for s0 in range(0, n, chunk):
        r = prow[s0:s0 + chunk]
        c = pcol[s0:s0 + chunk]
        m = len(r)
        rows = S[r]
        own = rows[np.arange(m), c]
        if allowed is not None:
            cand = allowed[r].copy()
            if not np.all(cand[np.arange(m), c]):
                raise ValueError("target class excluded by the gallery restriction")
        else:
            cand = np.ones(rows.shape, dtype=bool)
        if true_mask is not None:
            cand &= ~true_mask[r]
        cand[np.arange(m), c] = True
        a[s0:s0 + m] = np.count_nonzero((rows > own[:, None]) & cand, axis=1)
        t[s0:s0 + m] = np.count_nonzero((rows == own[:, None]) & cand, axis=1)
        kp[s0:s0 + m] = np.count_nonzero(cand, axis=1)
    return {"rr": rr_from_counts(a, t) if n else np.zeros(0), "top1": topk_from_counts(a, t, 1),
            "top5": topk_from_counts(a, t, 5), "kprime": kp, "chance": chance_rr(kp) if n else np.zeros(0),
            "a": a, "t": t}


# --------------------------------------------------------------------------- multi-label (query) metrics
@lru_cache(maxsize=None)
def first_relevant_chance(k: int, R: int) -> float:
    """E[1/rank of first relevant] for R relevant among k candidates in uniformly random order."""
    if R < 1 or R > k:
        raise ValueError((k, R))
    j = np.arange(1, k - R + 2, dtype=np.float64)
    logp = (gammaln(k - j + 1) - gammaln(R) - gammaln(k - j - R + 2)) - (gammaln(k + 1) - gammaln(R + 1) - gammaln(k - R + 1))
    return float(np.sum(np.exp(logp) / j))


def ap_chance(k: np.ndarray, R: np.ndarray) -> np.ndarray:
    """E[AP] for R relevant among k candidates in random order: (H_k + (R-1)/(k-1)(k - H_k)) / k."""
    k = np.asarray(k, dtype=np.int64)
    R = np.asarray(R, dtype=np.float64)
    H = harmonic(int(k.max(initial=1)))[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (H + np.where(k > 1, (R - 1.0) / np.maximum(k - 1, 1), 0.0) * (k - H)) / k
    return out


def _first_block_rr(a: int, t: int, r: int) -> float:
    if t == 1 or r == t:
        return 1.0 / (a + 1)
    j = np.arange(1, t - r + 2, dtype=np.float64)
    logp = (gammaln(t - j + 1) - gammaln(r) - gammaln(t - j - r + 2)) - (gammaln(t + 1) - gammaln(r + 1) - gammaln(t - r + 1))
    return float(np.sum(np.exp(logp) / (a + j)))


def query_metrics(S: np.ndarray, rel: np.ndarray) -> dict[str, np.ndarray]:
    """Per query row: tie-aware expected first-relevant RR and AP over all columns, with chance.

    S (n, k) scores; rel (n, k) bool relevance; every row must have >= 1 relevant column.
    """
    S = np.asarray(S)
    rel = np.asarray(rel, dtype=bool)
    n, k = S.shape
    R = rel.sum(1)
    if n == 0:
        z = np.zeros(0)
        return {"fr": z, "ap": z, "fr_chance": z, "ap_chance": z, "R": R, "k": np.zeros(0, np.int64)}
    if np.any(R < 1):
        raise ValueError("every query needs >= 1 relevant item")
    order = np.argsort(-S.astype(np.float64), axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, 1)
    Rs = np.take_along_axis(rel, order, 1)
    newb = np.ones((n, k), dtype=bool)
    newb[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
    g = np.cumsum(newb.ravel()) - 1
    nb = int(g[-1]) + 1
    t = np.bincount(g, minlength=nb).astype(np.int64)
    r = np.bincount(g, weights=Rs.ravel().astype(np.float64), minlength=nb).astype(np.int64)
    pos = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k))
    a = pos[newb]                                   # items above the block (within its row)
    brow = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k))[newb]
    rcum = np.cumsum(r)
    first_of_row = np.flatnonzero(a == 0)
    before_row = (rcum - r)[first_of_row][brow]
    R_above = rcum - r - before_row
    H = harmonic(k + 1)
    dH = H[a + t] - H[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        multi = (r / t) * ((R_above + 1) * dH + np.where(t > 1, (r - 1) / np.maximum(t - 1, 1), 0.0) * (t - (a + 1) * dH))
    single = r * (R_above + 1) / (a + 1.0)
    contrib = np.where(r == 0, 0.0, np.where(t > 1, multi, single))
    ap = np.bincount(brow, weights=contrib, minlength=n) / R
    hit = np.flatnonzero(r > 0)
    _, first = np.unique(brow[hit], return_index=True)
    fb = hit[first]
    fr = np.empty(n)
    for i, b in enumerate(fb):
        fr[i] = _first_block_rr(int(a[b]), int(t[b]), int(r[b]))
    kk = np.full(n, k, dtype=np.int64)
    frc = np.asarray([first_relevant_chance(k, int(x)) for x in R])
    return {"fr": fr, "ap": ap, "fr_chance": frc, "ap_chance": ap_chance(kk, R), "R": R, "k": kk}


# --------------------------------------------------------------------------- aggregation
@dataclass
class Aggregator:
    """Window -> cell mean within class -> mean over cells within class -> mean over classes.

    `cls`, `cell` are per-pair arrays (a window contributes once per class).
    """
    cls: np.ndarray
    cell: np.ndarray

    def __post_init__(self):
        self.cls = np.asarray(self.cls, dtype=np.int64)
        self.cell = np.asarray(self.cell, dtype=np.int64)
        ncell = int(self.cell.max(initial=-1)) + 1
        key = self.cls * max(ncell, 1) + self.cell
        ukey, self.cg = np.unique(key, return_inverse=True)
        self.cg_cls = ukey // max(ncell, 1)
        self.cg_cell = ukey % max(ncell, 1)
        self.n_cg = np.bincount(self.cg, minlength=len(ukey)).astype(np.float64)
        self.classes, self.cg_c = np.unique(self.cg_cls, return_inverse=True)
        self.ncell_c = np.bincount(self.cg_c, minlength=len(self.classes)).astype(np.float64)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def cg_means(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return np.bincount(self.cg, weights=v, minlength=len(self.n_cg)) / self.n_cg

    def class_values(self, v: np.ndarray) -> np.ndarray:
        if v.ndim == 2:
            return np.stack([self.class_values(c) for c in v.T], axis=1)
        return np.bincount(self.cg_c, weights=self.cg_means(v), minlength=self.n_classes) / self.ncell_c

    def balanced(self, v: np.ndarray) -> float:
        if len(self.cls) == 0:
            return float("nan")
        return float(np.mean(self.class_values(np.asarray(v, dtype=np.float64))))

    def family_macro(self, v: np.ndarray, family_of_class: np.ndarray) -> float:
        if len(self.cls) == 0:
            return float("nan")
        cv = self.class_values(np.asarray(v, dtype=np.float64))
        fam = np.asarray(family_of_class)[self.classes]
        uf, fi = np.unique(fam, return_inverse=True)
        return float(np.mean(np.bincount(fi, weights=cv) / np.bincount(fi)))

    def pair_weights(self) -> np.ndarray:
        """w with balanced(v) == w @ v."""
        return 1.0 / (self.n_classes * self.ncell_c[self.cg_c[self.cg]] * self.n_cg[self.cg])


def cell_balanced(v: np.ndarray, cell: np.ndarray) -> float:
    if len(v) == 0:
        return float("nan")
    _, ci = np.unique(cell, return_inverse=True)
    return float(np.mean(np.bincount(ci, weights=np.asarray(v, float)) / np.bincount(ci)))


# --------------------------------------------------------------------------- GZSL
def gzsl_summary(S: np.ndarray, prow: np.ndarray, pcol: np.ndarray, seen_cols: np.ndarray,
                 true_mask: np.ndarray | None, chunk: int = 4096) -> dict[str, np.ndarray]:
    """Per pair: own score, same-group strictly-above count and tie count (incl. self), and the
    other group's max score and number of candidates at that max (filtered candidates)."""
    prow = np.asarray(prow, dtype=np.int64)
    pcol = np.asarray(pcol, dtype=np.int64)
    seen_cols = np.asarray(seen_cols, dtype=bool)
    n = len(prow)
    out = {k: np.zeros(n, dtype=np.int64) for k in ("a_same", "t_same", "n_other")}
    out["own"] = np.zeros(n)
    out["other_max"] = np.full(n, -np.inf)
    out["is_seen"] = seen_cols[pcol]
    for s0 in range(0, n, chunk):
        r, c = prow[s0:s0 + chunk], pcol[s0:s0 + chunk]
        m = len(r)
        rows = S[r].astype(np.float64)
        own = rows[np.arange(m), c]
        cand = np.ones(rows.shape, dtype=bool) if true_mask is None else ~true_mask[r]
        cand[np.arange(m), c] = True
        same = cand & (seen_cols[None, :] == seen_cols[c][:, None])
        other = cand & ~same
        out["own"][s0:s0 + m] = own
        out["a_same"][s0:s0 + m] = np.count_nonzero((rows > own[:, None]) & same, axis=1)
        out["t_same"][s0:s0 + m] = np.count_nonzero((rows == own[:, None]) & same, axis=1)
        om = np.where(other, rows, -np.inf).max(axis=1)
        out["other_max"][s0:s0 + m] = om
        out["n_other"][s0:s0 + m] = np.count_nonzero(other & (rows == om[:, None]) & np.isfinite(om)[:, None], axis=1)
    return out


def gzsl_top1(summ: dict[str, np.ndarray], gammas: np.ndarray) -> np.ndarray:
    """Expected filtered top-1 per (gamma, pair) when gamma is subtracted from seen-class scores."""
    gammas = np.asarray(gammas, dtype=np.float64)[:, None]
    sign = np.where(summ["is_seen"], -1.0, 1.0)[None, :]
    with np.errstate(invalid="ignore"):
        diff = (summ["own"] - summ["other_max"])[None, :] + sign * gammas
    diff = np.where(np.isfinite(summ["other_max"])[None, :], diff, np.inf)
    ok = (summ["a_same"] == 0)[None, :]
    t = summ["t_same"][None, :].astype(np.float64)
    val = np.where(diff > 0, 1.0 / t, np.where(diff == 0, 1.0 / (t + summ["n_other"][None, :]), 0.0))
    return np.where(ok, val, 0.0)


def harmonic_mean(s: np.ndarray | float, u: np.ndarray | float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = 2 * s * u / (s + u)
    return np.where((s + u) > 0, h, 0.0)


def select_gamma(gammas: np.ndarray, H: np.ndarray, atol: float = 1e-12) -> int:
    """argmax H; ties -> smallest |gamma|, then smaller gamma."""
    gammas = np.asarray(gammas, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    best = np.flatnonzero(H >= np.max(H) - atol)
    o = np.lexsort((gammas[best], np.abs(gammas[best])))
    return int(best[o[0]])
