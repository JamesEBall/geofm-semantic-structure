"""RSA statistics shared by the production stage and its qualification (PLAN §2.1 criterion 4, TEST-PLAN CAL-5).

Author: James Edward Ball.

`statistics` is the production computation of `stages.analysis.rsa_run` lifted into a function so that the
qualification stage can run the identical code on simulated class geometries. Spearman RSA between
rank-transformed cosine distance vectors; Mantel permutation p; partial RSA after projecting out the
nuisance geometries (appearance-rich category geometry and geographic class-histogram distances).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


class RSAError(ValueError):
    pass


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        raise RSAError('RSA distance vector is constant')
    return float(a @ b / den)


def nuisance_projector(rx: np.ndarray, nuisance_vectors: list[np.ndarray]) -> np.ndarray:
    nuis = np.column_stack([np.ones(len(rx))] + [rankdata(v) for v in nuisance_vectors])
    if not np.isfinite(nuis).all():
        raise RSAError('Non-finite RSA nuisance geometry')
    U, D, _ = np.linalg.svd(nuis, full_matrices=False)
    return U[:, D > D.max() * 1e-12]


def statistics(X: np.ndarray, P: np.ndarray, nuisance_vectors: list[np.ndarray], B: int, seed: int,
               seed_tag: int = 10103, keep_null: bool = True) -> dict:
    """Production RSA on class prototypes X (image) and P (text).

    nuisance_vectors are condensed distance vectors over the same classes (unranked; ranked here).
    Returns spearman, mantel_p, partial_spearman, partial_mantel_p and (optionally) the null draws.
    Permutation b uses default_rng([seed, seed_tag, b]) exactly as the production stage does.
    """
    if np.any(np.linalg.norm(X, axis=1) == 0):
        raise RSAError('RSA has a zero feature prototype')
    dx = pdist(X, 'cosine')
    dt = pdist(P, 'cosine')
    rx = rankdata(dx)
    rt = rankdata(dt)
    observed = corr(rx, rt)
    matrix = squareform(rx)
    tri = np.triu_indices(len(X), 1)
    Q = nuisance_projector(rx, nuisance_vectors)
    target = rt - Q @ (Q.T @ rt)

    def partial(v):
        residual = v - Q @ (Q.T @ v)
        den = np.linalg.norm(residual) * np.linalg.norm(target)
        return float(residual @ target / den) if den > 1e-10 else None

    observed_partial = partial(rx)
    null = []
    partial_null = []
    for b in range(int(B)):
        perm = np.random.default_rng([int(seed), int(seed_tag), b]).permutation(len(X))
        permuted = matrix[perm[:, None], perm[None, :]][tri]
        null.append(corr(permuted, rt))
        partial_null.append(partial(permuted))
    null_a = np.asarray(null)
    mantel_p = (1 + int((null_a >= observed).sum())) / (1 + int(B))
    if observed_partial is not None and all(v is not None for v in partial_null):
        partial_mantel_p = (1 + sum(v >= observed_partial for v in partial_null)) / (1 + int(B))
    else:
        partial_mantel_p = None
    out = {'spearman': observed, 'mantel_p': mantel_p, 'partial_spearman': observed_partial,
           'partial_mantel_p': partial_mantel_p, 'n_classes': int(len(X)), 'permutations': int(B)}
    if keep_null:
        out['null'] = null_a
        out['partial_null'] = np.asarray([np.nan if v is None else v for v in partial_null])
    return out
