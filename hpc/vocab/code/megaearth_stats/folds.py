"""Deterministic fold construction (outer family / random-class folds, inner folds, scaling subsets)."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

SCHEME_IDS = {"family": 0, "random_class": 1}


def family_top_keys(family_idx: np.ndarray, top_key: Sequence[str]) -> dict[int, str]:
    """Top key of each family = most frequent top_key among its classes (ties -> smallest string)."""
    by_f: dict[int, Counter] = {}
    for f, k in zip(np.asarray(family_idx).tolist(), list(top_key)):
        by_f.setdefault(int(f), Counter())[str(k)] += 1
    out = {}
    for f, c in by_f.items():
        best = max(c.values())
        out[f] = sorted(k for k, v in c.items() if v == best)[0]
    return out


def stratified_deal(units: Sequence[int], strata: Sequence[str], n_folds: int,
                    rng: np.random.Generator) -> list[list[int]]:
    """Shuffle units within each stratum (strata visited in sorted order) and deal them round-robin
    with one global counter, so folds are balanced by stratum and as equal in size as possible."""
    units = np.asarray(units, dtype=np.int64)
    strata = np.asarray([str(s) for s in strata], dtype=object)
    order = np.argsort(units, kind="stable")
    units, strata = units[order], strata[order]
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    counter = 0
    for s in sorted(set(strata.tolist())):
        members = units[strata == s]
        for u in rng.permutation(members).tolist():
            folds[counter % n_folds].append(int(u))
            counter += 1
    return [sorted(f) for f in folds]


def make_folds(family_idx: np.ndarray, top_key: Sequence[str], n_folds: int, seed: int,
               anchor_families: Sequence[int] = ()) -> dict:
    """Outer folds: families stratified by top_key; random class folds (sensitivity); inner seed.

    Anchor families (always seen, never scored) are excluded from both fold lists.
    """
    family_idx = np.asarray(family_idx, dtype=np.int64)
    anchors = {int(a) for a in anchor_families}
    rng = np.random.default_rng([int(seed), 101])
    ftk = family_top_keys(family_idx, top_key)
    fams = sorted(f for f in ftk if f not in anchors)
    if len(fams) < n_folds:
        raise ValueError(f"need >= {n_folds} non-anchor families, have {len(fams)}")
    family_folds = stratified_deal(fams, [ftk[f] for f in fams], n_folds, rng)
    classes = [c for c in range(len(family_idx)) if int(family_idx[c]) not in anchors]
    perm = rng.permutation(np.asarray(classes, dtype=np.int64)).tolist()
    random_folds = [sorted(int(c) for c in perm[i::n_folds]) for i in range(n_folds)]
    inner_seed = int(rng.integers(0, 2**31 - 1))
    return {"family_folds": family_folds, "random_class_folds": random_folds, "inner_seed": inner_seed,
            "anchor_families": sorted(anchors), "seed": int(seed), "n_folds": int(n_folds)}


def inner_folds(units: Sequence[int], strata: Sequence[str], n_inner: int, inner_seed: int,
                scheme: str, outer: int, tag: int = 0) -> list[list[int]]:
    """Inner folds over seen units (families or classes), stratified by top_key, seeded by
    (inner_seed, scheme, outer fold, tag). tag=0 is the main fit (so scaling m='all' reproduces it)."""
    units = list(units)
    n = min(int(n_inner), len(units))
    if n < 2:
        raise ValueError(f"inner CV needs >= 2 units, have {len(units)}")
    rng = np.random.default_rng([int(inner_seed), SCHEME_IDS[scheme], int(outer), int(tag)])
    return stratified_deal(units, strata, n, rng)


def scaling_subsets(seen_units: Sequence[int], m, n_subsets: int, seed: int, tag: int, outer: int) -> list[list[int]] | None:
    """Seeded random subsets of m seen units (paired across routes). 'all' -> the full set once.
    Returns None when m >= number of seen units (identical to 'all')."""
    seen = sorted(int(u) for u in seen_units)
    if m == "all":
        return [seen]
    m = int(m)
    if m >= len(seen):
        return None
    out = []
    for s in range(int(n_subsets)):
        rng = np.random.default_rng([int(seed), int(tag), int(outer), m, s])
        out.append(sorted(rng.choice(np.asarray(seen), size=m, replace=False).tolist()))
    return out
