"""Two-way (test cell × held-out family) paired bootstrap, drawn once and shared by every analysis.

Per draw b with cell multiplicities v and family multiplicities u, the class-balanced metric of a per-pair
value x is

    metric_b(x) = Σ_c u_f(c) · [Σ_g v_g · m_cg(x)] / [Σ_g v_g · 1(n_cg > 0)]  /  Σ_c u_f(c) · 1(den_c > 0)

where m_cg(x) is the mean of x over the class's pairs in cell g (window -> cell -> class -> mean). A class
whose cells all have zero multiplicity is DROPPED from that draw's average (never set to zero); the
u-weighted dropped fraction is recorded per draw. The metric is linear in x, so metrics of paired
differences equal differences of metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .io import read_npz, write_npz


# --------------------------------------------------------------------------- draws
def draw_indices(n_cells: int, n_fams: int, B: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """V (B, n_cells) and U (B, n_fams) multinomial multiplicities (sampling with replacement)."""
    rng = np.random.default_rng([int(seed), 7919])
    V = rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells), size=B) if n_cells else np.zeros((B, 0), np.int64)
    U = rng.multinomial(n_fams, np.full(n_fams, 1.0 / n_fams), size=B) if n_fams else np.zeros((B, 0), np.int64)
    dt = np.uint16 if max(n_cells, n_fams) < 65535 else np.uint32
    return V.astype(dt), U.astype(dt)


@dataclass
class Draws:
    cells: np.ndarray        # sorted cell ids (study.cell values, i.e. cell1)
    families: np.ndarray     # sorted family ids
    V: np.ndarray
    U: np.ndarray
    meta: dict

    @property
    def B(self) -> int:
        return self.V.shape[0]

    def head(self, b: int) -> "Draws":
        return Draws(self.cells, self.families, self.V[:b], self.U[:b], self.meta)


def make_draws(cells: np.ndarray, families: np.ndarray, B: int, seed: int) -> Draws:
    cells = np.unique(np.asarray(cells, dtype=np.int64))
    families = np.unique(np.asarray(families, dtype=np.int64))
    V, U = draw_indices(len(cells), len(families), int(B), int(seed))
    return Draws(cells, families, V, U, {"B": int(B), "seed": int(seed), "n_cells": int(len(cells)),
                                         "n_families": int(len(families))})


def save_draws(path, d: Draws, extra_meta: dict | None = None) -> None:
    write_npz(path, {"cells": d.cells, "families": d.families, "V": d.V, "U": d.U},
              {**d.meta, **(extra_meta or {}), "complete": True})


def load_draws(path) -> Draws:
    a, meta = read_npz(path)
    return Draws(a["cells"], a["families"], a["V"], a["U"], meta)


# --------------------------------------------------------------------------- weighted metrics
class PairBoot:
    """Bootstrap engine for one pair support (cls, cell id, family id per pair)."""

    def __init__(self, cls: np.ndarray, cell: np.ndarray, family: np.ndarray, draws: Draws | None,
                 block: int = 512):
        cls = np.asarray(cls, dtype=np.int64)
        cell = np.asarray(cell, dtype=np.int64)
        family = np.asarray(family, dtype=np.int64)
        self.n = len(cls)
        self.block = int(block)
        if self.n == 0:
            self.n_cg = 0
            self.draws = draws
            return
        ncell = int(cell.max()) + 1
        key = cls * ncell + cell
        ukey, self.cg = np.unique(key, return_inverse=True)
        self.n_cg = len(ukey)
        self.cg_cls_id = ukey // ncell
        self.cg_cell_id = ukey % ncell
        self.cnt = np.bincount(self.cg, minlength=self.n_cg).astype(np.float64)
        self.classes, self.cg_c = np.unique(self.cg_cls_id, return_inverse=True)
        fam_of_class = np.zeros(len(self.classes), dtype=np.int64)
        fam_of_class[self.cg_c[self.cg]] = family
        self.fam_of_class = fam_of_class
        self.draws = draws
        if draws is not None:
            cpos = np.searchsorted(draws.cells, self.cg_cell_id)
            if np.any(cpos >= len(draws.cells)) or np.any(draws.cells[np.minimum(cpos, len(draws.cells) - 1)] != self.cg_cell_id):
                raise ValueError("pair cell not in the bootstrap cell list")
            fpos = np.searchsorted(draws.families, fam_of_class)
            if np.any(fpos >= len(draws.families)) or np.any(draws.families[np.minimum(fpos, len(draws.families) - 1)] != fam_of_class):
                raise ValueError("pair family not in the bootstrap family list")
            self.cpos, self.fpos = cpos, fpos
            # (n_cells × n_classes) incidence of class-cell groups
            self.A = sp.csr_matrix((np.ones(self.n_cg), (cpos, self.cg_c)), shape=(len(draws.cells), len(self.classes)))
        self._W = None
        self._dropped = None

    # class-cell means of per-pair values (n_pairs,) or (n_pairs, R)
    def cg_means(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return np.bincount(self.cg, weights=x, minlength=self.n_cg) / self.cnt
        out = np.zeros((self.n_cg, x.shape[1]))
        np.add.at(out, self.cg, x)
        return out / self.cnt[:, None]

    def point_weights(self) -> np.ndarray:
        ncell_c = np.bincount(self.cg_c, minlength=len(self.classes)).astype(np.float64)
        return 1.0 / (len(self.classes) * ncell_c[self.cg_c])

    def point(self, x: np.ndarray) -> np.ndarray | float:
        """Class-balanced (window->cell->class->mean) point estimate (all multiplicities 1)."""
        if self.n == 0:
            return np.nan if np.ndim(x) == 1 else np.full(np.shape(x)[1], np.nan)
        return self.point_weights() @ self.cg_means(x)

    def family_macro(self, x: np.ndarray) -> float:
        cvals = np.bincount(self.cg_c, weights=self.cg_means(x), minlength=len(self.classes)) / \
            np.bincount(self.cg_c, minlength=len(self.classes))
        uf, fi = np.unique(self.fam_of_class, return_inverse=True)
        return float(np.mean(np.bincount(fi, weights=cvals) / np.bincount(fi)))

    def class_values(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.cg_c, weights=self.cg_means(x), minlength=len(self.classes)) / \
            np.bincount(self.cg_c, minlength=len(self.classes))

    def _weights_block(self, V: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw class-cell weights (b × n_cg) and dropped fractions (b,)."""
        Vf = V.astype(np.float64)
        den = np.asarray((self.A.T @ Vf.T)).T                 # b × n_classes: Σ_g v_g 1(n_cg>0)
        uc = U.astype(np.float64)[:, self.fpos]                # b × n_classes
        present = den > 0
        Z = (uc * present).sum(1)                              # Σ_c u_c 1(den>0)
        tot = uc.sum(1)
        dropped = np.where(tot > 0, (uc * ~present).sum(1) / np.maximum(tot, 1e-300), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            cw = np.where(present, uc / np.where(present, den, 1.0), 0.0) / np.where(Z > 0, Z, np.nan)[:, None]
        W = Vf[:, self.cpos] * cw[:, self.cg_c]
        return W, dropped

    def draws_of(self, x: np.ndarray, B: int | None = None) -> np.ndarray:
        """(B,) or (B, R) bootstrap metric draws."""
        if self.draws is None:
            raise ValueError("no draws attached")
        m = self.cg_means(x)
        Bt = self.draws.B if B is None else min(int(B), self.draws.B)
        out = np.empty((Bt,) + m.shape[1:])
        if self.n == 0:
            out[:] = np.nan
            return out
        cache = self._W is not None and self._W.shape[0] >= Bt
        for s in range(0, Bt, self.block):
            e = min(Bt, s + self.block)
            if cache:
                W = self._W[s:e]
            else:
                W, _ = self._weights_block(self.draws.V[s:e], self.draws.U[s:e])
            out[s:e] = W @ m
        return out

    def cache_weights(self, B: int | None = None, max_bytes: float = 2e9) -> bool:
        Bt = self.draws.B if B is None else min(int(B), self.draws.B)
        if Bt * self.n_cg * 8 > max_bytes:
            return False
        W, d = self._weights_block(self.draws.V[:Bt], self.draws.U[:Bt])
        self._W, self._dropped = W, d
        return True

    def dropped(self, B: int | None = None) -> np.ndarray:
        Bt = self.draws.B if B is None else min(int(B), self.draws.B)
        if self._dropped is not None and len(self._dropped) >= Bt:
            return self._dropped[:Bt]
        out = np.empty(Bt)
        for s in range(0, Bt, self.block):
            e = min(Bt, s + self.block)
            _, out[s:e] = self._weights_block(self.draws.V[s:e], self.draws.U[s:e])
        return out


def naive_metric(cls, cell, family, x, v_of_cell: dict, u_of_family: dict) -> float:
    """Reference implementation by explicit replication (tests): replicate each cell v times and each
    family u times (copies of a family's classes are distinct classes), then class-balanced mean."""
    cls = np.asarray(cls)
    cell = np.asarray(cell)
    family = np.asarray(family)
    x = np.asarray(x, dtype=np.float64)
    class_vals = []
    for c in np.unique(cls):
        f = family[cls == c][0]
        u = u_of_family.get(int(f), 0)
        if u == 0:
            continue
        cell_means = []
        for g in np.unique(cell[cls == c]):
            m = x[(cls == c) & (cell == g)].mean()
            cell_means += [m] * v_of_cell.get(int(g), 0)
        if not cell_means:
            continue
        class_vals += [np.mean(cell_means)] * u
    return float(np.mean(class_vals)) if class_vals else float("nan")


# --------------------------------------------------------------------------- inference helpers
def p_null_centred(point: float, draws: np.ndarray) -> float:
    """One-sided null-centred bootstrap p for H0: Δ <= 0: (1 + #{(Δ*_b − Δ̂) >= Δ̂}) / (1 + B)."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not np.isfinite(point) or len(d) == 0:
        return float("nan")
    return float((1 + np.count_nonzero((d - point) >= point)) / (1 + len(d)))


def p_count_le0(draws: np.ndarray) -> float:
    """Legacy convention (PLAN §2.1 text): (1 + #{Δ*_b <= 0}) / (1 + B). Reported for reference only."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    return float((1 + np.count_nonzero(d <= 0)) / (1 + len(d))) if len(d) else float("nan")


def p_permutation(obs: float, null: np.ndarray) -> float:
    """(1 + #{null >= obs}) / (1 + B)."""
    n = np.asarray(null, dtype=np.float64)
    n = n[np.isfinite(n)]
    if not np.isfinite(obs) or len(n) == 0:
        return float("nan")
    return float((1 + np.count_nonzero(n >= obs)) / (1 + len(n)))


def quantile(draws: np.ndarray, q: float) -> float:
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    return float(np.quantile(d, q, method="linear")) if len(d) else float("nan")


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return quantile(draws, a), quantile(draws, 1.0 - a)


def holm(p: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1); NaN treated as 1."""
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def iut_p(ps: list[float]) -> float:
    """Intersection-union test: reject only if every component rejects -> p = max."""
    ps = [float(x) if x is not None and np.isfinite(x) else 1.0 for x in ps]
    return max(ps) if ps else 1.0
