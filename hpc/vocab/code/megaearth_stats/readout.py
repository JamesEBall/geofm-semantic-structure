"""Per-route readout engine: exclusion-exact ridge fits, inner-fold alpha selection, ZSL/GZSL scoring,
score matrices, multi-label metrics, probe, nulls and scaling curve.

All fits use training windows only; standardisation is per fit. Leakage exclusion (A1-B): every
outer/inner/scaling/null fit drops training windows whose footprint labels or ml classes with
coverage >= exclusion_min include a (pseudo-)held-out class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import folds as Fo
from . import metrics as M
from . import ridge as R
from .io import SplitAccessError, log
from .study import CSR, SPLITS, FeatureStore, PairTable, Study, build_pairs, csr_from_lists


# --------------------------------------------------------------------------- bundles
def make_bundles(lab: CSR, rows: np.ndarray) -> tuple[np.ndarray, CSR]:
    """Unique label sets among `rows` (first-occurrence order). bundle_of[w] = -1 elsewhere."""
    keys: dict[tuple, int] = {}
    lists: list[tuple] = []
    bundle_of = np.full(lab.n, -1, dtype=np.int64)
    ip, ix = lab.indptr, lab.indices
    for w in np.asarray(rows, dtype=np.int64).tolist():
        k = tuple(sorted(set(ix[ip[w]:ip[w + 1]].tolist())))
        b = keys.get(k)
        if b is None:
            b = len(lists)
            keys[k] = b
            lists.append(k)
        bundle_of[w] = b
    return bundle_of, csr_from_lists(lists)


# --------------------------------------------------------------------------- contexts
@dataclass
class InnerCtx:
    H: np.ndarray            # pseudo-held-out classes (sorted)
    rows: np.ndarray         # eligible training windows
    stats: R.Stats
    spec: R.Spectrum
    q: np.ndarray            # val query windows
    ZU: np.ndarray
    prow: np.ndarray
    pcol: np.ndarray
    true_mask: np.ndarray
    cls: np.ndarray
    cell: np.ndarray


@dataclass
class OuterCtx:
    scheme: str
    fold: int
    H: np.ndarray                    # held-out classes (sorted) = outer gallery
    seen: np.ndarray                 # seen classes
    rows: np.ndarray                 # eligible training windows
    stats: R.Stats
    spec: R.Spectrum
    inner: list
    n_train_candidates: int
    q: np.ndarray | None = None      # outer query windows (scored split)
    ZU: np.ndarray | None = None
    pair_idx: np.ndarray | None = None   # indices into the PairTable
    prow: np.ndarray | None = None
    pcol: np.ndarray | None = None
    true_mask: np.ndarray | None = None
    rel_ml: np.ndarray | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Selection:
    alpha: float
    index: int
    mrr: list
    n_pairs: int
    edge: bool
    fallback: bool = False

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "mrr": self.mrr, "n_pairs": self.n_pairs, "grid_edge": self.edge,
                "fallback": self.fallback}


class RouteEngine:
    """Fitting/scoring for one route spec × pooling on one footprint, for one stage ('val'|'test')."""

    def __init__(self, study: Study, feats: FeatureStore, fp: str, p: dict, scored_split: str):
        self.study, self.feats, self.fp, self.p = study, feats, fp, p
        if study.stage == "val" and scored_split == "test":
            raise SplitAccessError("val stage cannot score test windows")
        self.scored = SPLITS[scored_split]
        self.alphas = [float(a) for a in p["alphas"]]
        self.lab = study.labels_for(fp)
        self.excl = study.exclusion_sets(fp)
        self.family_excl = study.family_sets(fp)
        self.family_lab = study.family_sets(fp, exclusion=False)
        self.mlt = study.ml_truth()
        self.chunk = int(p.get("gram_chunk_rows", 16384))
        train = study.active & (study.split == SPLITS["train"])
        self.train_rows = np.flatnonzero(train & (self.lab.lengths() > 0))
        self.bundle_of, self.bundles = make_bundles(self.lab, self.train_rows)
        self.nb = self.bundles.n
        self.m0 = self._mean(self.train_rows)
        self.base = R.accumulate(self.feats.rows, self.train_rows, self.bundle_of[self.train_rows], self.nb,
                                 self.m0, self.chunk)
        self.pairs = build_pairs(study, fp, scored_split)
        self.val_rows = study.rows_for_split("val")
        self._bw_cache: dict = {}

    def _mean(self, rows: np.ndarray) -> np.ndarray:
        s = np.zeros(self.feats.D)
        for a in range(0, len(rows), self.chunk):
            s += self.feats.rows(rows[a:a + self.chunk]).astype(np.float64).sum(0)
        return s / max(len(rows), 1)

    def bw(self, P: np.ndarray, key=None):
        if key is not None and key in self._bw_cache:
            return self._bw_cache[key]
        out = R.bundle_weights(self.bundles, P)
        if key is not None:
            self._bw_cache[key] = out
        return out

    # ---- eligibility -------------------------------------------------------
    def eligible(self, rows: np.ndarray, forbidden: np.ndarray, allowed: np.ndarray | None = None,
                 excl: CSR | None = None, lab: CSR | None = None, *,
                 scheme: str = "random_class", family_excl: CSR | None = None) -> np.ndarray:
        excl = self.excl if excl is None else excl
        lab = self.lab if lab is None else lab
        ok = ~excl.any_in(forbidden)[rows]
        if scheme == "family":
            observed = self.family_excl if family_excl is None else family_excl
            nfamilies = max(int(observed.indices.max(initial=-1)),
                            int(self.family_lab.indices.max(initial=-1)),
                            int(self.study.family.max(initial=-1))) + 1
            blocked = np.zeros(nfamilies, dtype=bool)
            blocked[self.study.family[forbidden]] = True
            ok &= ~observed.any_in(blocked)[rows]
            if allowed is not None:
                permitted = np.zeros(nfamilies, dtype=bool)
                permitted[self.study.family[allowed]] = True
                ok &= self.family_lab.all_in(permitted)[rows]
        elif scheme != "random_class":
            raise ValueError(scheme)
        if allowed is not None:
            ok &= lab.all_in(allowed)[rows]
        return rows[ok]

    # ---- inner contexts ----------------------------------------------------
    def inner_contexts(self, scheme: str, outer: int, H_outer: np.ndarray, units: Sequence[int],
                       base_rows: np.ndarray, base_stats: R.Stats, tag: int, allowed: np.ndarray | None = None,
                       excl: CSR | None = None, lab: CSR | None = None, bundle_of: np.ndarray | None = None,
                       family_excl: CSR | None = None) -> list:
        st = self.study
        unit_of = st.unit_of_class(scheme)
        strata = st.unit_strata(scheme)
        units = sorted(int(u) for u in units)
        try:
            ifolds = Fo.inner_folds(units, [strata[u] for u in units], int(self.p["inner_folds"]),
                                    int(st.folds["inner_seed"]), scheme, outer, tag)
        except ValueError:
            return []
        bundle_of = self.bundle_of if bundle_of is None else bundle_of
        out = []
        for fu in ifolds:
            Hj = np.flatnonzero(np.isin(unit_of, np.asarray(fu)))
            Hj = Hj[st.scoreable_classes()[Hj]]
            fmask = np.zeros(st.K, dtype=bool)
            fmask[H_outer] = True
            fmask[Hj] = True
            amask = None
            if allowed is not None:
                amask = allowed.copy()
                amask[Hj] = False
            rows = self.eligible(base_rows, fmask, amask, excl, lab, scheme=scheme, family_excl=family_excl)
            if len(rows) < 2:
                out.append(None)
                continue
            stats = R.stats_for(self.feats.rows, base_stats, base_rows, rows, bundle_of, self.nb, self.m0, self.chunk)
            spec = R.fit_spectrum(stats, self.m0)
            # validation pairs: val windows, true labels in Hj (never the outer held-out classes)
            hm = np.zeros(st.K, dtype=bool)
            hm[Hj] = True
            vq = self.val_rows[self.lab.any_in(hm)[self.val_rows]]
            if len(vq) == 0:
                out.append(None)
                continue
            col_of = np.full(st.K, -1, dtype=np.int64)
            col_of[Hj] = np.arange(len(Hj))
            tm = self.lab.dense(vq, col_of, len(Hj))
            prow, pcol = np.nonzero(tm)
            ZU = spec.project(self.feats.rows(vq))
            out.append(InnerCtx(H=Hj, rows=rows, stats=stats, spec=spec, q=vq, ZU=ZU, prow=prow, pcol=pcol,
                                true_mask=tm, cls=Hj[pcol], cell=st.cell_idx[vq[prow]]))
        return out

    # ---- alpha selection ---------------------------------------------------
    def select_alpha(self, inner: list, P_train: np.ndarray, rank: int | None = None, bw=None,
                     P_gal: np.ndarray | None = None) -> Selection:
        """Pooled class-balanced filtered ZSL MRR over the inner folds' val pairs; ties -> larger alpha."""
        P_gal = P_train if P_gal is None else P_gal
        bw = self.bw(P_train) if bw is None else bw
        rr = [[] for _ in self.alphas]
        cls, cell = [], []
        for ic in inner:
            if ic is None or len(ic.prow) == 0:
                continue
            cr = R.cross_term(ic.spec, ic.stats, bw, P_train)
            for ai, S in enumerate(R.gallery_dot_scores(ic.spec, ic.ZU, cr, P_gal[ic.H], self.alphas, rank)):
                rr[ai].append(M.rank_pairs(S, ic.prow, ic.pcol, ic.true_mask)["rr"])
            cls.append(ic.cls)
            cell.append(ic.cell)
        if not cls:
            ai = len(self.alphas) - 1
            return Selection(self.alphas[ai], ai, [None] * len(self.alphas), 0, True, fallback=True)
        agg = M.Aggregator(np.concatenate(cls), np.concatenate(cell))
        mrr = np.asarray([agg.balanced(np.concatenate(r)) for r in rr])
        best = np.flatnonzero(mrr >= mrr.max() - 1e-12)
        ai = int(best.max())
        edge = ai in (0, len(self.alphas) - 1)
        return Selection(self.alphas[ai], ai, mrr.tolist(), int(sum(len(c) for c in cls)), edge)

    # ---- outer contexts ----------------------------------------------------
    def outer_context(self, scheme: str, f: int, *, queries: bool = True, inner: bool = True) -> OuterCtx:
        st = self.study
        H = st.heldout_sets(scheme)[f]
        H = H[st.scoreable_classes()[H]]
        fmask = np.zeros(st.K, dtype=bool)
        fmask[H] = True
        seen = np.flatnonzero(~fmask)
        rows = self.eligible(self.train_rows, fmask, scheme=scheme)
        stats = R.stats_for(self.feats.rows, self.base, self.train_rows, rows, self.bundle_of, self.nb, self.m0, self.chunk)
        spec = R.fit_spectrum(stats, self.m0)
        units = self.seen_units(scheme, H)
        ctx = OuterCtx(scheme=scheme, fold=f, H=H, seen=seen, rows=rows, stats=stats, spec=spec, inner=[],
                       n_train_candidates=len(self.train_rows))
        if inner:
            ctx.inner = self.inner_contexts(scheme, f, H, units, rows, stats, tag=0)
        ctx.info = {"n_train": int(len(rows)), "n_excluded": int(len(self.train_rows) - len(rows)),
                    "n_bundles": int(np.count_nonzero(stats.nb > 0)), "n_heldout": int(len(H)),
                    "n_inner": int(sum(1 for c in ctx.inner if c is not None)),
                    "n_additional_family_excluded": int(len(self.eligible(self.train_rows, fmask)) - len(rows))}
        if queries:
            self.attach_queries(ctx)
        return ctx

    def seen_units(self, scheme: str, H: np.ndarray) -> list[int]:
        st = self.study
        unit_of = st.unit_of_class(scheme)
        held_units = set(unit_of[H].tolist())
        anchors = st.anchors
        units = sorted(set(unit_of.tolist()) - held_units)
        if scheme == "family":
            units = [u for u in units if u not in anchors]
        else:
            units = [u for u in units if int(st.family[u]) not in anchors]
        return units

    def attach_queries(self, ctx: OuterCtx) -> None:
        st = self.study
        hm = np.zeros(st.K, dtype=bool)
        hm[ctx.H] = True
        cand = st.rows_for_split(self.scored)
        q = cand[self.lab.any_in(hm)[cand] | self.mlt.any_in(hm)[cand]]
        self.feats.check(q)
        col_of = np.full(st.K, -1, dtype=np.int64)
        col_of[ctx.H] = np.arange(len(ctx.H))
        pf = self.pairs.fold(st, ctx.scheme)
        pidx = np.flatnonzero(pf == ctx.fold)
        row_of = np.full(st.n, -1, dtype=np.int64)
        row_of[q] = np.arange(len(q))
        ctx.q = q
        ctx.pair_idx = pidx
        ctx.prow = row_of[self.pairs.win[pidx]]
        ctx.pcol = col_of[self.pairs.cls[pidx]]
        assert np.all(ctx.prow >= 0) and np.all(ctx.pcol >= 0)
        ctx.true_mask = self.lab.dense(q, col_of, len(ctx.H))
        ctx.rel_ml = self.mlt.dense(q, col_of, len(ctx.H)) | ctx.true_mask
        ctx.ZU = ctx.spec.project(self.feats.rows(q)) if len(q) else np.zeros((0, self.feats.D))

    # ---- outer prediction --------------------------------------------------
    def outer_scores(self, ctx: OuterCtx, P_train: np.ndarray, P_gal: np.ndarray, alpha: float,
                     rank: int | None = None, const: bool = False, bw=None, stats: R.Stats | None = None):
        bw = self.bw(P_train) if bw is None else bw
        cr = R.cross_term(ctx.spec, ctx.stats if stats is None else stats, bw, P_train)
        if const:
            pred = np.broadcast_to(cr.tbar, (len(ctx.q), len(cr.tbar)))
        else:
            pred = R.predict(ctx.spec, ctx.ZU, cr, alpha, rank)
        return R.cosine_scores(pred, P_gal[ctx.H]), cr


# --------------------------------------------------------------------------- result container
class Results:
    """Per-pair arrays (aligned to a PairTable), score matrices and metadata for one output file."""

    def __init__(self, pairs: PairTable):
        self.pairs = pairs
        self.arrays: dict[str, np.ndarray] = {"pairs/win": pairs.win, "pairs/cls": pairs.cls}
        self.meta: dict = {"keys": {}, "fits": {}}

    def pair_array(self, name: str, dtype=np.float64, fill=np.nan) -> np.ndarray:
        if name not in self.arrays:
            self.arrays[name] = np.full(self.pairs.n, fill, dtype=dtype)
        return self.arrays[name]


def key_name(scheme: str, ref: str, pooling: str, variant: str = "ridge") -> str:
    return f"{scheme}.{ref}.{pooling}.{variant}"


# --------------------------------------------------------------------------- main fitting part
def run_scheme(eng: RouteEngine, res: Results, scheme: str, pooling: str, refs: Sequence[str],
               variants: dict[str, dict], *, gzsl_ref: str | None = None, save_weights: dict | None = None,
               ckpt=None, progress: Callable[[str], None] | None = None, folds=None) -> None:
    """Fit/score every outer fold for one scheme × pooling. `variants` maps variant name to options
    {"rank": int|None, "const": bool, "refs": [..]} (ridge variant applies to all refs)."""
    st, p = eng.study, eng.p
    n_folds = len(st.heldout_sets(scheme))
    res.arrays[f"pairs/fold/{scheme}"] = eng.pairs.fold(st, scheme)
    chance = res.pair_array(f"chance/{scheme}")
    kprime = res.pair_array(f"kprime/{scheme}", np.int64, -1)
    mlq = {k: [] for k in ("win", "fold", "R_tau", "R_ml", "fr_chance_tau", "ap_chance_tau", "fr_chance_ml", "ap_chance_ml")}
    mlv: dict[str, dict[str, list]] = {}
    for f in (range(n_folds) if folds is None else folds):
        ctx = eng.outer_context(scheme, f)
        if progress:
            progress(f"{scheme}/{pooling} fold {f}: n_train={ctx.info['n_train']} excluded={ctx.info['n_excluded']}")
        res.arrays[f"queries/{scheme}/f{f}"] = ctx.q
        res.arrays[f"gallery/{scheme}/f{f}"] = ctx.H
        # query-level multi-label bookkeeping (route-independent)
        has_tau = ctx.true_mask.any(1)
        has_ml = ctx.rel_ml.any(1)
        mlq["win"].append(ctx.q)
        mlq["fold"].append(np.full(len(ctx.q), f))
        mlq["R_tau"].append(ctx.true_mask.sum(1))
        mlq["R_ml"].append(ctx.rel_ml.sum(1))
        for ref in refs:
            if ckpt is not None:
                ckpt.maybe_stop()
            P = st.protos[ref]
            bw = eng.bw(P, key=ref)
            for vname, vopt in variants.items():
                if vopt.get("refs") is not None and ref not in vopt["refs"]:
                    continue
                key = key_name(scheme, ref, pooling, vname)
                rank = vopt.get("rank")
                const = bool(vopt.get("const"))
                sel = eng.select_alpha(ctx.inner, P, rank=rank, bw=bw) if not const else \
                    Selection(float("nan"), -1, [], 0, False)
                S, cr = eng.outer_scores(ctx, P, P, sel.alpha, rank=rank, const=const, bw=bw)
                res.arrays[f"scores/{key}/f{f}"] = S
                r = M.rank_pairs(S, ctx.prow, ctx.pcol, ctx.true_mask)
                for m in ("rr", "top1", "top5"):
                    res.pair_array(f"{key}/{m}")[ctx.pair_idx] = r[m]
                chance[ctx.pair_idx] = r["chance"]
                kprime[ctx.pair_idx] = r["kprime"]
                # Within-key ranking (A4 S4): the gallery is restricted to held-out classes that share the
                # target's primary OSM key, so coarse natural-versus-built separation earns nothing.
                wk = within_key_ranks(st, ctx, S)
                res.pair_array(f"{key}/rr_withinkey")[ctx.pair_idx] = wk["rr"]
                res.pair_array(f"chance_withinkey/{scheme}")[ctx.pair_idx] = wk["chance"]
                # multi-label query metrics (tau truth and >= multilabel_min truth)
                d = mlv.setdefault(key, {"fr_tau": [], "ap_tau": [], "fr_ml": [], "ap_ml": []})
                for truth, rel, has in (("tau", ctx.true_mask, has_tau), ("ml", ctx.rel_ml, has_ml)):
                    fr = np.full(len(ctx.q), np.nan)
                    ap = np.full(len(ctx.q), np.nan)
                    if has.any():
                        qm = M.query_metrics(S[has], rel[has])
                        fr[has], ap[has] = qm["fr"], qm["ap"]
                    d[f"fr_{truth}"].append(fr)
                    d[f"ap_{truth}"].append(ap)
                fit = {"selection": sel.as_dict(), **ctx.info,
                       "eff_rank": None if const else ctx.spec.eff_rank(sel.alpha) if rank is None else
                       float(np.sum((ctx.spec.lam / (ctx.spec.lam + sel.alpha))[:rank]))}
                res.meta["fits"].setdefault(key, {})[str(f)] = fit
                res.meta["keys"][key] = {"scheme": scheme, "ref": ref, "pooling": pooling, "variant": vname}
                if sel.edge and not sel.fallback and not const:
                    log("warning", msg="alpha at grid edge", key=key, fold=f, alpha=sel.alpha)
                if save_weights is not None and not const:
                    W = R.weights(ctx.spec, cr, sel.alpha, rank)
                    save_weights[f"W/{key}/f{f}"] = W.astype(np.float32)
                    save_weights[f"mu/{key}/f{f}"] = ctx.spec.mu
                    save_weights[f"sd/{key}/f{f}"] = ctx.spec.sd
                    save_weights[f"tbar/{key}/f{f}"] = cr.tbar
                if gzsl_ref == ref and vname == "ridge":
                    gz = gzsl_fold(eng, ctx, P, sel.alpha, bw)
                    _store_gzsl(res, key_name("gzsl", ref, pooling), f, n_folds, gz)
        # first-relevant / AP chance (route-independent)
        k = len(ctx.H)
        for truth in ("tau", "ml"):
            Rv = mlq[f"R_{truth}"][-1]
            frc = np.full(len(ctx.q), np.nan)
            apc = np.full(len(ctx.q), np.nan)
            ok = Rv > 0
            if ok.any():
                frc[ok] = [M.first_relevant_chance(k, int(x)) for x in Rv[ok]]
                apc[ok] = M.ap_chance(np.full(int(ok.sum()), k), Rv[ok])
            mlq[f"fr_chance_{truth}"].append(frc)
            mlq[f"ap_chance_{truth}"].append(apc)
        del ctx
    for k_, v in mlq.items():
        res.arrays[f"mlq/{scheme}/{k_}"] = np.concatenate(v) if v else np.zeros(0)
    for key, d in mlv.items():
        for k_, v in d.items():
            res.arrays[f"{key}/q_{k_}"] = np.concatenate(v)


def within_key_ranks(st, ctx, S: np.ndarray) -> dict[str, np.ndarray]:
    """Rank each pair among gallery classes with the same top-level key; NaN when that key has one class."""
    keys = np.asarray([str(st.top_key[c]) for c in ctx.H], dtype=object)
    target_key = keys[ctx.pcol]
    rr = np.full(len(ctx.prow), np.nan)
    chance = np.full(len(ctx.prow), np.nan)
    for name in np.unique(target_key):
        cols = keys == name
        if cols.sum() < 2:
            continue
        sel = np.flatnonzero(target_key == name)
        allowed = np.broadcast_to(cols[None, :], S.shape)
        r = M.rank_pairs(S, ctx.prow[sel], ctx.pcol[sel], ctx.true_mask, allowed)
        rr[sel] = r["rr"]
        chance[sel] = r["chance"]
    return {"rr": rr, "chance": chance}


# --------------------------------------------------------------------------- GZSL
def gzsl_fold(eng: RouteEngine, ctx: OuterCtx, P: np.ndarray, alpha: float, bw) -> dict:
    """gamma tuned on pseudo-unseen inner folds (gallery = outer-seen classes), then outer scoring with
    gallery = all K classes (seen = outer-seen)."""
    st = eng.study
    grid = np.asarray(eng.p["_gamma_grid"], dtype=np.float64)
    Hs = []
    for ic in ctx.inner:
        if ic is None:
            continue
        gal = ctx.seen
        col_of = np.full(st.K, -1, dtype=np.int64)
        col_of[gal] = np.arange(len(gal))
        sm = np.zeros(st.K, dtype=bool)
        sm[gal] = True
        vq = eng.val_rows[eng.lab.any_in(sm)[eng.val_rows]]
        if len(vq) == 0:
            continue
        tm = eng.lab.dense(vq, col_of, len(gal))
        prow, pcol = np.nonzero(tm)
        cr = R.cross_term(ic.spec, ic.stats, bw, P)
        pred = R.predict(ic.spec, ic.spec.project(eng.feats.rows(vq)), cr, alpha)
        S = R.cosine_scores(pred, P[gal])
        pseudo_unseen = np.isin(gal, ic.H)
        summ = M.gzsl_summary(S, prow, pcol, ~pseudo_unseen, tm)
        cls = gal[pcol]
        cell = st.cell_idx[vq[prow]]
        T1 = M.gzsl_top1(summ, grid)
        sv = _cb_rows(T1, summ["is_seen"], cls, cell)
        uv = _cb_rows(T1, ~summ["is_seen"], cls, cell)
        Hs.append(M.harmonic_mean(sv, uv))
    if Hs:
        Hmean = np.mean(np.stack(Hs), axis=0)
        gi = M.select_gamma(grid, Hmean)
        gamma = float(grid[gi])
    else:
        Hmean, gamma = np.full(len(grid), np.nan), 0.0
    # outer: all K gallery, queries = scored-split windows with any label
    allc = np.arange(st.K)
    cand = st.rows_for_split(eng.scored)
    q = cand[eng.lab.lengths()[cand] > 0]
    tm = eng.lab.dense(q, allc, st.K)
    prow, pcol = np.nonzero(tm)
    cr = R.cross_term(ctx.spec, ctx.stats, bw, P)
    pred = R.predict(ctx.spec, ctx.spec.project(eng.feats.rows(q)), cr, alpha)
    S = R.cosine_scores(pred, P)
    seen_cols = np.ones(st.K, dtype=bool)
    seen_cols[ctx.H] = False
    summ = M.gzsl_summary(S, prow, pcol, seen_cols, tm)
    T = M.gzsl_top1(summ, np.asarray([0.0, gamma]))
    rr = M.rank_pairs(S, prow, pcol, tm)["rr"]
    # map to the main pair table
    key = q[prow] * st.K + pcol
    pkey = eng.pairs.win * st.K + eng.pairs.cls
    pos = np.searchsorted(pkey, key)
    ok = (pos < len(pkey)) & (pkey[np.minimum(pos, len(pkey) - 1)] == key)
    return {"gamma": gamma, "H_inner": Hmean, "pos": pos[ok], "is_seen": summ["is_seen"][ok],
            "top1_g0": T[0][ok], "top1_cal": T[1][ok], "rr_g0": rr[ok]}


def _cb_rows(T: np.ndarray, mask: np.ndarray, cls: np.ndarray, cell: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(T.shape[0])
    agg = M.Aggregator(cls[mask], cell[mask])
    w = agg.pair_weights()
    return T[:, mask] @ w


def _store_gzsl(res: Results, key: str, f: int, n_folds: int, gz: dict) -> None:
    u = gz["pos"][~gz["is_seen"]]
    s = gz["pos"][gz["is_seen"]]
    res.pair_array(f"{key}/unseen_top1_g0")[u] = gz["top1_g0"][~gz["is_seen"]]
    res.pair_array(f"{key}/unseen_top1_cal")[u] = gz["top1_cal"][~gz["is_seen"]]
    res.pair_array(f"{key}/unseen_rr_g0")[u] = gz["rr_g0"][~gz["is_seen"]]
    for nm, v in (("seen_top1_g0_sum", gz["top1_g0"]), ("seen_top1_cal_sum", gz["top1_cal"])):
        arr = res.pair_array(f"{key}/{nm}", fill=0.0)
        np.add.at(arr, s, v[gz["is_seen"]])
    cnt = res.pair_array(f"{key}/seen_count", fill=0.0)
    np.add.at(cnt, s, 1.0)
    g = res.pair_array(f"{key}/gamma_by_fold_placeholder", fill=np.nan) if False else None  # noqa: F841
    res.arrays[f"{key}/gamma/f{f}"] = np.asarray([gz["gamma"]])
    res.arrays[f"{key}/H_inner/f{f}"] = np.asarray(gz["H_inner"])


# --------------------------------------------------------------------------- supervised probe
def run_probe(eng: RouteEngine, res: Results, pooling: str) -> None:
    """One-hot (unit multi-hot) ridge over all K classes on all labelled train windows; alpha by val
    class-balanced filtered top-1 (gallery all K); scored split class-balanced top-1 and MRR."""
    st = eng.study
    P = np.eye(st.K)
    bw = R.bundle_weights(eng.bundles, P)
    spec = R.fit_spectrum(eng.base, eng.m0)
    cr = R.cross_term(spec, eng.base, bw, P)
    allc = np.arange(st.K)
    vq = eng.val_rows[eng.lab.lengths()[eng.val_rows] > 0]
    tab = []
    if len(vq):
        tm = eng.lab.dense(vq, allc, st.K)
        prow, pcol = np.nonzero(tm)
        ZU = spec.project(eng.feats.rows(vq))
        agg = M.Aggregator(pcol, st.cell_idx[vq[prow]])
        for a in eng.alphas:
            S = R.cosine_scores(R.predict(spec, ZU, cr, a), P)
            tab.append(agg.balanced(M.rank_pairs(S, prow, pcol, tm)["top1"]))
        tab_a = np.asarray(tab)
        ai = int(np.flatnonzero(tab_a >= tab_a.max() - 1e-12).max())
    else:
        ai = len(eng.alphas) - 1
    alpha = eng.alphas[ai]
    cand = st.rows_for_split(eng.scored)
    q = cand[eng.lab.lengths()[cand] > 0]
    pk = f"probe.{pooling}"
    if len(q):
        tm = eng.lab.dense(q, allc, st.K)
        prow, pcol = np.nonzero(tm)
        S = R.cosine_scores(R.predict(spec, spec.project(eng.feats.rows(q)), cr, alpha), P)
        r = M.rank_pairs(S, prow, pcol, tm)
        key = q[prow] * st.K + pcol
        pkey = eng.pairs.win * st.K + eng.pairs.cls
        pos = np.searchsorted(pkey, key)
        ok = (pos < len(pkey)) & (pkey[np.minimum(pos, len(pkey) - 1)] == key)
        for m in ("rr", "top1"):
            res.pair_array(f"{pk}/{m}")[pos[ok]] = r[m][ok]
        res.pair_array(f"{pk}/chance")[pos[ok]] = r["chance"][ok]
    res.meta.setdefault("probe", {})[pooling] = {"alpha": alpha, "val_top1_by_alpha": tab,
                                                 "grid_edge": ai in (0, len(eng.alphas) - 1)}


# --------------------------------------------------------------------------- nulls (one outer fold)
def run_nulls_fold(eng: RouteEngine, f: int, ref: str, n_text: int, n_shuffle: int, seed: int,
                   tag_text: int, tag_shuffle: int, ckpt=None) -> dict:
    """Text-target permutations (prototypes permuted among seen classes for training targets and inner
    galleries; outer gallery unchanged; alpha re-selected) and within-cell image-label-bundle shuffles
    (inner eligibility recomputed with shuffled bundles; alpha re-selected). Returns per-pair RR."""
    st = eng.study
    P = st.protos[ref]
    ctx = eng.outer_context("family", f)
    seen = ctx.seen
    out: dict = {"pair_idx": ctx.pair_idx}
    state = ckpt.load(f"nulls-f{f}", None) if ckpt is not None else None
    text_rr = np.full((n_text, len(ctx.pair_idx)), np.nan)
    text_alpha = np.full(n_text, np.nan)
    shuf_rr = np.full((n_shuffle, len(ctx.pair_idx)), np.nan)
    shuf_alpha = np.full(n_shuffle, np.nan)
    unchanged = np.full(n_shuffle, np.nan)
    start_t, start_s = 0, 0
    if state:
        text_rr[:state["t"]] = np.asarray(state["text_rr"], dtype=np.float64).reshape(state["t"], -1)
        text_alpha[:state["t"]] = state["text_alpha"]
        shuf_rr[:state["s"]] = np.asarray(state["shuf_rr"], dtype=np.float64).reshape(state["s"], -1)
        shuf_alpha[:state["s"]] = state["shuf_alpha"]
        unchanged[:state["s"]] = state["unchanged"]
        start_t, start_s = state["t"], state["s"]

    def save(t, s):
        if ckpt is not None:
            ckpt.save(f"nulls-f{f}", {"t": t, "s": s, "text_rr": text_rr[:t].ravel().tolist(),
                                      "text_alpha": text_alpha[:t].tolist(), "shuf_rr": shuf_rr[:s].ravel().tolist(),
                                      "shuf_alpha": shuf_alpha[:s].tolist(), "unchanged": unchanged[:s].tolist()})

    for b in range(start_t, n_text):
        rng = np.random.default_rng([seed, tag_text, f, b])
        Pb = P.copy()
        Pb[seen] = P[seen[rng.permutation(len(seen))]]
        bw = R.bundle_weights(eng.bundles, Pb)
        sel = eng.select_alpha(ctx.inner, Pb, bw=bw)
        S, _ = eng.outer_scores(ctx, Pb, P, sel.alpha, bw=bw)
        text_rr[b] = M.rank_pairs(S, ctx.prow, ctx.pcol, ctx.true_mask)["rr"]
        text_alpha[b] = sel.alpha
        if ckpt is not None:
            if ckpt.due():
                save(b + 1, start_s)
            ckpt.maybe_stop()
    bw = eng.bw(P, key=ref)
    units = eng.seen_units("family", ctx.H)
    for b in range(start_s, n_shuffle):
        rng = np.random.default_rng([seed, tag_shuffle, f, b])
        src = np.arange(st.n)
        rows = ctx.rows
        cells = st.cell_idx[rows]
        o = np.lexsort((rows, cells))
        rs, cs = rows[o], cells[o]
        bounds = np.flatnonzero(np.diff(cs)) + 1
        for grp in np.split(rs, bounds):
            src[grp] = grp[rng.permutation(len(grp))]
        lab_s, excl_s = eng.lab.take(src), eng.excl.take(src)
        # Shuffled targets cannot make physical held-out image content eligible.
        # Preserve physical and reassigned family observations for inner fits.
        family_excl_s = eng.family_excl.union(eng.family_excl.take(src))
        bundle_s = eng.bundle_of[src]
        unchanged[b] = float(np.mean(bundle_s[rows] == eng.bundle_of[rows]))
        Sb = R.accumulate(eng.feats.rows, rows, bundle_s[rows], eng.nb, eng.m0, eng.chunk, gram=False)
        stats_s = ctx.stats.with_bundles(Sb.Sb, Sb.nb)
        inner = eng.inner_contexts("family", f, ctx.H, units, rows, stats_s, tag=0, excl=excl_s, lab=lab_s,
                                   bundle_of=bundle_s, family_excl=family_excl_s)
        sel = eng.select_alpha(inner, P, bw=bw)
        S, _ = eng.outer_scores(ctx, P, P, sel.alpha, bw=bw, stats=stats_s)
        shuf_rr[b] = M.rank_pairs(S, ctx.prow, ctx.pcol, ctx.true_mask)["rr"]
        shuf_alpha[b] = sel.alpha
        if ckpt is not None:
            if ckpt.due():
                save(n_text, b + 1)
            ckpt.maybe_stop()
    out.update(text_rr=text_rr, text_alpha=text_alpha, shuf_rr=shuf_rr, shuf_alpha=shuf_alpha,
               shuf_unchanged=unchanged)
    return out


# --------------------------------------------------------------------------- scaling (one outer fold)
def run_scaling_fold(eng: RouteEngine, f: int, ref: str, ms: Sequence, n_subsets: int, seed: int, tag: int,
                     ckpt=None) -> dict:
    """For each m: seeded subsets of m seen families (paired across routes); train windows whose labels
    lie within the subset (and no held-out exclusion); alpha re-selected on min(inner_folds, m) inner
    folds within the subset; ZSL on the fold's held-out pairs. Returns per-pair RR averaged over subsets."""
    st = eng.study
    P = st.protos[ref]
    bw = eng.bw(P, key=ref)
    ctx = eng.outer_context("family", f, inner=False)
    units = eng.seen_units("family", ctx.H)
    unit_of = st.unit_of_class("family")
    out = {"pair_idx": ctx.pair_idx, "m": [], "rr": [], "alphas": [], "n_train": [], "n_subsets": [],
           "eff_rank": [], "n_bundles": []}
    state = ckpt.load(f"scaling-f{f}", None) if ckpt is not None else None
    done = {} if not state else state["done"]
    for m in ms:
        subs = Fo.scaling_subsets(units, m, n_subsets, seed, tag, f)
        if subs is None:
            continue
        mkey = str(m)
        acc = done.get(mkey, {"next": 0, "sum": [0.0] * len(ctx.pair_idx), "alphas": [], "n_train": [],
                              "eff_rank": [], "n_bundles": []})
        for si in range(acc["next"], len(subs)):
            sub = subs[si]
            if m == "all":
                allowed = None
                rows, stats = ctx.rows, ctx.stats
                inner = eng.inner_contexts("family", f, ctx.H, units, rows, stats, tag=0)
                spec = ctx.spec
            else:
                allowed = np.isin(unit_of, np.asarray(sub)) & st.scoreable_classes() | \
                    (np.isin(st.family, np.asarray(sorted(st.anchors), dtype=np.int64)) if st.anchors else False)
                fmask = np.zeros(st.K, dtype=bool)
                fmask[ctx.H] = True
                rows = eng.eligible(ctx.rows, fmask, allowed, scheme="family")
                if len(rows) < 2:
                    acc["alphas"].append(None)
                    acc["n_train"].append(int(len(rows)))
                    acc["eff_rank"].append(None)
                    acc["n_bundles"].append(0)
                    acc["sum"] = [float("nan")] * len(ctx.pair_idx)
                    acc["next"] = si + 1
                    continue
                stats = R.stats_for(eng.feats.rows, ctx.stats, ctx.rows, rows, eng.bundle_of, eng.nb, eng.m0, eng.chunk)
                spec = R.fit_spectrum(stats, eng.m0)
                inner = eng.inner_contexts("family", f, ctx.H, sub, rows, stats, tag=1 + si + 1000 * int(m),
                                           allowed=allowed)
            sel = eng.select_alpha(inner, P, bw=bw)
            sctx = OuterCtx(scheme="family", fold=f, H=ctx.H, seen=ctx.seen, rows=rows, stats=stats, spec=spec,
                            inner=inner, n_train_candidates=ctx.n_train_candidates, q=ctx.q,
                            ZU=spec.project(eng.feats.rows(ctx.q)) if spec is not ctx.spec else ctx.ZU,
                            pair_idx=ctx.pair_idx, prow=ctx.prow, pcol=ctx.pcol, true_mask=ctx.true_mask)
            S, _ = eng.outer_scores(sctx, P, P, sel.alpha, bw=bw)
            rr = M.rank_pairs(S, ctx.prow, ctx.pcol, ctx.true_mask)["rr"]
            acc["sum"] = (np.asarray(acc["sum"]) + rr).tolist()
            acc["alphas"].append(sel.alpha)
            acc["n_train"].append(int(len(rows)))
            acc["eff_rank"].append(spec.eff_rank(sel.alpha))
            acc["n_bundles"].append(int(np.count_nonzero(stats.nb > 0)))
            acc["next"] = si + 1
            done[mkey] = acc
            if ckpt is not None:
                if ckpt.due():
                    ckpt.save(f"scaling-f{f}", {"done": done})
                ckpt.maybe_stop()
        done[mkey] = acc
        out["m"].append(mkey)
        out["rr"].append(np.asarray(acc["sum"]) / len(subs))
        out["alphas"].append(acc["alphas"])
        out["n_train"].append(acc["n_train"])
        out["n_subsets"].append(len(subs))
        out["eff_rank"].append(acc["eff_rank"])
        out["n_bundles"].append(acc["n_bundles"])
    return out


def _chance_rr(ctx: OuterCtx) -> np.ndarray:
    """Expected RR of an image-blind all-tied prediction (used when a subset has no trainable windows)."""
    k = len(ctx.H) - (ctx.true_mask[ctx.prow].sum(1) - 1)
    return M.chance_rr(k)
