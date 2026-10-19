"""Study data: windows/classes/folds/prototypes, label CSR sets, pair tables, split-isolated features."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import folds as F
from .io import SplitAccessError, log, read_json, sha256_array, sha256_file

SPLITS = {"train": 0, "val": 1, "test": 2}


# --------------------------------------------------------------------------- CSR label sets
@dataclass
class CSR:
    indptr: np.ndarray          # int64 (n+1,)
    indices: np.ndarray         # int64 (nnz,)
    data: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.n, dtype=np.int64), self.lengths())

    def row(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def count_in(self, mask: np.ndarray) -> np.ndarray:
        """Per row: number of entries whose column is True in `mask` (bool over columns)."""
        hit = mask[self.indices].astype(np.float64)
        return np.bincount(self.row_ids(), weights=hit, minlength=self.n).astype(np.int64)

    def any_in(self, mask: np.ndarray) -> np.ndarray:
        return self.count_in(mask) > 0

    def all_in(self, mask: np.ndarray) -> np.ndarray:
        return self.count_in(mask) == self.lengths()

    def filter_data(self, min_value: float) -> "CSR":
        keep = self.data >= min_value
        return self.select_entries(keep)

    def select_entries(self, keep: np.ndarray) -> "CSR":
        rid = self.row_ids()[keep]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rid, minlength=self.n))]).astype(np.int64)
        return CSR(indptr, self.indices[keep], None if self.data is None else self.data[keep])

    def blank_rows(self, rows_mask: np.ndarray) -> "CSR":
        return self.select_entries(~rows_mask[self.row_ids()])

    def union(self, other: "CSR") -> "CSR":
        rid = np.concatenate([self.row_ids(), other.row_ids()])
        col = np.concatenate([self.indices, other.indices])
        key = np.unique(rid * (int(max(col.max(initial=0), 0)) + 1) + col)
        ncol = int(max(col.max(initial=0), 0)) + 1
        r, c = key // ncol, key % ncol
        indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=self.n))]).astype(np.int64)
        return CSR(indptr, c.astype(np.int64))

    def take(self, src: np.ndarray) -> "CSR":
        """Row i of the result is row src[i] of self."""
        src = np.asarray(src, dtype=np.int64)
        lens = self.lengths()[src]
        indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        starts = self.indptr[src]
        idx = np.repeat(starts - indptr[:-1], lens) + np.arange(indptr[-1], dtype=np.int64)
        return CSR(indptr, self.indices[idx], None if self.data is None else self.data[idx])

    def dense(self, rows: np.ndarray, col_of: np.ndarray, ncols: int) -> np.ndarray:
        """Bool (len(rows), ncols): entry (i, col_of[c]) True for c in row rows[i] with col_of[c] >= 0."""
        sub = self.take(rows)
        out = np.zeros((len(rows), ncols), dtype=bool)
        cols = col_of[sub.indices]
        ok = cols >= 0
        out[sub.row_ids()[ok], cols[ok]] = True
        return out


def csr_from_lists(lists: Sequence[Sequence[int]], data: Sequence[Sequence[float]] | None = None) -> CSR:
    lens = np.asarray([len(x) for x in lists], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    idx = np.fromiter((int(v) for x in lists for v in x), dtype=np.int64, count=int(indptr[-1]))
    dat = None
    if data is not None:
        dat = np.fromiter((float(v) for x in data for v in x), dtype=np.float64, count=int(indptr[-1]))
    return CSR(indptr, idx, dat)


def _arrow_list_csr(col) -> CSR:
    import pyarrow as pa
    import pyarrow.compute as pc
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    lens = pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False).astype(np.int64)
    vals = pc.list_flatten(arr).to_numpy(zero_copy_only=False)
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    assert len(vals) == indptr[-1]
    return CSR(indptr, np.asarray(vals))


# --------------------------------------------------------------------------- the study
@dataclass
class Study:
    root: Path
    stage: str
    p: dict
    n: int
    split: np.ndarray
    cell: np.ndarray
    cell_idx: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    doy: np.ndarray
    continent: np.ndarray
    coverage: np.ndarray
    purity: np.ndarray
    snow: np.ndarray
    label_stable: np.ndarray | None
    source: np.ndarray | None
    dist_train_km: np.ndarray | None
    class_sel: np.ndarray
    labels: dict[str, CSR]
    ml: CSR
    K: int
    family: np.ndarray
    top_key: np.ndarray
    class_key: np.ndarray
    name: np.ndarray
    folds: dict
    protos: dict[str, np.ndarray]
    hashes: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    labels_full: dict = field(default_factory=dict)

    retained: np.ndarray | None = None
    family_labels: dict[str, CSR] = field(default_factory=dict)
    family_observed: dict[str, CSR] = field(default_factory=dict)

    @property
    def active(self) -> np.ndarray:
        # The legacy synthetic-table interface has no post-QA mask. Canonical
        # inputs always supply a validated mask in canonical.load().
        return np.ones(self.n, dtype=bool) if self.retained is None else self.retained

    def rows_for_split(self, split: str | int) -> np.ndarray:
        code = SPLITS[split] if isinstance(split, str) else int(split)
        return np.flatnonzero(self.active & (self.split == code))

    # ---- footprints ------------------------------------------------------
    def footprint(self, route_name: str) -> str:
        prefix = route_name.split("-", 1)[0] if "-" in route_name else ""
        keys = sorted(self.p["footprint_label_column"], key=len, reverse=True)
        for k in keys:
            if prefix and prefix.startswith(k):
                return k
        return "r1"

    def labels_for(self, fp: str) -> CSR:
        if fp not in self.labels:
            self.warnings.append(f"labels for footprint {fp} unavailable; using r1 labels")
            log("warning", msg=f"footprint {fp} label column missing; falling back to r1")
            return self.labels["r1"]
        return self.labels[fp]

    def exclusion_sets(self, fp: str) -> CSR:
        """Classes whose presence excludes a training window from a fit holding them out:
        route-footprint labels (>= tau) union ml classes with coverage >= exclusion_min."""
        key = ("excl", fp)
        if key not in self.cache:
            ml = self.ml.filter_data(float(self.p["exclusion_min"])) if self.ml.data is not None else self.ml
            self.cache[key] = self.labels_for(fp).union(ml)
        return self.cache[key]

    def family_sets(self, fp: str, *, exclusion: bool = True) -> CSR:
        """Full frozen-taxonomy families, independent of the admitted scoring gallery."""
        tables = self.family_observed if exclusion else self.family_labels
        if tables:
            return tables.get(fp, tables["r1"])
        # Legacy synthetic studies contain only the supplied scoring vocabulary.
        source = self.exclusion_sets(fp) if exclusion else self.labels_for(fp)
        return csr_from_lists([sorted(set(self.family[source.row(i)].tolist())) for i in range(self.n)])

    def ml_truth(self) -> CSR:
        """>= multilabel_min classes (sensitivity truth for multi-label metrics)."""
        return self.ml.filter_data(float(self.p["multilabel_min"])) if self.ml.data is not None else self.ml

    # ---- schemes ---------------------------------------------------------
    @property
    def anchors(self) -> set[int]:
        return {int(a) for a in self.p.get("anchor_families", [])} | {int(a) for a in self.folds.get("anchor_families", [])}

    def heldout_sets(self, scheme: str) -> list[np.ndarray]:
        if scheme == "family":
            out = []
            for fams in self.folds["family_folds"]:
                fm = np.isin(self.family, np.asarray(fams, dtype=np.int64))
                out.append(np.flatnonzero(fm))
            return out
        if scheme == "random_class":
            return [np.asarray(sorted(f), dtype=np.int64) for f in self.folds["random_class_folds"]]
        raise ValueError(scheme)

    def fold_of_class(self, scheme: str) -> np.ndarray:
        f = np.full(self.K, -1, dtype=np.int64)
        for i, h in enumerate(self.heldout_sets(scheme)):
            f[h] = i
        return f

    def unit_of_class(self, scheme: str) -> np.ndarray:
        return self.family.copy() if scheme == "family" else np.arange(self.K, dtype=np.int64)

    def unit_strata(self, scheme: str) -> dict[int, str]:
        if scheme == "family":
            return F.family_top_keys(self.family, self.top_key.tolist())
        return {c: str(self.top_key[c]) for c in range(self.K)}

    def scoreable_classes(self) -> np.ndarray:
        anchors = self.anchors
        return np.asarray([int(self.family[c]) not in anchors for c in range(self.K)], dtype=bool)

    # ---- histograms (labels-only, all splits) ---------------------------
    def class_histograms(self, fp: str, kind: str) -> np.ndarray:
        lab = self.labels_full.get(fp, self.labels_full["r1"]) if self.labels_full else self.labels_for(fp)
        rid = lab.row_ids()
        if kind == "geo":
            g = float(self.p["geo_grid_deg"])
            gi = np.floor((self.lat + 90.0) / g).astype(np.int64)
            gj = np.floor((self.lon + 180.0) / g).astype(np.int64)
            ncol = int(np.ceil(360.0 / g)) + 1
            bins = gi * ncol + gj
        elif kind == "season":
            nb = int(self.p["season_bins"])
            bins = np.minimum(((self.doy.astype(np.float64) - 1.0) % 366.0) * nb // 366.0, nb - 1).astype(np.int64)
        else:
            raise ValueError(kind)
        _, binc = np.unique(bins, return_inverse=True)
        nbin = int(binc.max()) + 1
        H = np.zeros((self.K, nbin))
        keep = self.active[rid]
        np.add.at(H, (lab.indices[keep], binc[rid[keep]]), 1.0)
        return H


def _col(tab, name, default=None):
    if name in tab.column_names:
        return tab.column(name).to_numpy(zero_copy_only=False)
    return default


def load_study(root: str | Path, cfg: Mapping, p: Mapping, stage: str, *, need_folds: bool = True) -> Study:
    """Load windows/classes/folds/prototypes. In stage 'val' every test row's labels are blanked
    (and FeatureStore refuses test rows), so no val-stage computation can score test windows."""
    import pyarrow.parquet as pq
    root = Path(root)
    if stage not in ("val", "test", "labels"):
        raise ValueError(stage)
    if (root / "assemble" / "windows_qa.parquet").exists():
        from .canonical import load
        return load(root, cfg, p, stage)
    wpath, cpath = root / "data" / "windows.parquet", root / "data" / "classes.parquet"
    tab = pq.read_table(wpath)
    n = tab.num_rows
    wid = _col(tab, "window_id")
    if wid is not None and not np.array_equal(np.asarray(wid, dtype=np.int64), np.arange(n)):
        raise ValueError("windows.parquet: window_id must equal the row index")
    split_s = _col(tab, "split")
    split = np.asarray([SPLITS[str(s)] for s in split_s], dtype=np.int8)
    cell = np.asarray(_col(tab, "cell1"), dtype=np.int64)
    _, cell_idx = np.unique(cell, return_inverse=True)
    labels = {}
    for fp, colname in sorted(p["footprint_label_column"].items()):
        if colname in tab.column_names:
            labels[fp] = _arrow_list_csr(tab.column(colname))
    if "r1" not in labels:
        raise ValueError("windows.parquet: r1 label column missing")
    ml = _arrow_list_csr(tab.column("ml_idx")) if "ml_idx" in tab.column_names else CSR(np.zeros(n + 1, np.int64), np.zeros(0, np.int64))
    if "ml_cov" in tab.column_names:
        ml.data = _arrow_list_csr(tab.column("ml_cov")).indices.astype(np.float64)
    ctab = pq.read_table(cpath)
    cidx = np.asarray(_col(ctab, "class_idx"), dtype=np.int64)
    order = np.argsort(cidx)
    if not np.array_equal(cidx[order], np.arange(len(cidx))):
        raise ValueError("classes.parquet: class_idx must be 0..K-1")
    K = len(cidx)
    family = np.asarray(_col(ctab, "family_idx"), dtype=np.int64)[order]
    top_key = np.asarray([str(x) for x in _col(ctab, "top_key")], dtype=object)[order]
    class_key = np.asarray([str(x) for x in _col(ctab, "class_key", np.asarray([str(i) for i in range(K)]))], dtype=object)[order]
    name = np.asarray([str(x) for x in _col(ctab, "name", class_key)], dtype=object)[order] if "name" in ctab.column_names else class_key
    folds_path = root / "data" / "folds.json"
    folds = read_json(folds_path)
    if folds is None:
        if need_folds:
            raise FileNotFoundError(f"{folds_path} missing (run analyse_fit.py --make-folds)")
        folds = {}
    protos = {}
    ppath = root / "text" / "prototypes.npz"
    if ppath.exists():
        with np.load(ppath, allow_pickle=False) as z:
            for k in z.files:
                P = np.asarray(z[k], dtype=np.float64)
                if P.ndim != 2 or P.shape[0] != K:
                    continue
                nrm = np.linalg.norm(P, axis=1)
                if np.any(nrm <= 0) or not np.all(np.isfinite(P)):
                    raise ValueError(f"prototypes[{k}]: zero or non-finite rows")
                protos[k] = P / nrm[:, None]
    hashes = {"windows": sha256_file(wpath), "classes": sha256_file(cpath)}
    if ppath.exists():
        hashes["prototypes"] = sha256_file(ppath)
    if folds_path.exists():
        hashes["folds"] = sha256_file(folds_path)
    class_sel = np.asarray(_col(tab, "class_idx"), dtype=np.int64)

    def f64(name, default=np.nan):
        v = _col(tab, name)
        return np.full(n, default, dtype=np.float64) if v is None else np.asarray(v, dtype=np.float64)

    st = Study(root=root, stage=stage, p=dict(p), n=n, split=split, cell=cell, cell_idx=cell_idx.astype(np.int64),
               lat=f64("lat"), lon=f64("lon"), doy=f64("doy", 1.0), continent=np.asarray([str(x) for x in _col(tab, "continent", np.asarray(["other"] * n))], dtype=object),
               coverage=f64("coverage"), purity=f64("purity"), snow=f64("snow_frac"),
               label_stable=None if _col(tab, "label_stable") is None else np.asarray(_col(tab, "label_stable"), dtype=bool),
               source=None if _col(tab, "source") is None else np.asarray([str(x) for x in _col(tab, "source")], dtype=object),
               dist_train_km=None if _col(tab, "dist_train_km") is None else f64("dist_train_km"),
               class_sel=class_sel, labels=labels, ml=ml, K=K, family=family, top_key=top_key, class_key=class_key,
               name=name, folds=folds, protos=protos, hashes=hashes)
    # label-only histograms may use all splits (pre-registered: labels-only information)
    st.labels_full = dict(labels)
    if stage == "val":
        test = split == SPLITS["test"]
        st.labels = {k: v.blank_rows(test) for k, v in labels.items()}
        st.ml = ml.blank_rows(test)
        st.class_sel = np.where(test, -1, class_sel)
        for k in ("coverage", "purity", "snow"):
            setattr(st, k, np.where(test, np.nan, getattr(st, k)))
    return st


# --------------------------------------------------------------------------- pairs
@dataclass
class PairTable:
    """All (window, class) pairs with class in the window's footprint labels, for one split.
    Sorted by (window, class). Per-scheme folds are attached by `fold_of`."""
    fp: str
    split: int
    win: np.ndarray
    cls: np.ndarray
    cell: np.ndarray
    family: np.ndarray

    @property
    def n(self) -> int:
        return len(self.win)

    def fold(self, study: Study, scheme: str) -> np.ndarray:
        return study.fold_of_class(scheme)[self.cls]

    def key(self) -> str:
        return sha256_array(np.stack([self.win, self.cls]))


def build_pairs(study: Study, fp: str, split: str | int) -> PairTable:
    sc = SPLITS[split] if isinstance(split, str) else int(split)
    if study.stage == "val" and sc == SPLITS["test"]:
        raise SplitAccessError("val stage cannot build test pairs")
    lab = study.labels_for(fp)
    rid = lab.row_ids()
    keep = study.active[rid] & (study.split[rid] == sc) & study.scoreable_classes()[lab.indices]
    win, cls = rid[keep], lab.indices[keep]
    o = np.lexsort((cls, win))
    win, cls = win[o], cls[o]
    return PairTable(fp=fp, split=sc, win=win, cls=cls, cell=study.cell_idx[win], family=study.family[cls])


# --------------------------------------------------------------------------- features
class FeatureStore:
    """Column-concatenation of features/<dir>/<pooling>.npy, restricted to allowed rows.

    In stage 'val' only train+val rows are loaded; asking for any test row raises SplitAccessError.
    """

    def __init__(self, study: Study, spec: Sequence[str], pooling: str):
        root = study.root
        self.spec, self.pooling = list(spec), pooling
        allowed = study.active.copy()
        if study.stage == "val":
            allowed &= study.split != SPLITS["test"]
        self.allowed = allowed
        self.pos = np.full(study.n, -1, dtype=np.int64)
        self.pos[allowed] = np.arange(int(allowed.sum()))
        rows = np.flatnonzero(allowed)
        mats, hashes = [], []
        for d in self.spec:
            fdir = root / "features" / d
            if hasattr(study, "original_class_idx"):
                from .canonical import feature_path
                source = feature_path(root, d, pooling)
                metadata = source.with_suffix(".meta.json")
                if not metadata.exists():
                    raise FileNotFoundError(f"Missing feature metadata: {metadata}")
            else:
                if not (fdir / "complete.json").exists():
                    raise FileNotFoundError(f"{fdir}/complete.json missing (features incomplete)")
                source = fdir / f"{pooling}.npy"
            arr = np.load(source, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[0] != study.n:
                raise ValueError(f"{fdir}/{pooling}.npy has shape {arr.shape}, expected ({study.n}, D)")
            m = np.asarray(arr[rows] if not allowed.all() else arr, dtype=np.float32)
            if not np.all(np.isfinite(m)):
                raise ValueError(f"{fdir}/{pooling}.npy: non-finite values")
            mats.append(m)
            hashes.append(sha256_array(m))
        self.X = mats[0] if len(mats) == 1 else np.ascontiguousarray(np.hstack(mats))
        self.D = self.X.shape[1]
        self.hash = "+".join(hashes)

    def rows(self, idx: np.ndarray) -> np.ndarray:
        p = self.pos[np.asarray(idx, dtype=np.int64)]
        if np.any(p < 0):
            raise SplitAccessError("attempt to read feature rows outside the stage's allowed splits")
        return self.X[p]

    def check(self, idx: np.ndarray) -> None:
        if np.any(self.pos[np.asarray(idx, dtype=np.int64)] < 0):
            raise SplitAccessError("attempt to read feature rows outside the stage's allowed splits")


def available_poolings(root: Path, spec: Sequence[str], poolings: Sequence[str]) -> list[str]:
    if (Path(root) / "assemble/windows_qa.parquet").exists():
        from .canonical import feature_path
        return [pl for pl in poolings if all(feature_path(root, d, pl).exists() for d in spec)]
    return [pl for pl in poolings if all((Path(root) / "features" / d / f"{pl}.npy").exists() for d in spec)]


def write_folds(root: Path, folds: dict) -> Path:
    from .io import write_json
    return write_json(Path(root) / "data" / "folds.json", folds)


def ensure_folds(root: Path, cfg: Mapping, p: Mapping) -> dict:
    """Create data/folds.json deterministically (config seed) if absent; return it."""
    path = Path(root) / "data" / "folds.json"
    if path.exists():
        return json.loads(path.read_text())
    import pyarrow.parquet as pq
    ctab = pq.read_table(Path(root) / "data" / "classes.parquet")
    cidx = np.asarray(ctab.column("class_idx").to_numpy(), dtype=np.int64)
    o = np.argsort(cidx)
    fam = np.asarray(ctab.column("family_idx").to_numpy(), dtype=np.int64)[o]
    tk = [str(x) for x in np.asarray(ctab.column("top_key").to_numpy(zero_copy_only=False), dtype=object)[o]]
    folds = F.make_folds(fam, tk, int(p["outer_folds"]), int(p["seed"]), p.get("anchor_families", []))
    write_folds(root, folds)
    return folds
