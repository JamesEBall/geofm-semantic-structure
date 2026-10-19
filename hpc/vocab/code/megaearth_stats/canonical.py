"""Read canonical stage artifacts without changing their permanent identifiers."""
from pathlib import Path
from datetime import datetime
import numpy as np
from .io import read_json, sha256_file
from .study import Study, SPLITS, csr_from_lists


def load(root, cfg, p, stage):
    import pyarrow.parquet as pq
    root = Path(root)
    paths = {"windows": root / 'assemble/windows_qa.parquet', 'labels': root / 'labels_exact/labels.parquet',
             'vocab': root / 'vocab/classes.parquet', 'admission': root / 'admit/classes_final.parquet',
             'retention': root / 'admit/windows_final.parquet',
             'families': root / 'families/families.parquet', 'folds': root / 'families/folds.json'}
    tables = {k: pq.read_table(v).to_pylist() for k,v in paths.items() if v.suffix == '.parquet'}
    windows, labels = tables['windows'], tables['labels']
    from .cohort import validate_retained
    retained = validate_retained(windows, tables['retention'])
    if not windows or len(windows) != len(labels):
        raise ValueError('Missing or misaligned canonical windows and labels')
    for i,(w,l) in enumerate(zip(windows, labels)):
        if w['widx'] != i or l['widx'] != i or w['window_id'] != l['window_id']:
            raise ValueError('Canonical widx/window_id mismatch')
    vocab = sorted(tables['vocab'], key=lambda r:r['class_idx'])
    original = np.asarray([r['class_idx'] for r in vocab], dtype=np.int64)
    if not np.array_equal(original, np.arange(len(original))):
        raise ValueError('Vocabulary IDs must be contiguous immutable class_idx')
    admission = {r['class_idx']: r for r in tables['admission']}
    if set(admission) != set(original):
        raise ValueError('Admission table does not cover the frozen vocabulary')
    kept = np.asarray([i for i in original if admission[int(i)]['admitted']],dtype=np.int64)
    if not len(kept):
        raise ValueError('No admitted classes')
    # Internal numerical indices only. All output records retain this explicit mapping.
    mapping = {int(c):i for i,c in enumerate(kept)}
    families = {r['class_idx']:r['family_id'] for r in tables['families']}
    if set(families) != set(original):
        raise ValueError('Family table does not cover the frozen vocabulary')
    family = np.asarray([families[int(i)] for i in kept],dtype=np.int64)
    folds = read_json(paths['folds'])
    if not folds:
        raise ValueError('Frozen folds are missing')
    folds = dict(folds)
    folds['random_class_folds'] = [[mapping[c] for c in fold if c in mapping] for fold in folds['random_class_folds']]
    folds['family_folds'] = [[int(f) for f in fold if f in set(family)] for fold in folds['family_folds']]
    if any(not fold for fold in folds['family_folds']):
        raise ValueError('Admission emptied a frozen outer family fold; no folds may be redesigned')
    def lab_column(col, cov=None):
        ids, values = [], []
        for row in labels:
            if col not in row:
                raise ValueError(f'Missing exact footprint labels: {col}')
            source = row[col]
            data = row.get(cov) if cov else None
            if cov and (data is None or len(data) != len(source)):
                raise ValueError(f'Missing or misaligned label coverage: {cov}')
            if any(c < 0 or c >= len(original) for c in source):
                raise ValueError('Label class ID outside vocabulary')
            ids.append([mapping[c] for c in source if c in mapping])
            if cov:
                values.append([v for c,v in zip(source,data) if c in mapping])
        return csr_from_lists(ids, values if cov else None)
    labs = {fp:lab_column(col) for fp,col in p['footprint_label_column'].items()}
    ml = lab_column('multi_idx', 'multi_cov')
    # Admission limits scoring targets, not the observed taxonomy used to exclude
    # related held-out families. Keep full taxonomy observations at permanent widx.
    family_labels, family_observed = {}, {}
    for fp, col in p['footprint_label_column'].items():
        positive, observed = [], []
        for row in labels:
            fams = {int(families[c]) for c in row[col]}
            positive.append(sorted(fams))
            observed.append(sorted(fams | {int(families[c]) for c, v in
                zip(row['multi_idx'], row['multi_cov']) if v >= float(p['exclusion_min'])}))
        family_labels[fp] = csr_from_lists(positive)
        family_observed[fp] = csr_from_lists(observed)
    n=len(windows)
    split=np.asarray([SPLITS[w['split']] for w in windows],dtype=np.int8)
    cells=np.asarray([w['deg_cell'] for w in windows])
    _,cell=np.unique(cells,return_inverse=True)
    # One geographic cell must have one split across all observations.
    for c in np.unique(cell):
        if len(np.unique(split[cell==c])) != 1:
            raise ValueError('Geographic cell crosses splits')
    def floats(key,default=np.nan):
        return np.asarray([w.get(key, default) for w in windows],dtype=np.float64)
    protos={}
    hashes={k:sha256_file(v) for k,v in paths.items()}
    for path in sorted((root/'text').glob('emb_*.npy')):
        P=np.load(path,allow_pickle=False)
        if P.ndim != 2 or len(P) != len(original) or not np.isfinite(P).all():
            raise ValueError(f'Invalid prototype array {path}')
        P=np.asarray(P[kept],dtype=np.float64)
        norm=np.linalg.norm(P,axis=1)
        if np.any(norm <= 0):
            raise ValueError(f'Zero prototype {path}')
        ref=path.stem[4:];protos[ref]=P/norm[:,None];hashes[ref]=sha256_file(path)
    missing=set(p['refs'])-set(protos)
    if missing:
        raise ValueError(f'Missing text references: {sorted(missing)}')
    doy=np.asarray([datetime.fromisoformat(w['datetime'].replace('Z','+00:00')).timetuple().tm_yday for w in windows])
    st=Study(root,stage,dict(p),n,split,cell,cell,floats('lat'),floats('lon'),doy,
             np.asarray([w.get('continent','other') for w in windows]),floats('coverage'),floats('purity'),floats('snow_frac'),
             None,np.asarray([w.get('source','unknown') for w in windows]),None,
             np.asarray([mapping.get(w.get('target_class_idx',-1),-1) for w in windows]),labs,ml,len(kept),family,
             np.asarray([vocab[i]['primary_key'] for i in kept]),np.asarray([vocab[i]['class_id'] for i in kept]),
             np.asarray([vocab[i]['name'] for i in kept]),folds,protos,hashes)
    st.retained=retained
    st.family_labels=family_labels
    st.family_observed=family_observed
    st.original_class_idx=kept
    st.window_ids=np.asarray([w['window_id'] for w in windows])
    st.labels_full=dict(labs)
    if stage=='val':
        test=split==SPLITS['test']
        st.labels={k:v.blank_rows(test) for k,v in labs.items()};st.ml=ml.blank_rows(test)
        st.family_labels={k:v.blank_rows(test) for k,v in family_labels.items()}
        st.family_observed={k:v.blank_rows(test) for k,v in family_observed.items()}
        st.class_sel=np.where(test,-1,st.class_sel)
        for k in ('coverage','purity','snow'):
            setattr(st,k,np.where(test,np.nan,getattr(st,k)))
    return st


def feature_path(root, spec, pooling):
    root=Path(root)
    if '-' in spec:
        route,model=spec.split('-',1)
        return root/'extract'/route/model/f'{pooling}.npy'
    return root/'features'/f'{spec}.npy'
