"""Validate the retained analysis cohort without changing canonical row IDs.

Author: James Edward Ball. Excluded rows remain in stored arrays, but cannot
contribute to a fitted transform, readout, score or nuisance histogram.
"""
from pathlib import Path

import numpy as np


def validate_retained(windows, records):
    if len(windows) != len(records) or not windows:
        raise ValueError('Retained-window mask is missing or has the wrong row count')
    for i, (window, record) in enumerate(zip(windows, records)):
        if (window['widx'] != i or record.get('widx') != i or
                record.get('window_id') != window['window_id'] or
                type(record.get('retained')) is not bool):
            raise ValueError('Retained-window mask identity or boolean mismatch')
    mask = np.asarray([r['retained'] for r in records], dtype=bool)
    if not mask.any():
        raise ValueError('Retained-window cohort is empty')
    return mask


def retained_mask(root, windows):
    import pyarrow.parquet as pq
    records = pq.read_table(Path(root) / 'admit/windows_final.parquet').to_pylist()
    if hasattr(windows, 'to_dict'):
        windows = windows.to_dict('records')
    return validate_retained(windows, records)
