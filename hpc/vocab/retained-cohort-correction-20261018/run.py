#!/usr/bin/env python3
"""Retained-cohort derivative runner. Author: James Edward Ball.

This runner never submits jobs. Source runs are inputs; all receipts, plans,
checkpoints and recomputed outputs belong to an explicitly separate output.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(RELEASE / 'code'))

REFRESH = {('r1', 'dofa'), ('r1', 'random_dofa_101'),
           ('r1', 'random_dofa_202'), ('r1', 'random_dofa_303'), ('r2', 'dofa')}
ANALYSES = ('analysis_readouts', 'analysis_nulls', 'analysis_scaling', 'analysis_rsa')
INHERIT = ('assemble', 'admit', 'labels_exact', 'text', 'families',
           'vocab', 'pilot_check', 'weights_fetch', 'acquire', 'acquire_pilot')
STAGES = ('features_fit', 'features', 'extract', *ANALYSES,
          'analysis_inference', 'analysis_qualify')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + '\n'
    try:
        with path.open('x') as f:
            f.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            raise ValueError(f'Immutable record differs: {path}')


def separate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, non-nested namespaces')
    if output == RELEASE or RELEASE in output.parents or output in RELEASE.parents:
        raise ValueError('Output must be outside the immutable corrected release')
    return source, output


def verified_manifest(source, manifest, checked=None):
    checked = set() if checked is None else checked
    doc = read(manifest)
    if not doc.get('files'):
        raise ValueError(f'Empty source freeze: {manifest}')
    for rel, expected in doc['files'].items():
        path = Path(rel)
        if not path.is_absolute():
            path = source / path
            if source not in path.resolve().parents:
                raise ValueError(f'Source manifest path escapes run: {rel}')
        key = (str(path.resolve()), expected)
        if key in checked:
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Source freeze mismatch: {path}')
        checked.add(key)
    return doc


def code_hashes():
    roots = (RELEASE / 'code', RELEASE / 'config.d', HERE)
    paths = [RELEASE / 'config.json']
    for root in roots:
        paths.extend(p for p in root.rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.json', '.md', '.sh', '.sbatch')
                     and '__pycache__' not in p.parts)
    return {str(p.relative_to(RELEASE)): digest(p) for p in sorted(set(paths))}


def link(source, destination):
    if not source.exists():
        raise ValueError(f'Missing inherited input: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def audit_cohort(source, cfg):
    import numpy as np
    from megaearth_stats.io import params
    from megaearth_stats.study import load_study, build_pairs
    st = load_study(source, cfg, params(cfg), 'test')
    pairs = build_pairs(st, 'r1', 'test')
    if (st.K != 188 or len(np.unique(st.family)) != 91 or pairs.n != 8755
            or len(np.unique(pairs.win)) != 5942 or not st.active[pairs.win].all()):
        raise ValueError('Corrected gallery, family or retained test-pair audit differs')
    train = st.rows_for_split('train')
    train = train[st.labels_for('r1').lengths()[train] > 0]
    exact, family = st.exclusion_sets('r1'), st.family_sets('r1')
    additional, retained_fit = [], []
    for held in st.heldout_sets('family'):
        hm = np.zeros(st.K, dtype=bool)
        hm[held] = True
        fm = np.zeros(max(family.indices.max(initial=-1), st.family.max(initial=-1)) + 1,
                      dtype=bool)
        fm[st.family[held]] = True
        before = train[~exact.any_in(hm)[train]]
        after = before[~family.any_in(fm)[before]]
        additional.append(len(before) - len(after))
        retained_fit.append(len(after))
    if additional != [26, 66, 216, 157, 205] or retained_fit != [17585, 15498, 14359, 7902, 14680]:
        raise ValueError('Full-taxonomy family exclusion differs from the frozen audit')
    return {'admitted_gallery_classes': st.K, 'retained_test_pairs': pairs.n,
            'retained_scored_test_windows': len(np.unique(pairs.win)),
            'family_outer_additional_exclusions': additional,
            'family_outer_train_counts': retained_fit}


def context(args, *, preparing=False):
    from mc.context import Ctx
    from mc.core import Store, load_config
    source, output = separate(args.source_run, args.output)
    cfg = load_config(RELEASE, 'full')
    if not preparing:
        identity = read(output / '_retained/identity.json')
        if (identity['source_run'] != str(source) or identity['release'] != str(RELEASE)
                or identity['code'] != code_hashes()
                or identity['config_fingerprint'] != cfg['_fingerprint']):
            raise ValueError('Derivative identity, code or configuration changed')
    cfg['analysis_input_freeze'] = 'f3_cpu' if args.phase == 'cpu' else 'f3'
    root = Path(cfg['root'])
    return Ctx(cfg, RELEASE, root, output, root / 'shared', Store(output),
               threads=args.threads, gpus=int(args.stage == 'extract'),
               profile='full', tmp=Path(os.environ.get('TMPDIR', '/tmp')))


def prepare(args):
    import numpy as np
    import pandas as pd
    from megaearth_stats.cohort import retained_mask
    from mc.core import load_config
    source, output = separate(args.source_run, args.output)
    if output.exists():
        raise ValueError('Preparation requires a new output directory')
    cfg = load_config(RELEASE, 'full')
    freezes = {}
    checked = set()  # process-local reuse only; every new prepare reads source bytes again
    for name in ('f1', 'f2', 'f3'):
        path = source / f'freeze_{name}/manifest.json'
        doc = verified_manifest(source, path, checked)
        accepted = {doc.get('config_fingerprint')} | {
            a['config_fingerprint'] for a in doc.get('amendments', [])}
        if cfg['_fingerprint'] not in accepted:
            raise ValueError(f'Configuration differs from source {name}')
        freezes[name] = digest(path)
    windows = pd.read_parquet(source / 'assemble/windows_qa.parquet')
    mask = retained_mask(source, windows)
    if len(windows) != 44355 or int(mask.sum()) != 34992:
        raise ValueError('Frozen cohort count differs from 44,355 / 34,992')
    counts = {s: int(np.sum(mask & (windows['split'].to_numpy() == s)))
              for s in ('train', 'val', 'test')}
    if counts != {'train': 24574, 'val': 3539, 'test': 6879}:
        raise ValueError('Frozen retained split counts differ')
    cohort_audit = audit_cohort(source, cfg)
    expected, unavailable = [], []
    f3 = read(source / 'freeze_f3/manifest.json')['files']
    for route in cfg['extraction']['routes']:
        for model in route['models']:
            pair = (route['route'], model)
            base = source / 'extract' / pair[0] / pair[1]
            present = [(base / (pool + '.npy')).is_file() for pool in ('mean', 'cls')]
            if any(present) and not all(present):
                raise ValueError(f'Partial source route: {pair}')
            if all(present):
                for pool in ('mean', 'cls'):
                    path = base / (pool + '.npy')
                    key = str(path.relative_to(source))
                    if (str(path.resolve()), f3.get(key)) not in checked:
                        raise ValueError(f'Array absent from original F3: {key}')
                    if np.load(path, mmap_mode='r', allow_pickle=False).shape[0] != 44355:
                        raise ValueError(f'Array row count differs: {key}')
                expected.append(list(pair))
            elif pair in REFRESH:
                expected.append(list(pair))
            else:
                unavailable.append(list(pair))
    from stages.extract import required_models
    available = {f'{r}-{m}' for r, m in expected}
    if required_models(cfg) - available:
        raise ValueError('Missing registered source encoder')
    # Original F2 pins the weights inventory; also pin the actual refresh checkpoint.
    weight_name = 'DOFA_ViT_base_e100.pth'
    weights = read(source / 'weights_fetch/manifest.json')['files']
    weight = Path(cfg['root']) / 'shared/weights/dofa' / weight_name
    if digest(weight) != weights['dofa.' + weight_name]['sha256']:
        raise ValueError('DOFA checkpoint differs from original frozen weights inventory')
    output.mkdir(parents=True)
    for name in INHERIT:
        link(source / name, output / name)
    for route, model in expected:
        if (route, model) not in REFRESH:
            link(source / 'extract' / route / model, output / 'extract' / route / model)
    identity = {'author': 'James Edward Ball', 'source_run': str(source),
                'release': str(RELEASE), 'code': code_hashes(), 'source_freezes': freezes,
                'config_fingerprint': cfg['_fingerprint'], 'expected_routes': expected,
                'unavailable_source_routes': unavailable, 'canonical_rows': 44355,
                'retained_rows': 34992, 'retained_splits': counts,
                'corrections': ['retained_cohort_every_fit_and_readout',
                                'full_taxonomy_family_exclusion'],
                'cohort_audit': cohort_audit,
                'retention_sha256': digest(source / 'admit/windows_final.parquet')}
    write_once(output / '_retained/identity.json', identity)
    ctx = context(args, preparing=True)
    # Reuse already verified bytes within this prepare process. The new F2 stores
    # actual original file entries, so subsequent verification checks each file.
    files = {str((Path(p) if Path(p).is_absolute() else source / p).resolve()): sha
             for p, sha in read(source / 'freeze_f2/manifest.json')['files'].items()}
    files.update({str((RELEASE / p).resolve()): sha for p, sha in identity['code'].items()})
    files.update({str((source / f'freeze_{n}/manifest.json').resolve()): sha for n, sha in freezes.items()})
    files['_retained/identity.json'] = digest(output / '_retained/identity.json')
    files[str(weight.resolve())] = weights['dofa.' + weight_name]['sha256']
    write_once(output / 'freeze_f2/manifest.json', {
        'name': 'f2', 'files': files, 'config_fingerprint': cfg['_fingerprint'],
        'release': str(RELEASE), 'author': 'James Edward Ball',
        'source_f2_manifest_sha256': freezes['f2']})
    print(json.dumps(identity, indent=2))


def receipt_path(ctx, stage, unit):
    return ctx.p('_retained', 'receipts', stage, unit + '.json')


def validate_receipt(ctx, stage, unit):
    rec = read(receipt_path(ctx, stage, unit))
    if rec['identity'] != digest(ctx.p('_retained/identity.json')):
        raise ValueError('Receipt belongs to a different derivative')
    for rel, expected in rec['outputs'].items():
        if digest(ctx.run / rel) != expected:
            raise ValueError(f'Completed output changed: {rel}')
    return rec


def require_complete(ctx, stage):
    return validate_receipt(ctx, stage, '__complete__')


def spec(ctx, name):
    import stages.imagery as imagery
    import stages.extract as extract
    import stages.analysis as analysis
    import stages.qualify as qualify
    return next(s for module in (imagery, extract, analysis, qualify)
                for s in module.STAGES if s.name == name)


def dependencies(ctx, stage):
    if stage == 'features':
        require_complete(ctx, 'features_fit')
    if stage == 'extract':
        require_complete(ctx, 'features_fit')
    if stage in ANALYSES:
        require_complete(ctx, 'features')
        from stages._extract_freeze import verify_freeze
        verify_freeze(ctx, ctx.cfg['analysis_input_freeze'])
    if stage == 'analysis_inference':
        if ctx.cfg['analysis_input_freeze'] != 'f3':
            raise ValueError('Inference requires the full derivative freeze')
        for name in ANALYSES:
            require_complete(ctx, name)
            exact_receipts(ctx, name, load_plan(ctx, name, 'full'))
    if stage == 'analysis_qualify':
        require_complete(ctx, 'analysis_inference')


def plan_path(ctx, stage, phase):
    return ctx.p('_retained', 'plans', phase + '-' + stage + '.json')


def load_plan(ctx, stage, phase):
    return read(plan_path(ctx, stage, phase))


def plan(ctx, args):
    dependencies(ctx, args.stage)
    ctx.stage = args.stage
    units = spec(ctx, args.stage).plan(ctx)
    if args.stage == 'extract':
        units = [u for u in units if (u['route'], u['model']) in REFRESH]
    if args.stage in ANALYSES and args.phase == 'cpu':
        pending = {f'{r}-{m}' for r, m in REFRESH}
        units = [u for u in units if u['route'] not in pending]
    if len({u['id'] for u in units}) != len(units) or not units:
        raise ValueError('Empty or duplicate unit plan')
    if args.stage in ANALYSES and args.phase == 'full':
        old_path = plan_path(ctx, args.stage, 'cpu')
        if old_path.exists():
            current = {u['id']: u for u in units}
            if any(current.get(u['id']) != u for u in read(old_path)):
                raise ValueError('CPU unit plan changed in full phase')
    write_once(plan_path(ctx, args.stage, args.phase), units)
    ctx.store.write_units(args.stage, units)
    print(json.dumps({'stage': args.stage, 'phase': args.phase, 'units': len(units)}))


def record(ctx, stage, uid, unit, outputs, seconds=0):
    hashes = {}
    for path in outputs:
        path = Path(path)
        if not path.is_absolute():
            path = ctx.run / path
        if ctx.run not in path.resolve().parents:
            raise ValueError('A stage tried to register an output outside the derivative')
        hashes[str(path.relative_to(ctx.run))] = digest(path)
    write_once(receipt_path(ctx, stage, uid), {
        'identity': digest(ctx.p('_retained/identity.json')), 'unit': unit,
        'outputs': hashes, 'seconds': seconds,
        'input_freeze': ctx.cfg['analysis_input_freeze']})


def exact_receipts(ctx, stage, units):
    expected = {u['id']: u for u in units}
    records = ctx.p('_retained', 'receipts', stage)
    found = {p.stem for p in records.glob('*.json')} - {'__complete__'}
    if found != set(expected):
        raise ValueError(f'Incomplete/excess receipt IDs for {stage}: '
                         f'missing={sorted(set(expected)-found)}, extra={sorted(found-set(expected))}')
    for uid, unit in expected.items():
        if validate_receipt(ctx, stage, uid)['unit'] != unit:
            raise ValueError('Receipt unit parameters differ')


def verify_frozen_paths(ctx, paths):
    doc = read(ctx.p('freeze_f2/manifest.json'))
    frozen = {str((Path(k) if Path(k).is_absolute() else ctx.run / k).resolve()): v
              for k, v in doc['files'].items()}
    for path in paths:
        path = Path(path).resolve()
        if frozen.get(str(path)) != digest(path):
            raise ValueError(f'Refresh input absent from or changed since derivative F2: {path}')


def verify_raw_inputs(ctx, unit):
    if ctx.stage not in ('features_fit', 'features', 'extract'):
        return
    import pandas as pd
    verify_frozen_paths(ctx, [ctx.p('assemble/windows_qa.parquet'),
                              ctx.p('admit/windows_final.parquet')])
    if ctx.stage == 'extract':
        table = pd.read_parquet(ctx.p('assemble/windows_qa.parquet'))
        shards = table.sort_values('widx').iloc[unit['start']:unit['stop']].shard_dir.unique()
        verify_frozen_paths(ctx, [ctx.shared / 'weights/dofa/DOFA_ViT_base_e100.pth'])
    else:
        shards = [unit['shard_dir']]
    for shard in shards:
        path = ctx.run / shard
        verify_frozen_paths(ctx, [path / 'manifest.json'])
        if ctx.stage == 'extract':
            from stages._imagery_io import verify_shard
            verify_shard(path)


def run_unit(ctx, args):
    from mc.core import Checkpoint
    from mc.runner import install_stop_handlers, stop_requested, _apply_threads
    dependencies(ctx, args.stage)
    units = load_plan(ctx, args.stage, args.phase)
    if args.unit_index is not None:
        if args.unit_index < 0 or args.unit_index >= len(units):
            raise ValueError('Unit array index outside frozen plan')
        args.unit = units[args.unit_index]['id']
    unit = next(u for u in units if u['id'] == args.unit)
    ctx.stage, ctx.unit = args.stage, args.unit
    existing = receipt_path(ctx, args.stage, args.unit)
    if existing.exists():
        if validate_receipt(ctx, args.stage, args.unit)['unit'] != unit:
            raise ValueError('Resume unit parameters differ')
        return
    # Exclusive claim prevents duplicate Slurm units from sharing checkpoint files.
    if not ctx.store.claim(args.stage, args.unit):
        raise ValueError('Unit already claimed')
    install_stop_handlers()
    _apply_threads(args.threads)
    ckpt = Checkpoint(ctx.store, args.stage, args.unit,
                      stop_flag=lambda: stop_requested(ctx.store))
    started = time.monotonic()
    try:
        verify_raw_inputs(ctx, unit)
        result = spec(ctx, args.stage).run(ctx, unit, ckpt) or {}
        record(ctx, args.stage, args.unit, unit, result.get('outputs', []),
               time.monotonic() - started)
        ckpt.clear()
    finally:
        ctx.store.release(args.stage, args.unit)


def finalize(ctx, args):
    dependencies(ctx, args.stage)
    if args.stage in ANALYSES and args.phase != 'full':
        raise ValueError('CPU work stays provisional until the full plan is complete')
    units = load_plan(ctx, args.stage, args.phase)
    exact_receipts(ctx, args.stage, units)
    if receipt_path(ctx, args.stage, '__complete__').exists():
        require_complete(ctx, args.stage)
        return
    ctx.stage = args.stage
    ctx.store.write_units(args.stage, units)
    stage = spec(ctx, args.stage)
    if args.stage == 'extract':
        # Finalise only owned refreshed routes: inherited arrays are read-only inputs.
        cfg = copy.deepcopy(ctx.cfg)
        cfg['extraction']['routes'] = [
            {'route': r, 'models': sorted(m for rr, m in REFRESH if rr == r)}
            for r in sorted({r for r, _ in REFRESH})]
        ctx.cfg = cfg
    if stage.finalize:
        stage.finalize(ctx)
    outputs = [p for p in ctx.stage_dir().rglob('*') if p.is_file()
               and ctx.run in p.resolve().parents]
    record(ctx, args.stage, '__complete__', {'ids': [u['id'] for u in units]}, outputs)


def freeze(ctx, args):
    from stages._extract_freeze import make_freeze, verify_freeze
    require_complete(ctx, 'features')
    name = ctx.cfg['analysis_input_freeze']
    if args.phase == 'full':
        require_complete(ctx, 'extract')
        verify_freeze(ctx, 'f3_cpu')
    identity = read(ctx.p('_retained/identity.json'))
    expected = {tuple(v) for v in identity['expected_routes']}
    if args.phase == 'cpu':
        expected -= REFRESH
    for route, model in expected:
        for pooling in ('mean', 'cls'):
            if not ctx.p('extract', route, model, pooling + '.npy').is_file():
                raise ValueError(f'Expected encoder unavailable: {route}/{model}/{pooling}')
    paths = [ctx.p(n) for n in ('assemble', 'admit', 'labels_exact', 'text',
                                'families', 'vocab', 'features_fit', 'features')]
    paths += [ctx.p('extract', r, m) for r, m in sorted(expected)]
    paths += [ctx.p('_retained/identity.json'), ctx.p('freeze_f2/manifest.json')]
    path = make_freeze(ctx, name, paths)
    if args.phase == 'full':
        cpu, full = read(ctx.p('freeze_f3_cpu/manifest.json')), read(path)
        if any(full['files'].get(k) != v for k, v in cpu['files'].items()):
            raise ValueError('Full freeze changes CPU input, code, cohort or feature hashes')
    print(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'plan', 'unit', 'finalize', 'freeze'))
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=('cpu', 'full'), default='cpu')
    p.add_argument('--stage', choices=STAGES, default='features_fit')
    p.add_argument('--unit')
    p.add_argument('--unit-index', type=int)
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = p.parse_args(argv)
    if args.action == 'prepare':
        prepare(args)
    else:
        ctx = context(args)
        {'plan': plan, 'unit': run_unit, 'finalize': finalize, 'freeze': freeze}[args.action](ctx, args)


if __name__ == '__main__':
    main()
