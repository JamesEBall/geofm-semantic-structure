"""Immutable input and code manifests for the three analysis boundaries."""
from pathlib import Path
from mc.core import atomic_write_json, read_json, sha256_file
from mc.registry import StageFailed


def make_freeze(ctx, name, paths):
    files = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise StageFailed(f'Freeze {name}: required input missing: {path}')
        for p in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.'):
                try:
                    key = str(p.relative_to(ctx.run))
                except ValueError:
                    key = str(p.resolve())
                files[key] = sha256_file(p)
    # Freeze the complete deployed code and protocol, not just the numerical package.
    for p in sorted(ctx.code.rglob('*')):
        if (p.is_file() and p.suffix in ('.py','.json','.md','.yaml','.sh','.sbatch') and
                not any(part in ('__pycache__','env','tests','profiles') or part.startswith('work-') for part in p.relative_to(ctx.code).parts)):
            files[str(p.resolve())] = sha256_file(p)
    doc = {'name':name.lower(),'files':files,'config_fingerprint':ctx.cfg.get('_fingerprint'),
           'author':'James Edward Ball','release':str(ctx.code.resolve())}
    dest = ctx.p('freeze_' + name.lower(),'manifest.json')
    old = read_json(dest)
    if old and old != doc:
        raise StageFailed('Freeze already exists with different content; create a new run')
    atomic_write_json(dest, doc)
    return dest


def verify_freeze(ctx, name):
    doc = read_json(ctx.p('freeze_' + name.lower(),'manifest.json'))
    if not doc or not doc.get('files'):
        raise StageFailed('Missing or empty freeze: ' + name)
    accepted = {doc.get('config_fingerprint')} | {a['config_fingerprint'] for a in doc.get('amendments', [])}
    if ctx.cfg.get('_fingerprint') not in accepted:
        raise StageFailed('Configuration differs from frozen release and no recorded amendment covers it')
    for rel, want in doc['files'].items():
        p = Path(rel) if Path(rel).is_absolute() else ctx.run / rel
        if not p.is_file() or sha256_file(p) != want:
            raise StageFailed('Frozen input mismatch: ' + rel)
    return doc


def amend_freeze(ctx, name, amendment, note=''):
    """Record a dated protocol amendment against an existing freeze without touching its file hashes.

    The original config fingerprint stays in the manifest; the amended fingerprint is appended with
    the amendment id, so `verify_freeze` accepts either and the change is visible in provenance.
    """
    import time
    dest = ctx.p('freeze_' + name.lower(), 'manifest.json')
    doc = read_json(dest)
    if not doc:
        raise StageFailed('No freeze to amend: ' + name)
    for rel, want in doc['files'].items():
        p = Path(rel) if Path(rel).is_absolute() else ctx.run / rel
        if not p.is_file() or sha256_file(p) != want:
            raise StageFailed('Frozen input changed; an amendment cannot cover file changes: ' + rel)
    entry = {'amendment': amendment, 'note': note, 'config_fingerprint': ctx.cfg.get('_fingerprint'),
             'previous_config_fingerprint': doc.get('config_fingerprint'), 'time': time.time(), 'release': str(ctx.code.resolve())}
    doc.setdefault('amendments', []).append(entry)
    atomic_write_json(dest, doc)
    return entry
