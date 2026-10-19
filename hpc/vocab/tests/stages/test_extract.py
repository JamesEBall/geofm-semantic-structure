from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mc.core import atomic_write_json
from mc.registry import StageFailed
from stages.extract import load_batch, route_pixels
from stages._extract_freeze import make_freeze,verify_freeze


def arrays():
    a=np.empty((1,4,256,256),dtype='uint16');b=np.empty((1,6,128,128),dtype='uint16')
    for i in range(4):a[:,i]=1000+100*i
    for i in range(6):b[:,i]=2000+100*i
    return a,b


@pytest.mark.parametrize('route,size,bands',[('r1',128,6),('r2',64,6),('r2g1875',64,6),('r1c64',64,6),('r3',256,10)])
def test_route_bands_offsets_dimensions(route,size,bands):
    a,b=arrays();x=route_pixels(a,b,[-1000],[.0001],route)
    assert x.shape==(1,bands,size,size) and x.dtype==np.float32
    expected=[0,.01,.02,.03,.1,.11,.12,.13,.14,.15] if route=='r3' else [0,.01,.02,.13,.14,.15]
    np.testing.assert_allclose(x[0,:,0,0],expected,atol=1e-7)


def test_spatial_centres_and_r2_area_average():
    a,b=arrays();a[:]=0;b[:]=0
    a[0,0]=np.arange(256)[None,:]
    x=route_pixels(a,b,[0],[1],'r1')
    np.testing.assert_array_equal(x[0,0,0],np.arange(64,192))
    x=route_pixels(a,b,[0],[1],'r1c64')
    np.testing.assert_array_equal(x[0,0,0],np.arange(96,160))
    x=route_pixels(a,b,[0],[1],'r2')
    assert x[0,0,0,0]==68.5 and x[0,0,0,-1]==186.5


@pytest.mark.parametrize('route', ['r1', 'r2', 'r2g1875', 'r1c64', 'r3', 'r1rgb224'])
def test_batch_loader_preserves_pixels_and_interleaved_row_order(tmp_path, monkeypatch, route):
    import pandas as pd
    rng = np.random.default_rng(17)
    stored = {}
    for shard in ('a', 'b'):
        (tmp_path / shard).mkdir()
        stored[shard] = (rng.integers(0, 10000, (3, 4, 256, 256), dtype=np.uint16),
                         rng.integers(0, 10000, (3, 6, 128, 128), dtype=np.uint16))
        for filename, values in zip(('b10.npy', 'b20.npy'), stored[shard]):
            np.save(tmp_path / shard / filename, values)
    table = pd.DataFrame({'shard_dir': ['b', 'a', 'b', 'a'], 'row': [2, 0, 1, 0],
                          'dn_offset': [-1000, 0, -1000, 0],
                          'dn_scale': [.0001, .0002, .0001, .0003]}, index=[90, 4, 90, 2])
    expected = route_pixels(*(np.stack([stored[s][band][r] for s, r in
                                        zip(table.shard_dir, table.row)]) for band in (0, 1)),
                            table.dn_offset, table.dn_scale, route)
    real_load = np.load
    mappings = []

    def tracked_load(*args, **kwargs):
        assert all(m._mmap.closed for m in mappings)
        mapped = real_load(*args, **kwargs)
        mappings.append(mapped)
        return mapped

    monkeypatch.setattr(np, 'load', tracked_load)
    actual = load_batch(SimpleNamespace(run=tmp_path), table, route)
    np.testing.assert_array_equal(actual, expected)
    assert len(mappings) == 4
    assert all(m._mmap.closed for m in mappings)


def test_batch_loader_closes_mapping_on_bad_row(tmp_path, monkeypatch):
    import pandas as pd
    np.save(tmp_path / 'b10.npy', np.zeros((1, 4, 256, 256), dtype=np.uint16))
    mapping = np.load(tmp_path / 'b10.npy', mmap_mode='r')
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: mapping)
    table = pd.DataFrame({'shard_dir': ['.'], 'row': [10], 'dn_offset': [0], 'dn_scale': [1]})
    with pytest.raises(IndexError):
        load_batch(SimpleNamespace(run=tmp_path), table, 'r1')
    assert mapping._mmap.closed


def test_batch_loader_large_batch_under_low_file_limit(tmp_path):
    import subprocess
    import sys
    # A fresh process keeps the reduced resource limit away from pytest/plugins.
    # Tiny arrays isolate descriptor use from the separately tested pixel routes.
    script = '''
import resource, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from stages import extract
root = Path(sys.argv[1])
values = np.arange(8, dtype=np.uint16).reshape(2, 1, 2, 2)
for name in ('b10.npy', 'b20.npy'):
    np.save(root / name, values)
table = pd.DataFrame({'shard_dir': ['.'] * 512, 'row': [1, 0] * 256,
                      'dn_offset': [0] * 512, 'dn_scale': [1] * 512})
extract.route_pixels = lambda a, b, *args: (a, b)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, soft), hard))
for _ in range(3):
    result = extract.load_batch(SimpleNamespace(run=root), table, 'r1')
    for band in result:
        np.testing.assert_array_equal(band, values[table.row.to_numpy()])
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True, timeout=30)


def test_freeze_detects_changed_source_and_refuses_replacement(tmp_path):
    run=tmp_path/'run';code=tmp_path/'code';code.mkdir();run.mkdir()
    (code/'main.py').write_text('x=1')
    source=run/'source.json';atomic_write_json(source,{'x':1})
    ctx=SimpleNamespace(run=run,code=code,cfg={'_fingerprint':'abc'},p=lambda *x:run.joinpath(*x))
    make_freeze(ctx,'f1',[source]);verify_freeze(ctx,'f1')
    atomic_write_json(source,{'x':2})
    with pytest.raises(StageFailed):verify_freeze(ctx,'f1')
    with pytest.raises(StageFailed):make_freeze(ctx,'f1',[source])


def test_freeze_accepts_recorded_amendment_but_not_file_changes(tmp_path):
    from stages._extract_freeze import amend_freeze
    run=tmp_path/'run';code=tmp_path/'code';code.mkdir();run.mkdir();(code/'main.py').write_text('x=1')
    source=run/'source.json';atomic_write_json(source,{'x':1})
    ctx=SimpleNamespace(run=run,code=code,cfg={'_fingerprint':'abc'},p=lambda *x:run.joinpath(*x))
    make_freeze(ctx,'f1',[source]);verify_freeze(ctx,'f1')
    ctx.cfg['_fingerprint']='def'
    with pytest.raises(StageFailed,match='amendment'):verify_freeze(ctx,'f1')
    entry=amend_freeze(ctx,'f1','A5',note='gate statistics restricted')
    assert entry['previous_config_fingerprint']=='abc' and entry['config_fingerprint']=='def'
    verify_freeze(ctx,'f1')
    ctx.cfg['_fingerprint']='abc';verify_freeze(ctx,'f1')          # the original still verifies
    atomic_write_json(source,{'x':2})
    with pytest.raises(StageFailed):amend_freeze(ctx,'f1','A6')      # file changes are never amendable
