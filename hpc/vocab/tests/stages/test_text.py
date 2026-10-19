import numpy as np
from stages.text import make_families, pool_phrases


def test_any_reference_and_transitive_family_merge():
    rows = [dict(class_id=str(i), primary_key='natural', primary_value=str(i), name=str(i)) for i in range(4)]
    x = np.eye(4); y = np.eye(4)
    x[1] = x[0]; y[2] = y[1]
    fam, edges = make_families(rows, {'a': x, 'b': y}, .9)
    assert list(fam) == [0, 0, 0, 1]
    assert any(e[2] == 'a' for e in edges) and any(e[2] == 'b' for e in edges)


def test_primary_and_names_group_without_visual_features():
    rows = [dict(class_id=str(i), primary_key='landuse', primary_value='orchard' if i < 2 else 'forest', name=str(i)) for i in range(3)]
    fam, _ = make_families(rows, {'a': np.eye(3)}, .9)
    assert list(fam) == [0, 0, 1]


def test_phrase_normalisation_and_explicit_description_missing():
    rows = [{'class_idx': 0, 'phrase_kind': 'P1'}, {'class_idx': 0, 'phrase_kind': 'P2'},
            {'class_idx': 1, 'phrase_kind': 'P1'}]
    pooled, missing = pool_phrases(rows, np.array([[10, 0], [0, 2], [1, 1]]), 2)
    np.testing.assert_allclose(pooled['main'][0], [2**-.5, 2**-.5], rtol=1e-6)
    assert missing == [False, True]


def test_prefetch_plans_from_area_store_and_uses_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib, shapely
    from stages import text, _text_wiki as wiki
    from stages._osm_geometry import connect
    from stages._osm_buildareas import SCHEMA
    run = tmp_path / 'run'; (run / 'osm_areas').mkdir(parents=True); (run / 'vocab').mkdir()
    db = connect(run / 'osm_areas/areas.sqlite'); db.executescript(SCHEMA)
    w = shapely.to_wkb(shapely.Polygon([(0, 0), (1, 0), (1, 1)]))
    rows = [(i, 'natural=wood', w, 'way', i, 1, hashlib.sha1(w).digest()) for i in range(150)]
    rows += [(1000 + i, 'landuse=farmland|crop=rice', w, 'way', 1000 + i, 1, hashlib.sha1(w).digest()) for i in range(120)]
    rows += [(5000, 'amenity=zoo', w, 'way', 5000, 1, hashlib.sha1(w).digest())]       # below the floor
    db.executemany('INSERT INTO areas VALUES (?,?,?,?,?,?,?)', rows); db.commit(); db.close()
    cfg = {'text_prefetch': {'min_polygons': 100, 'max_units': 2, 'throttle_s': 0.0}}
    ctx = SimpleNamespace(cfg=cfg, shared=tmp_path / 'shared', p=lambda *x: run.joinpath(*x), units=lambda: units,
                          requests_session=lambda: SimpleNamespace(close=lambda: None))
    units = text.prefetch_plan(ctx)
    items = sorted(i for u in units for i in u['items'])
    assert 'tag:natural=wood' in items and 'tag:landuse=farmland' in items and 'tag:crop=rice' in items
    assert not any('zoo' in i for i in items) and len(units) == 2
    calls = []
    monkeypatch.setattr(wiki, 'load_or_fetch', lambda shared, sf, base, item, timeout, throttle: (calls.append(item) or ({'item': item}, True)))
    out = text.prefetch_run(ctx, units[0], SimpleNamespace(maybe_stop=lambda: None))
    assert out['metrics']['items'] == len(units[0]['items']) == len(calls)
    assert all(str(p).startswith(str(tmp_path / 'shared' / 'text' / 'taginfo')) for p in out['outputs'])
    assert text.prefetch_final(ctx)['items'] == len(items)
