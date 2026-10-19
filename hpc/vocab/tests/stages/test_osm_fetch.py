from types import SimpleNamespace
import pytest
from stages.osm import fetch_plan, validate_range, EXPECTED_BYTES


def test_ranges_cover_snapshot_once():
    units=fetch_plan(SimpleNamespace(cfg={'osm':{'preferred_snapshot':'planet-251229.osm.pbf','range_mib':256}}))
    assert units[0]['start']==0 and units[-1]['stop']==EXPECTED_BYTES-1
    assert all(a['stop']+1==b['start'] for a,b in zip(units,units[1:]))
    assert sum(u['stop']-u['start']+1 for u in units)==EXPECTED_BYTES


def test_range_validation_rejects_full_body_and_wrong_object():
    with pytest.raises(RuntimeError):
        validate_range(SimpleNamespace(status_code=200,headers={}),0,99)
    with pytest.raises(RuntimeError):
        validate_range(SimpleNamespace(status_code=206,headers={'Content-Range':'bytes 0-99/100'}),0,99)
    validate_range(SimpleNamespace(status_code=206,headers={'Content-Range':f'bytes 0-99/{EXPECTED_BYTES}'}),0,99)
