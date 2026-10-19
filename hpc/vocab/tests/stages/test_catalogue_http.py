"""Provider pacing, access refusal and complete-query caching. Author: James Edward Ball."""
import copy
import json
from email.utils import formatdate
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from mc.core import Store, Yielded
from stages import imagery
from stages._catalogue_http import CatalogueSession
from stages._imagery_catalogue import SOURCES, search


class Clock:
    def __init__(self): self.now = 1000.
    def time(self): return self.now
    def sleep(self, delay): self.now += delay


class Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.body, self.headers = status, body or {'features': [], 'links': []}, headers or {}
        self.closed = False
    def close(self): self.closed = True
    def json(self): return self.body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('provider error', response=self)


class Transport:
    def __init__(self, responses, clock): self.responses, self.clock, self.calls = list(responses), clock, []
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def post(self, url, **kw):
        self.calls.append((self.clock.time(), url, copy.deepcopy(kw)))
        result = self.responses.pop(0)
        if isinstance(result, Exception): raise result
        return result
    get = post


@pytest.fixture
def ctx(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[2]/'config.json').read_text())
    cfg['mc'] = {'catalogue_request_interval_s': .5, 'catalogue_http_retries': 2}
    return SimpleNamespace(cfg=cfg, shared=tmp_path/'shared', store=Store(tmp_path/'run'))


def wrapped(ctx, transport, clock):
    return CatalogueSession(ctx, transport, clock=clock.time, sleep=clock.sleep)


def test_workers_share_provider_budget(ctx):
    clock = Clock(); transport = Transport([Response() for _ in range(3)], clock)
    a, b = wrapped(ctx, transport, clock), wrapped(ctx, transport, clock)
    url = SOURCES['earth_search'][0]+'/search'
    a.post(url); b.post(url); a.post(url)
    assert [x[0] for x in transport.calls] == [1000., 1000.5, 1001.]


@pytest.mark.parametrize('header', ['30', formatdate(1030., usegmt=True)])
def test_retry_after_applies_to_retry_and_other_workers(ctx, header):
    clock = Clock(); transport = Transport([Response(429, headers={'Retry-After': header}), Response(), Response()], clock)
    a, b = wrapped(ctx, transport, clock), wrapped(ctx, transport, clock)
    url = SOURCES['earth_search'][0]+'/search'
    a.post(url); b.post(url)
    assert transport.calls[1][0] >= 1030.
    assert transport.calls[2][0]-transport.calls[1][0] >= .5


def test_forbidden_pauses_without_retry_or_empty_catalogue(ctx):
    clock = Clock(); transport = Transport([Response(403), Response()], clock)
    client = wrapped(ctx, transport, clock)
    with pytest.raises(Yielded, match='status=403'):
        client.post(SOURCES['earth_search'][0]+'/search')
    assert ctx.store.flag('PAUSE') and len(transport.calls) == 1
    with pytest.raises(Yielded):
        wrapped(ctx, transport, clock).post(SOURCES['planetary_computer'][0]+'/search')
    assert len(transport.calls) == 1


def test_exhausted_transport_errors_pause_without_losing_cause(ctx):
    clock = Clock(); transport = Transport([requests.Timeout() for _ in range(3)], clock)
    with pytest.raises(Yielded, match='kind=Timeout'):
        wrapped(ctx, transport, clock).post(SOURCES['earth_search'][0]+'/search')
    assert len(transport.calls) == 3 and ctx.store.flag('PAUSE')


def test_full_pagination_and_query_are_preserved(ctx):
    clock = Clock(); url = SOURCES['earth_search'][0]+'/search'
    a = {'id':'a','properties':{'eo:cloud_cover':10,'s2:snow_ice_percentage':0,'datetime':'2025-06-01'}}
    b = {**a, 'id':'b'}
    transport = Transport([Response(body={'features':[a], 'links':[{'rel':'next','href':url,'method':'POST','merge':True,'body':{'token':'page2'}}]}), Response(body={'features':[b],'links':[]})], clock)
    result = search(wrapped(ctx,transport,clock), 'earth_search', [17,59,18,60], '2025-05-01/2025-09-30', ctx.cfg)
    assert [r['id'] for r in result] == ['a','b']
    first, second = [c[2]['json'] for c in transport.calls]
    assert first['limit'] == 100 and second == {**first, 'token':'page2'}


def test_complete_source_cache_survives_fallback_refusal(ctx, monkeypatch):
    clock = Clock(); pc = {'id':'pc','properties':{'eo:cloud_cover':0,'s2:snow_ice_percentage':0,'datetime':'2025-06-01'}}
    transport = Transport([Response(body={'features':[pc],'links':[]}), Response(403), Response()], clock)
    ctx.requests_session = lambda **kw: transport
    import stages._catalogue_http as module
    original = module.CatalogueSession
    monkeypatch.setattr(module, 'CatalogueSession', lambda ctx, session, stop: original(ctx,session,stop,clock.time,clock.sleep))
    with pytest.raises(Yielded): imagery.fetch_catalogue(ctx,'x0197y0149','north')
    assert not imagery.catalogue_path(ctx,'x0197y0149','north').exists()
    ctx.store.set_flag('PAUSE', False); clock.now += 301
    doc, path = imagery.fetch_catalogue(ctx,'x0197y0149','north')
    assert path.exists() and doc['sources']['planetary_computer'] == [pc]
    assert doc['sources']['earth_search'] == []  # only after the provider returned a valid empty response
    assert len([c for c in transport.calls if 'planetarycomputer' in c[1]]) == 1
    assert len(transport.calls) == 3
