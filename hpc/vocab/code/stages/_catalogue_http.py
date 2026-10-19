"""Shared catalogue request pacing and explicit access-failure handling.

Author: James Edward Ball. These controls change request timing only.
"""
from __future__ import annotations

import fcntl
import random
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from mc.core import Yielded, atomic_write_json, read_json


class CatalogueSession:
    def __init__(self, ctx, session, stop=lambda: None, clock=None, sleep=None):
        self.ctx, self.session, self.stop = ctx, session, stop
        self.clock, self.sleep = clock or time.time, sleep or time.sleep
        settings = ctx.cfg.get('mc', {})
        self.interval = max(.5, float(settings.get('catalogue_request_interval_s', .5)))
        self.retries = int(settings.get('catalogue_http_retries', 5))
        self.directory = ctx.shared / 'catalogue_http'
        self.directory.mkdir(parents=True, exist_ok=True)

    def _state(self, host, update):
        with (self.directory / (host+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / (host+'.json')
            state = read_json(path, {})
            result = update(state)
            atomic_write_json(path, state)
            return result

    def _wait(self, host):
        while True:
            self.stop()
            if self.ctx.store.flag('PAUSE'):
                raise Yielded('Catalogue requests paused')
            def reserve(state):
                now = self.clock()
                delay = max(state.get('next_at', 0), state.get('blocked_until', 0)) - now
                if delay <= 0:
                    state['next_at'] = now + self.interval
                return delay
            delay = self._state(host, reserve)
            if delay <= 0:
                return
            self.sleep(min(delay, 1.))

    def _cooldown(self, host, delay):
        def update(state):
            state['blocked_until'] = max(state.get('blocked_until', 0), self.clock()+delay)
        self._state(host, update)

    def _pause(self, host, status, kind):
        self._cooldown(host, 300.)
        note = f'Catalogue access requires review: provider={host}, status={status}, kind={kind}. Sources and quality gates preserved.'
        if not self.ctx.store.flag('PAUSE'):
            self.ctx.store.set_flag('PAUSE', note=note)
        self.ctx.store.event('catalogue_access_pause', provider=host, status=status, error_kind=kind)
        raise Yielded(note)

    def _retry_after(self, response):
        value = response.headers.get('Retry-After')
        if value is None:
            return 0.
        try:
            return max(0., float(value))
        except ValueError:
            try:
                return max(0., parsedate_to_datetime(value).timestamp()-self.clock())
            except (TypeError, ValueError, OverflowError):
                return 0.

    def _request(self, method, url, **kwargs):
        import requests
        from ._imagery_catalogue import SOURCES
        host = urlsplit(url).netloc
        if urlsplit(url).scheme != 'https' or host not in {urlsplit(v[0]).netloc for v in SOURCES.values()}:
            raise ValueError('Unregistered catalogue host')
        for attempt in range(self.retries+1):
            self._wait(host)
            response = None
            try:
                requested_at = self.clock()
                response = getattr(self.session, method)(url, **kwargs)
                status, kind = response.status_code, 'HTTP'
                self.ctx.store.event('catalogue_http_response', provider=host, status=status,
                                     requested_at=requested_at, elapsed_s=self.clock()-requested_at)
                if status in (401, 403):
                    response.close()
                    self._pause(host, status, kind)
                if status not in (429, 500, 502, 503, 504):
                    return response
                delay = max(self._retry_after(response), 2**(attempt+1)+random.random())
                response.close()
            except (requests.ConnectionError, requests.Timeout) as exc:
                status, kind = None, type(exc).__name__
                delay = 2**(attempt+1)+random.random()
            # Store only provider, status and timing: response bodies and URLs may carry secrets.
            self.ctx.store.event('catalogue_http_retry', provider=host, status=status,
                                 error_kind=kind, attempt=attempt+1, delay_s=delay)
            self._cooldown(host, delay)
            if attempt == self.retries:
                self._pause(host, status, kind)
        raise AssertionError('Unreachable catalogue retry state')

    def post(self, url, **kwargs):
        return self._request('post', url, **kwargs)

    def get(self, url, **kwargs):
        return self._request('get', url, **kwargs)
