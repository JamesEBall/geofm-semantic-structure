"""Asset downloads must follow a rename or fresh clone. Author: James Edward Ball."""
import subprocess
from types import SimpleNamespace
import pytest
from study import releases


@pytest.mark.parametrize('origin', ['git@github.com:JamesEBall/old-name.git', 'https://github.com/JamesEBall/old-name.git'])
def test_resolves_current_origin_after_rename(monkeypatch, origin):
    monkeypatch.delenv('GEOFM_ASSET_REPOSITORY', raising=False)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'git':
            return SimpleNamespace(stdout=origin, returncode=0)
        assert kwargs['check'] is True
        return SimpleNamespace(stdout='JamesEBall/renamed-study\n', returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    assert releases.repository() == 'JamesEBall/renamed-study'
    assert calls[1][2] == 'repos/JamesEBall/old-name'


def test_mirror_needs_no_lookup(monkeypatch):
    monkeypatch.setenv('GEOFM_ASSET_REPOSITORY', 'reviewer/mirror')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Unexpected lookup'))
    assert releases.repository() == 'reviewer/mirror'


def test_source_zip_defaults_to_public_repository(monkeypatch):
    monkeypatch.delenv('GEOFM_ASSET_REPOSITORY', raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=128, stdout=''))
    assert releases.repository() == 'JamesEBall/geofm-semantic-structure'


def test_failed_lookup_is_not_silently_redirected(monkeypatch):
    monkeypatch.delenv('GEOFM_ASSET_REPOSITORY', raising=False)
    def run(command, **kwargs):
        if command[0] == 'git':
            return SimpleNamespace(stdout='git@github.com:JamesEBall/private.git', returncode=0)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        releases.repository()
