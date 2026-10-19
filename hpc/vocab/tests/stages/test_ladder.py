"""Encoder-ladder additions (A4): weights plan, DOFA reconstruction, RGB route, adapter guards."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from mc.core import load_config, atomic_write_json
from stages import weights
from stages.extract import route_pixels
from stages._extract_models import arch_of, LADDER
from stages import _dofa

CODE = Path(__file__).resolve().parents[2]


def test_weights_plan_follows_registered_routes():
    cfg = load_config(CODE)
    units = weights.plan(SimpleNamespace(cfg=cfg))
    models = {u['model'] for u in units}
    assert models == {'prithvi_tl', 'dofa', 'dinov2', 'remoteclip'}
    assert len({u['id'] for u in units}) == len(units)
    pinned = [u for u in units if u['sha256']]
    assert pinned and all(u['model'] == 'remoteclip' for u in pinned)
    cfg['extraction']['routes'] = [{'route': 'r1', 'models': ['clay', 'random_dofa_202']}]
    assert {u['model'] for u in weights.plan(SimpleNamespace(cfg=cfg))} == {'dofa'}


def test_weights_cached_file_is_reused_and_pin_mismatch_fails(tmp_path):
    ctx = SimpleNamespace(shared=tmp_path, requests_session=lambda: pytest.fail('network must not be used'))
    unit = {'id': 'dofa.x.bin', 'model': 'dofa', 'file': 'x.bin', 'url': 'https://example.invalid/x', 'sha256': None}
    dest = tmp_path / 'weights/dofa/x.bin'; dest.parent.mkdir(parents=True)
    dest.write_bytes(b'abc')
    import hashlib
    atomic_write_json(dest.with_suffix('.bin.json'), {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()})
    out = weights.run(ctx, unit, SimpleNamespace(maybe_stop=lambda: None))
    assert out['metrics']['cached'] and dest in out['outputs']
    unit['sha256'] = '0' * 64
    with pytest.raises(Exception, match='declared pin'):
        weights.run(ctx, unit, SimpleNamespace(maybe_stop=lambda: None))


def test_arch_names_and_rungs():
    assert arch_of('random_prithvi100_303') == 'prithvi100' and arch_of('random_dofa_101') == 'dofa'
    assert arch_of('prithvi_tl') == 'prithvi_tl' and arch_of('clay') == 'clay'
    assert LADDER['dofa'] == '1b' and LADDER['prithvi_tl'] == 2 and LADDER['remoteclip'] == 4 and LADDER['prithvi'] == 1


def test_rgb_reference_route_is_display_converted_and_224():
    a = np.full((2, 4, 256, 256), 1000 + 4000, dtype='uint16')     # reflectance 0.4 -> display 1.0
    b = np.full((2, 6, 128, 128), 1000, dtype='uint16')
    a[:, 2] = 1000 + 1000                                            # B04 reflectance 0.1
    x = route_pixels(a, b, [-1000, -1000], [1e-4, 1e-4], 'r1rgb224')
    assert x.shape == (2, 3, 224, 224) and x.dtype == np.float32
    assert np.all((x >= 0) & (x <= 1))
    np.testing.assert_allclose(x[0, 0, 0, 0], round(255 * (0.1 / 0.4) ** (1 / 2.2)) / 255, atol=1e-6)   # red = B04
    np.testing.assert_allclose(x[0, 1:, 0, 0], 1.0, atol=1e-6)


def test_dofa_reconstruction_tokens_and_seeded_random_init():
    torch.manual_seed(101)
    small = _dofa.build(embed_dim=32, depth=1, num_heads=2)
    waves = torch.tensor([_dofa.WAVELENGTHS_UM[b] for b in ('B02', 'B03', 'B04', 'B8A', 'B11', 'B12')])
    x = torch.rand(2, 6, 128, 128)
    t = small.forward_features(x, waves)
    assert t.shape == (2, 1 + 8 * 8, 32)          # padded stride-16 conv: 8x8 tokens at 128 px
    t224 = small.forward_features(torch.rand(1, 6, 224, 224), waves)
    assert t224.shape == (1, 1 + 14 * 14, 32)
    assert torch.isfinite(t).all()
    # The checkpoint's positional table is a fixed 2-D sin-cos grid with a zero class row.
    assert small.pos_embed.shape == (1, 197, 32) and torch.allclose(small.pos_embed[0, 0], torch.zeros(32))
    torch.manual_seed(101); a = _dofa.build(embed_dim=32, depth=1, num_heads=2)
    torch.manual_seed(101); b = _dofa.build(embed_dim=32, depth=1, num_heads=2)
    torch.manual_seed(202); c = _dofa.build(embed_dim=32, depth=1, num_heads=2)
    assert torch.equal(a.blocks[0].attn.qkv.weight, b.blocks[0].attn.qkv.weight)
    assert not torch.equal(a.blocks[0].attn.qkv.weight, c.blocks[0].attn.qkv.weight)
    # Loading tolerates only the decoder projector and mask token as extras.
    state = {k: v for k, v in small.state_dict().items()}
    state['projector.weight'] = torch.zeros(32, 32); state['mask_token'] = torch.zeros(1, 1, 16)
    assert _dofa.load_encoder(small, state) == ['mask_token', 'projector.weight']
    with pytest.raises(ValueError, match='mismatch'):
        _dofa.load_encoder(small, {k: v for k, v in state.items() if 'blocks.0.norm1' not in k})


def test_dofa_checkpoint_shapes_match_reconstruction():
    """Shapes recorded from DOFA_ViT_base_e100.pth on 17 Oct 2026 (the file itself stays on the HPC)."""
    m = _dofa.build()
    sd = m.state_dict()
    assert tuple(sd['pos_embed'].shape) == (1, 197, 768)
    assert tuple(sd['patch_embed.weight_generator.fc_weight.weight'].shape) == (16 * 16 * 768, 128)
    assert tuple(sd['patch_embed.weight_generator.weight_tokens'].shape) == (128, 128)
    assert tuple(sd['blocks.0.attn.qkv.weight'].shape) == (2304, 768) and 'blocks.0.attn.qkv.bias' in sd
    assert tuple(sd['blocks.11.mlp.fc2.weight'].shape) == (768, 3072) and 'blocks.12.norm1.weight' not in sd
    assert 'norm.weight' in sd and 'fc_norm.weight' not in sd


class _Resp:
    def __init__(self, status, body, headers):
        self.status_code, self._body, self.headers = status, body, headers
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)
    def iter_content(self, n):
        for i in range(0, len(self._body), 2): yield self._body[i:i+2]


class _Session:
    def __init__(self, payload, honour_range=True):
        self.payload, self.honour_range, self.calls = payload, honour_range, []
    def get(self, url, headers=None, **kw):
        self.calls.append(dict(headers or {}))
        rng = (headers or {}).get('Range')
        if rng and self.honour_range:
            start = int(rng.split('=')[1].rstrip('-'))
            return _Resp(206, self.payload[start:], {'Content-Range': f'bytes {start}-{len(self.payload)-1}/{len(self.payload)}',
                                                     'x-repo-commit': 'abc'})
        return _Resp(200, self.payload, {'Content-Length': str(len(self.payload))})
    def close(self): pass


def test_weight_download_resumes_partial_file_and_hashes_whole(tmp_path):
    payload = bytes(range(64)) * 4
    dest = tmp_path / 'w.pt'; dest.parent.mkdir(exist_ok=True)
    (tmp_path / 'w.pt.part').write_bytes(payload[:100])
    s = _Session(payload)
    rec = weights.download(SimpleNamespace(), {'id': 'm.w', 'url': 'u', 'model': 'dofa'}, dest, SimpleNamespace(maybe_stop=lambda: None), session=s)
    import hashlib
    assert rec['bytes'] == len(payload) and rec['sha256'] == hashlib.sha256(payload).hexdigest()
    assert rec['resumed_from'] == 100 and s.calls[0]['Range'] == 'bytes=100-' and rec['repo_commit'] == 'abc'
    assert dest.read_bytes() == payload and not (tmp_path / 'w.pt.part').exists()


def test_weight_download_restarts_when_server_ignores_range(tmp_path):
    payload = b'z' * 50
    dest = tmp_path / 'w.pt'
    (tmp_path / 'w.pt.part').write_bytes(b'q' * 10)
    rec = weights.download(SimpleNamespace(), {'id': 'm.w', 'url': 'u', 'model': 'dofa'}, dest, SimpleNamespace(maybe_stop=lambda: None), session=_Session(payload, honour_range=False))
    assert rec['resumed_from'] == 0 and dest.read_bytes() == payload


def test_weight_download_refuses_runaway_body(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, 'MAX_BYTES', 10)
    with pytest.raises(Exception, match='ceiling'):
        weights.download(SimpleNamespace(), {'id': 'm.w', 'url': 'u', 'model': 'dofa'}, tmp_path / 'w.pt', SimpleNamespace(maybe_stop=lambda: None), session=_Session(b'x' * 11))


def test_governor_run_state_isolates_failed_branches():
    from mc.governor import run_state
    rep = {'stages': {'a': {'state': 'failed'}, 'b': {'state': 'dep_failed'}, 'c': {'state': 'ready'}}, 'owned_jobs': [], 'actions': []}
    v = run_state(rep, 0, 3)
    assert v['exit'] is None and v['failed'] == ['a', 'b']          # c can still run
    rep['stages']['c'] = {'state': 'complete'}
    idle = 0
    for _ in range(2):
        v = run_state(rep, idle, 3); idle = v['idle_cycles']; assert v['exit'] is None
    assert run_state(rep, idle, 3)['exit'] == 2                     # nothing left to do, something failed
    done = {'stages': {'a': {'state': 'complete'}}, 'owned_jobs': [], 'actions': []}
    assert run_state(done, 0, 3)['exit'] == 0
    gated = {'stages': {'a': {'state': 'complete'}, 'rel': {'state': 'gated'}}, 'owned_jobs': [], 'actions': []}
    assert run_state(gated, 2, 3)['exit'] == 0                       # waits on a human gate, nothing running
    running = {'stages': {'a': {'state': 'blocked'}}, 'owned_jobs': [1], 'actions': []}
    assert run_state(running, 99, 3)['exit'] is None and run_state(running, 99, 3)['idle_cycles'] == 0


def test_required_models_are_the_registered_architectures():
    from stages.extract import required_models
    cfg = load_config(CODE)
    req = required_models(cfg)
    assert {'r1-clay', 'r1-prithvi', 'r1-prithvi100', 'r1-random_clay_101', 'r1-random_prithvi100_303'} <= req
    assert not any(k in req for k in ('r1-dofa', 'r1-prithvi_tl', 'r1rgb224-dinov2', 'r2-clay'))


def test_ladder_config_is_closed_and_consistent():
    cfg = load_config(CODE)
    routes = {f"{r['route']}-{m}" for r in cfg['extraction']['routes'] for m in r['models']}
    ladder = cfg['analysis']['ladder']
    listed = {x for v in ladder.values() for x in v}
    assert listed <= routes, listed - routes
    assert 'r1-dofa' in ladder['rung1b_eo_imagery_natural_image_init'] and 'r1-dofa' not in ladder['rung1_imagery_only']
    assert set(ladder['rung3_natural_image']) == {'r1rgb224-dinov2', 'r1rgb224-dofa'}
    assert cfg['analysis']['matched_contrasts'] == [['r1-prithvi', 'r1-prithvi_tl']]
    assert all(f'r1-random_{a}_{s}' in ladder['rung0_random'] for a in ('clay', 'prithvi', 'prithvi100', 'dofa') for s in (101, 202, 303))
    bs = cfg['extraction']['batch_size']
    assert all(m.split('_')[1] if m.startswith('random_') else m in bs[r['route']] for r in cfg['extraction']['routes'] for m in r['models'])
