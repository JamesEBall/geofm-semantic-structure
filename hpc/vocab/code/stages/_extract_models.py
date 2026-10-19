"""Pinned model adapters extended from the validated fresh-data extractor.
Author: James Edward Ball. Model imports occur only inside build_adapter.

Registered encoders (amendment A4 ladder):
  rung 1  prithvi, prithvi100                imagery only: no sample metadata, no teacher, no language
  rung 1b dofa                              EO imagery, wavelength as sensor constant; natural-image initialisation/teacher to verify
  rung 2  clay, prithvi_tl                   imagery plus metadata conditioning (Clay also a DINOv2 teacher)
  rung 3  dinov2                             natural-image self-supervision, RGB display conversion
  rung 4  remoteclip                         explicit image-text alignment, RGB; positive reference only
  rung 0  random_<arch>_<seed>               untrained architectures, seeds 101/202/303
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

LADDER = {'prithvi': 1, 'prithvi100': 1, 'dofa': '1b', 'clay': 2, 'prithvi_tl': 2, 'dinov2': 3, 'remoteclip': 4}
RGB_ARCHS = ('dinov2', 'remoteclip')
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_features(mean, cls, n):
    import numpy as np
    if mean.shape != cls.shape or mean.ndim != 2 or mean.shape[0] != n:
        raise ValueError('Invalid feature shapes')
    for x in (mean, cls):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Non-finite or non-float32 features')
        norms = np.linalg.norm(x, axis=1)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('Invalid feature norms')


def device_safe_position_encoding(encoder):
    """Keep regenerated positional tensors on the encoder's existing device."""
    original = encoder.interpolate_pos_encoding
    def interpolate(*args, **kwargs):
        return original(*args, **kwargs).to(device=encoder.pos_embed.device,
                                           dtype=encoder.pos_embed.dtype)
    encoder.interpolate_pos_encoding = interpolate


def arch_of(name):
    return name.split('_', 1)[1].rsplit('_', 1)[0] if name.startswith('random_') else name


def _prithvi_module(src):
    spec = importlib.util.spec_from_file_location('prithvi_pinned_' + src.name, src / 'prithvi_mae.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_adapter(root, name, device, pixels=128, clay_gsd=10., ten_bands=False, weights=None,
                  band_stats=None, rgb=False):
    """Return (model, forward, provenance). `forward(raw)` takes float32 route pixels (reflectance, or
    [0,1] RGB for `rgb` routes) and returns (mean-token, cls-token) float32 arrays."""
    import torch
    import numpy as np
    import yaml
    sys.path.insert(0, str(root / 'package/src'))
    from study.models import WEIGHTS
    random = name.startswith('random_')
    arch = arch_of(name)
    if random:
        seed = int(name.rsplit('_', 1)[1])
        if seed not in (101, 202, 303):
            raise ValueError('Random control seed must be 101, 202 or 303')
        torch.manual_seed(seed)
    if (arch in RGB_ARCHS) and not rgb:
        raise ValueError(f'{name}: RGB reference encoders run only on the RGB display route')
    if rgb and arch not in RGB_ARCHS + ('dofa',):
        raise ValueError(f'{name}: only the RGB references and DOFA run on the RGB display route')
    weights = Path(weights) if weights is not None else root / 'weights'
    weight = root / 'weights' / (arch + '.pt')
    checkpoint_sha = None
    weight_files = {}
    if not random and arch in ('clay', 'prithvi'):
        checkpoint_sha = sha(weight)
        if checkpoint_sha != WEIGHTS[arch][1]:
            raise ValueError(f'Checkpoint hash mismatch: {weight}')
    domain = 'digital_number'      # how `prep` expects reflectance to be scaled before normalisation
    if arch == 'clay':
        from clay_probe.vendor.model import Encoder
        kw = dict(mask_ratio=0., patch_size=8, shuffle=False, dim=1024,
                  depth=24, heads=16, dim_head=64, mlp_ratio=4)
        if random:
            model = Encoder(**kw)
        else:
            with torch.device('meta'):
                model = Encoder(**kw)
            state = torch.load(weight, weights_only=True, mmap=True, map_location='cpu')['state_dict']
            prefix = 'model.encoder.'
            model.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                   if k.startswith(prefix)}, strict=True, assign=True)
        meta_path = root / 'package/src/clay_probe/vendor/metadata.yaml'
        sensor = yaml.safe_load(meta_path.read_text())['sentinel-2-l2a']['bands']
        bands = (['blue', 'green', 'red', 'nir', 'rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22'] if ten_bands else ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22'])
        means = [sensor['mean'][b] for b in bands]
        stds = [sensor['std'][b] for b in bands]
        waves = torch.tensor([sensor['wavelength'][b] for b in bands], device=device)
        # Pinned Clay constructs spatial positional encoding on CPU internally.
        # Keep scalar GSD on CPU to avoid a needless GPU-to-CPU synchronisation.
        gsd = torch.tensor(clay_gsd, dtype=torch.float32)
        def tokens(x):
            b = len(x)
            return model(dict(pixels=x, time=torch.zeros(b, 4, device=device),
                              latlon=torch.zeros(b, 4, device=device), gsd=gsd,
                              waves=waves))[0]
        src = root / 'package/src/clay_probe/vendor'
    elif arch in ('prithvi', 'prithvi_tl'):
        src = root / 'package/vendor/prithvi_eo_300m'
        module = _prithvi_module(src)
        if arch == 'prithvi':
            cfg = json.loads((src / 'config.json').read_text())['pretrained_cfg']
        else:
            # Same architecture, data and objective as `prithvi`, pretrained WITH time and location
            # conditioning (ladder rung 2). Coordinates are withheld at evaluation, as Clay's are zeroed.
            weight = weights / 'prithvi_tl' / 'Prithvi_EO_V2_300M_TL.pt'
            raw_cfg = json.loads((weights / 'prithvi_tl' / 'config.json').read_text())
            cfg = raw_cfg.get('pretrained_cfg', raw_cfg)
            if not set(cfg.get('coords_encoding') or []) >= {'time', 'location'}:
                raise ValueError('Prithvi-TL config does not declare time and location encodings')
        model = module.PrithviMAE(**cfg)
        if not random:
            if arch == 'prithvi_tl':
                checkpoint_sha = sha(weight); weight_files[weight.name] = checkpoint_sha
            model.load_state_dict(torch.load(weight, weights_only=True, mmap=True,
                                             map_location='cpu'), strict=True)
        # Upstream regenerates resized position embeddings from NumPy on CPU.
        device_safe_position_encoding(model.encoder)
        means, stds = cfg['mean'], cfg['std']
        def tokens(x):
            return model.forward_features(x[:, :, None])[-1]
    elif arch == 'prithvi100':
        src = root / 'weights/prithvi100'
        manifest100 = json.loads((root / 'package/data/prithvi100/model.json').read_text())
        for entry in manifest100['files']:
            if random and entry['name'].endswith('.pt'):
                continue
            if sha(src / entry['name']) != entry['sha256']:
                raise ValueError('Prithvi100 source/checkpoint hash mismatch: ' + entry['name'])
        module = _prithvi_module(src)
        cfg = json.loads((src / 'config.json').read_text())['pretrained_cfg']
        cfg.update(img_size=pixels, num_frames=1)
        model = module.PrithviViT(**cfg)
        if model.pos_embed.shape != (1, (pixels // 16) ** 2 + 1, 768):
            raise ValueError('Prithvi100 positional embedding shape mismatch')
        if not random:
            weight = src / 'Prithvi_EO_V1_100M.pt'
            checkpoint_sha = sha(weight)
            state = torch.load(weight, weights_only=True, mmap=True, map_location='cpu')
            enc = {k.removeprefix('encoder.'): v for k, v in state.items()
                   if k.startswith('encoder.') and 'pos_embed' not in k}
            if set(enc) != set(dict(model.named_parameters())):
                raise ValueError('Prithvi100 parameter mismatch')
            loaded = model.load_state_dict(enc, strict=False)
            if loaded.missing_keys != ['pos_embed'] or loaded.unexpected_keys:
                raise ValueError('Prithvi100 checkpoint load mismatch')
        means, stds = cfg['mean'], cfg['std']
        domain = 'reflectance'
        def tokens(x):
            return model.forward_features(x[:, :, None])[-1]
    elif arch == 'dofa':
        from ._dofa import build, load_encoder, WAVELENGTHS_UM
        if ten_bands:
            raise ValueError('DOFA is registered for the six-band routes only')
        if not rgb and (not band_stats or len(band_stats['mean']) != 6):
            raise ValueError('DOFA needs the six training-split band statistics from features_fit')
        model = build()
        src = Path(__file__).resolve().parent / '_dofa'
        extras = []
        if not random:
            weight = weights / 'dofa' / 'DOFA_ViT_base_e100.pth'
            checkpoint_sha = sha(weight); weight_files[weight.name] = checkpoint_sha
            extras = load_encoder(model, torch.load(weight, weights_only=True, map_location='cpu'))
        if rgb:
            # Same-input comparator for the natural-image rungs: the three display bands, ImageNet statistics.
            waves = torch.tensor([WAVELENGTHS_UM[b] for b in ('B04', 'B03', 'B02')], dtype=torch.float32, device=device)
            means, stds = list(IMAGENET_MEAN), list(IMAGENET_STD)
            domain = 'rgb_unit'
        else:
            waves = torch.tensor([WAVELENGTHS_UM[b] for b in ('B02', 'B03', 'B04', 'B8A', 'B11', 'B12')],
                                 dtype=torch.float32, device=device)
            means, stds = list(band_stats['mean']), list(band_stats['std'])
            domain = 'reflectance'
        def tokens(x):
            return model.forward_features(x, waves)
    elif arch == 'dinov2':
        from transformers import Dinov2Config, Dinov2Model
        src = weights / 'dinov2'
        if random:
            model = Dinov2Model(Dinov2Config.from_pretrained(src, local_files_only=True))
        else:
            model = Dinov2Model.from_pretrained(src, local_files_only=True, torch_dtype=torch.float32,
                                                attn_implementation='eager')
            for f in sorted(src.glob('*')):
                if f.is_file() and f.suffix in ('.safetensors', '.bin', '.pt', '.pth'):
                    weight_files[f.name] = sha(f)
            checkpoint_sha = weight_files.get('model.safetensors')
        means, stds = list(IMAGENET_MEAN), list(IMAGENET_STD)
        domain = 'rgb_unit'
        def tokens(x):
            return model(pixel_values=x).last_hidden_state
    elif arch == 'remoteclip':
        import open_clip
        src = weights / 'remoteclip'
        weight = src / 'RemoteCLIP-ViT-B-32.pt'
        # open_clip 2.32: an unpretrained ViT-B-32 skeleton (no download), GELU as in the published recipe;
        # the pinned RemoteCLIP state dict is then loaded strictly.
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, force_quick_gelu=False,
                                                            device='cpu', precision='fp32')
        if not random:
            checkpoint_sha = sha(weight); weight_files[weight.name] = checkpoint_sha
            if checkpoint_sha != WEIGHTS['remoteclip'][1]:
                raise ValueError('RemoteCLIP checkpoint hash mismatch')
            model.load_state_dict(torch.load(weight, weights_only=True, mmap=True, map_location='cpu'), strict=True)
        means, stds = list(CLIP_MEAN), list(CLIP_STD)
        domain = 'rgb_unit'
        def tokens(x):
            # Pooled image embedding; mean and cls outputs coincide, as in the published expert study.
            y = model.encode_image(x)
            return torch.stack((y, y), dim=1)
    else:
        raise ValueError(f'Unsupported architecture: {arch}')
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    mean = torch.tensor(means, device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(stds, device=device, dtype=torch.float32)[None, :, None, None]
    def prep(x):
        if domain == 'digital_number':        # Clay, Prithvi 300M: checkpoint statistics in DN units
            return (x * 10000. - mean) / std
        if domain == 'reflectance':           # Prithvi 100M (DN stats / 10000), DOFA (training-split stats)
            return (x - mean / (10000. if arch == 'prithvi100' else 1.)) / (std / (10000. if arch == 'prithvi100' else 1.))
        return (x - mean) / std               # RGB references on [0, 1] display values
    def forward(raw):
        x = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))
        if str(device).startswith('cuda'):
            x = x.pin_memory()
        x = x.to(device, non_blocking=True)
        t = tokens(prep(x))
        return (t[:, 1:].mean(1).cpu().numpy(), t[:, 0].cpu().numpy())
    bands = (['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12'] if ten_bands else
             ['B04','B03','B02'] if rgb else ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'])
    provenance = dict(checkpoint_sha256=checkpoint_sha, model=name, architecture=arch, ladder_rung=0 if random else LADDER[arch],
                      precision='float32', normalisation_mean=means, normalisation_std=stds, input_bands=bands,
                      input_pixels=pixels, reflectance_multiplier=10000. if domain == 'digital_number' else None,
                      clay_gsd=clay_gsd if arch == 'clay' else None,
                      metadata={'clay': 'zero time and location', 'prithvi_tl': 'time and location withheld (None)'}.get(arch, 'none'),
                      pooling=['mean_non_cls_final_tokens', 'cls_final_token'] if arch != 'remoteclip' else ['pooled_image_embedding'] * 2,
                      normalisation_order={'digital_number': 'digital_number_domain', 'reflectance': 'reflectance_domain',
                                           'rgb_unit': 'rgb_display_domain'}[domain],
                      band_statistics_source='features_fit train split' if (arch == 'dofa' and not rgb) else 'imagenet' if rgb else 'checkpoint',
                      positional_table='bicubic resample of the 14x14 table; padded stride-16 conv (upstream)' if arch == 'dofa' else None,
                      weight_files=weight_files,
                      sources={str(p.relative_to(root)) if str(p).startswith(str(root)) else p.name: sha(p)
                               for p in sorted(Path(src).rglob('*')) if p.suffix in ('.py', '.json', '.yaml')})
    return model, forward, provenance
