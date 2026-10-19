"""DOFA (Dynamic One-For-All) ViT-B/16 encoder, reconstructed for frozen feature extraction.

Author of this adapter: James Edward Ball. The wavelength-conditioned patch embedding is the
vendored upstream module (`wave_dynamic_layer.py`); the transformer trunk is timm's standard ViT
block, which matches the released checkpoint's tensor names (`blocks.N.{norm1,attn.qkv,attn.proj,
norm2,mlp.fc1,mlp.fc2}`, `norm`, `cls_token`, `pos_embed`). The decoder `projector` and the MAE
`mask_token` are not part of the encoder and are ignored on load.
"""
from __future__ import annotations
from functools import partial

# Sentinel-2A central wavelengths in micrometres, as DOFA expects (it multiplies by 1000 internally).
WAVELENGTHS_UM = {'B02': 0.490, 'B03': 0.560, 'B04': 0.665, 'B05': 0.705, 'B06': 0.740, 'B07': 0.783,
                  'B08': 0.842, 'B8A': 0.865, 'B11': 1.610, 'B12': 2.190}
ENCODER_KEYS_IGNORED = ('projector.weight', 'projector.bias', 'mask_token')


def build(embed_dim=768, depth=12, num_heads=12, img_size=224, patch_size=16, wv_planes=128):
    import torch
    import torch.nn as nn
    from timm.models.vision_transformer import Block
    from .wave_dynamic_layer import Dynamic_MLP_OFA

    class OFAViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.grid = img_size // patch_size
            self.patch_embed = Dynamic_MLP_OFA(wv_planes=wv_planes, inter_dim=128, kernel_size=patch_size,
                                               embed_dim=embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, embed_dim), requires_grad=False)
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads, 4.0, qkv_bias=True, norm_layer=norm_layer)
                                         for _ in range(depth)])
            self.norm = norm_layer(embed_dim)
            nn.init.normal_(self.cls_token, std=0.02)
            # Fixed 2-D sin-cos positional table, as in the upstream constructor.
            self.pos_embed.data.copy_(sincos_2d(embed_dim, self.grid, cls=True))

        def pos_for(self, h, w):
            """Resample the 14x14 table to the token grid of the current input (bicubic, no cls)."""
            import torch.nn.functional as F
            cls, grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
            if (h, w) == (self.grid, self.grid):
                return cls, grid
            g = grid.reshape(1, self.grid, self.grid, self.embed_dim).permute(0, 3, 1, 2)
            g = F.interpolate(g, size=(h, w), mode='bicubic', align_corners=False, antialias=False)
            return cls, g.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)

        def forward_features(self, x, waves):
            b, c, hh, ww = x.shape
            tokens, _ = self.patch_embed(x, waves)              # conv stride 16, padding 1
            h = (hh + 2 - self.patch_size) // self.patch_size + 1
            w = (ww + 2 - self.patch_size) // self.patch_size + 1
            if tokens.shape[1] != h * w:
                raise ValueError('DOFA token grid differs from the expected padded-conv size')
            cls_pos, grid_pos = self.pos_for(h, w)
            x = tokens + grid_pos
            x = torch.cat(((self.cls_token + cls_pos).expand(b, -1, -1).to(x.dtype), x), dim=1)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    return OFAViT()


def sincos_2d(embed_dim, grid, cls=False):
    import numpy as np
    import torch
    gh = np.arange(grid, dtype=np.float32)
    gw = np.arange(grid, dtype=np.float32)
    g = np.meshgrid(gw, gh)
    g = np.stack(g, axis=0).reshape(2, 1, grid, grid)
    def one(d, pos):
        omega = np.arange(d // 2, dtype=np.float64) / (d / 2.0)
        omega = 1.0 / 10000 ** omega
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([one(embed_dim // 2, g[0]), one(embed_dim // 2, g[1])], axis=1)
    if cls:
        emb = np.concatenate([np.zeros((1, embed_dim)), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


def load_encoder(model, state):
    """Strict for every encoder tensor; the decoder projector and mask token are the only extras."""
    extra = {k for k in state if k in ENCODER_KEYS_IGNORED}
    state = {k: v for k, v in state.items() if k not in ENCODER_KEYS_IGNORED}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f'DOFA checkpoint mismatch: missing={result.missing_keys[:5]} unexpected={result.unexpected_keys[:5]}')
    return sorted(extra)
