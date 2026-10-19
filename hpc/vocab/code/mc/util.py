"""Small helpers shared by stages (stdlib-only at import time)."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def seed_for(cfg: dict, *parts: object) -> int:
    """Deterministic 32-bit seed from the registered seed and any keys (stage, unit, fold, ...)."""
    h = hashlib.sha256("|".join([str(cfg["seed"]), *map(str, parts)]).encode()).digest()
    return int.from_bytes(h[:4], "little")


def lfs_stripe(path: os.PathLike | str, count: int = -1, size: str = "4M") -> bool:
    """Stripe a (new, empty) directory across OSTs so big files inside it spread IO. No-op off Lustre."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    lfs = shutil.which("lfs")
    if not lfs:
        return False
    r = subprocess.run([lfs, "setstripe", "-c", str(count), "-S", size, str(p)],
                       capture_output=True, text=True)
    return r.returncode == 0


def shard_of(key: str, n: int) -> int:
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % n


def chunks(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]
