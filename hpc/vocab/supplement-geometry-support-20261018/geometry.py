"""Production-faithful geometry functions with exact pdist tie handling. Author: James Edward Ball."""

import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist


def normalise(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12) or not np.isfinite(x).all():
        raise ValueError("zero or nonfinite prototype")
    return x / norm


def distance_ranks(x):
    """Use production pdist arithmetic so exact/near ties get identical ranks."""
    x = np.asarray(x)
    normalise(x)  # Validate finite, nonzero prototypes without changing pdist input.
    k, d = x.shape[-2:]
    distances = np.stack([pdist(a, "cosine") for a in x.reshape(-1, k, d)])
    distances = distances.reshape(x.shape[:-2] + (k * (k - 1) // 2,))
    return rankdata(distances, axis=-1)


def correlation(a, b):
    a, b = np.broadcast_arrays(a, b)
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(
        np.sum(a * b, axis=-1), den, out=np.full(den.shape, np.nan), where=den > 1e-10
    )


def geometry_values(baseline, semantic, appearance, geo, text):
    """Point statistics identical to production, batched to avoid Python per-draw overhead.

    SVD is performed on the same intercept plus raw nuisance-rank columns as production.
    Each bootstrap receives its own rebuilt nuisance geometry and its own projection.
    """
    single = np.ndim(baseline) == 2
    if single:
        baseline, semantic, appearance, geo = (
            x[None] for x in (baseline, semantic, appearance, geo)
        )
    rb, rs = distance_ranks(baseline), distance_ranks(semantic)
    ra, rg = distance_ranks(appearance), distance_ranks(geo)
    rt = rankdata(pdist(text, "cosine"))
    ob, os_ = correlation(rb, rt), correlation(rs, rt)
    nuis = np.stack([np.ones_like(ra), ra, rg], axis=-1)
    u, s, _ = np.linalg.svd(nuis, full_matrices=False)
    u = u * (s > s.max(axis=-1, keepdims=True) * 1e-12)[:, None, :]

    def residual(v):
        v = np.broadcast_to(v, rb.shape)
        return v - np.einsum("bmi,bi->bm", u, np.einsum("bmi,bm->bi", u, v))

    tb = residual(rt)
    br, sr = residual(rb), residual(rs)

    # Production partial() correlates projected residuals without an extra centring step.
    def partial(v):
        den = np.linalg.norm(v, axis=-1) * np.linalg.norm(tb, axis=-1)
        return np.divide(
            np.sum(v * tb, axis=-1),
            den,
            out=np.full(den.shape, np.nan),
            where=den > 1e-10,
        )

    pb, ps = partial(br), partial(sr)
    out = np.column_stack([ob, os_, pb, ps, os_ - ob, ps - pb])
    return out[0] if single else out


def wilson(successes, n):
    if not n:
        return {
            "successes": successes,
            "n": n,
            "rate": None,
            "mc_se": None,
            "wilson95": None,
        }
    p, z = successes / n, 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "successes": int(successes),
        "n": int(n),
        "rate": float(p),
        "mc_se": float(np.sqrt(p * (1 - p) / n)),
        "wilson95": [float(centre - half), float(centre + half)],
    }


def band_result(rate, band):
    r = dict(rate)
    r["criterion_band"] = band
    r["point_rate_pass"] = (
        None if rate["rate"] is None else bool(band[0] <= rate["rate"] <= band[1])
    )
    bounds = rate["wilson95"]
    r["mc_interval_entirely_in_band"] = (
        None if bounds is None else bool(band[0] <= bounds[0] and bounds[1] <= band[1])
    )
    return r
