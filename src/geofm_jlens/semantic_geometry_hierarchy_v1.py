"""Finite-inventory geographic geometry. Author: James Edward Ball."""
import itertools
import numpy as np
from scipy.stats import rankdata


def unit(x):
    a = np.asarray(x, dtype=float)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or (norms <= 1e-12).any():
        raise ValueError('Non-finite or zero vector')
    return a / norms


def prototypes(features, membership, parents, components):
    """Patch mean per parent, parent mean per component, then unit component means."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(membership)
    parents, components = np.asarray(parents), np.asarray(components)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Require finite nonempty scene matrix')
    if y.dtype != bool or y.ndim != 2 or y.shape[0] != len(x):
        raise ValueError('Membership must be an aligned boolean matrix')
    if parents.shape != (len(x),) or components.shape != (len(x),):
        raise ValueError('Unaligned hierarchy')
    for parent in np.unique(parents):
        if len(np.unique(components[parents == parent])) != 1:
            raise ValueError('Parent spans components')
    vectors, support = [], []
    for category in range(y.shape[1]):
        mask = y[:, category]
        if not mask.any():
            raise ValueError(f'Unsupported category {category}')
        means, counts = [], []
        for component in np.unique(components[mask]):
            selected = mask & (components == component)
            ps = np.unique(parents[selected])
            parent_means = [x[selected & (parents == p)].mean(0) for p in ps]
            means.append(unit(np.mean(parent_means, axis=0)))
            counts.append({'component': str(component), 'parents': len(ps),
                           'patches': int(selected.sum())})
        vectors.append(unit(np.mean(means, axis=0)))
        support.append({'patches': int(mask.sum()),
                        'parents': len(np.unique(parents[mask])),
                        'components': len(means), 'component_counts': counts})
    return np.stack(vectors), support


def distances(x):
    a = unit(x)
    d = 1 - a @ a.T
    np.fill_diagonal(d, 0.)
    return d


def distance_matrix(d):
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 3:
        raise ValueError('Require square matrix with at least three concepts')
    if not np.isfinite(d).all() or not np.allclose(d, d.T, atol=1e-12, rtol=0):
        raise ValueError('Require finite symmetric distances')
    return d


def centred_rank(v):
    ranks = rankdata(v, method='average')
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm <= 1e-12:
        raise ValueError('Constant distance ranks')
    return ranks / norm


def rsa(a, b):
    a, b = distance_matrix(a), distance_matrix(b)
    if a.shape != b.shape:
        raise ValueError('Distance shape mismatch')
    ix = np.triu_indices(len(a), 1)
    return float(centred_rank(a[ix]) @ centred_rank(b[ix]))


def correspondence_distribution(a, b, strata=None):
    """Exact identity-permutation reference distribution, including identity once."""
    a, b = distance_matrix(a), distance_matrix(b)
    if a.shape != b.shape or len(a) > 8:
        raise ValueError('Require aligned matrices of at most eight categories')
    n = len(a)
    if strata is None:
        permutations = itertools.permutations(range(n))
    else:
        groups = [list(g) for g in strata]
        if not groups or any(not g for g in groups) or sorted(sum(groups, [])) != list(range(n)):
            raise ValueError('Strata must partition concept indices')
        def restricted():
            for choice in itertools.product(*(itertools.permutations(g) for g in groups)):
                p = list(range(n))
                for group, perm in zip(groups, choice):
                    for i, j in zip(group, perm):
                        p[i] = j
                yield p
        permutations = restricted()
    ix = np.triu_indices(n, 1)
    left = centred_rank(a[ix])
    # Concept permutation only reorders the same pair distances and their tied ranks.
    ranks = np.zeros_like(b)
    ranks[ix] = centred_rank(b[ix])
    ranks += ranks.T
    orders = np.asarray(list(permutations), dtype=int)
    values = np.array([left @ ranks[p[ix[0]], p[ix[1]]] for p in orders])
    observed = float(left @ ranks[ix])
    return {'observed': observed, 'values': values, 'permutations': orders,
            'exceedance_fraction': float(np.mean(values >= observed - 1e-12)),
            'resolution': 1 / len(values)}


def linear_cka(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3:
        raise ValueError('Unaligned feature matrices')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite features')
    x, y = x - x.mean(0), y - y.mean(0)
    a, b = x @ x.T, y @ y.T
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        raise ValueError('Degenerate centred features')
    return float(np.sum(a * b) / denom)


def neighbour_overlap(a, b, k):
    a, b = distance_matrix(a), distance_matrix(b)
    if a.shape != b.shape or not isinstance(k, int) or not 1 <= k < len(a):
        raise ValueError('Invalid neighbour comparison')
    def inclusion(d):
        probabilities = np.zeros_like(d)
        for i in range(len(d)):
            other = np.array([j for j in range(len(d)) if j != i])
            values = d[i, other]
            threshold = np.sort(values)[k-1]
            less, equal = values < threshold, values == threshold
            probabilities[i, other[less]] = 1
            probabilities[i, other[equal]] = (k - int(less.sum())) / int(equal.sum())
        return probabilities
    return float(np.sum(inclusion(a) * inclusion(b)) / (len(a) * k))
