"""Frozen nuisance matching, independent of model scores. Author: James Edward Ball."""
import hashlib
import itertools
import numpy as np


def matched_pairs(z, membership, patch_ids, parents, components, strata, caliper):
    z, y = np.asarray(z, float), np.asarray(membership)
    n = len(patch_ids)
    if z.ndim != 2 or len(z) != n or not np.isfinite(z).all():
        raise ValueError('Require aligned finite nuisance matrix')
    if y.ndim != 2 or len(y) != n or y.dtype != bool:
        raise ValueError('Require aligned complete-gallery membership')
    if len(set(patch_ids)) != n or any(len(a) != n for a in [parents, components, strata]):
        raise ValueError('Duplicate IDs or unaligned hierarchy')
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError('Require positive finite caliper')
    for p in set(parents):
        if len({components[i] for i in range(n) if parents[i] == p}) != 1:
            raise ValueError('Parent spans components')
    def canonical(category, other):
        grouped = {}
        for i in range(n):
            if y[i, category] and not y[i, other]:
                grouped.setdefault(parents[i], []).append(i)
        return [min(indices, key=lambda i: hashlib.sha256(
            ('dw-nuisance-match-v1|' + patch_ids[i]).encode()).hexdigest())
                for _, indices in sorted(grouped.items())]
    output = []
    for left, right in itertools.combinations(range(y.shape[1]), 2):
        a, b = canonical(left, right), canonical(right, left)
        edges = []
        for i in a:
            for j in b:
                if strata[i] != strata[j] or parents[i] == parents[j]:
                    continue
                distance = float(np.sqrt(np.mean((z[i]-z[j])**2)))
                if distance <= caliper:
                    key = hashlib.sha256('|'.join(sorted([patch_ids[i], patch_ids[j]])).encode()).hexdigest()
                    edges.append((distance, key, i, j))
        used, kept = set(), []
        for distance, key, i, j in sorted(edges):
            if parents[i] in used or parents[j] in used:
                continue
            used.update([parents[i], parents[j]])
            kept.append({'left_patch': patch_ids[i], 'right_patch': patch_ids[j],
                         'left_index': i, 'right_index': j, 'distance': distance,
                         'tie_hash': key, 'left_parent': parents[i], 'right_parent': parents[j],
                         'left_component': str(components[i]), 'right_component': str(components[j])})
        before = (z[a].mean(0)-z[b].mean(0)).tolist() if a and b else None
        after = (np.mean([z[e['left_index']] for e in kept], axis=0)-np.mean(
            [z[e['right_index']] for e in kept], axis=0)).tolist() if kept else None
        output.append({'category_indices': [left, right], 'left_candidate_parents': len(a),
                       'right_candidate_parents': len(b), 'eligible_edges': len(edges),
                       'matched_edges': kept, 'unique_matched_parents': len(used),
                       'unmatched_left_parents': len(a)-len(kept),
                       'unmatched_right_parents': len(b)-len(kept),
                       'standardised_mean_difference_before': before,
                       'standardised_mean_difference_after': after,
                       'status': 'supported' if kept else 'unavailable_no_matches'})
    return output
