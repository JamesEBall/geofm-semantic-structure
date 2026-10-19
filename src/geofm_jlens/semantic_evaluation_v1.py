"""Fixed-gallery evaluation aggregation. Author: James Edward Ball."""
import numpy as np
from scipy.stats import t
from .semantic_retrieval_v1 import retrieval_metrics


def aggregate(values, parents, components):
    values = np.asarray(values, dtype=float)
    parents, components = np.asarray(parents), np.asarray(components)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Require nonempty finite query values')
    if parents.shape != values.shape or components.shape != values.shape:
        raise ValueError('Unaligned query hierarchy')
    by_parent, parent_component = {}, {}
    for parent in np.unique(parents):
        mask = parents == parent
        group = np.unique(components[mask])
        if len(group) != 1:
            raise ValueError('Parent spans components')
        by_parent[str(parent)] = float(values[mask].mean())
        parent_component[str(parent)] = str(group[0])
    by_component = {}
    for component in sorted(set(parent_component.values())):
        by_component[component] = float(np.mean([
            value for parent, value in by_parent.items()
            if parent_component[parent] == component]))
    return {'component_mean': float(np.mean(list(by_component.values()))),
            'pooled_query_mean': float(values.mean()),
            'parent_means': by_parent, 'component_means': by_component,
            'queries': len(values), 'parents': len(by_parent),
            'components': len(by_component)}


def summarise_retrieval(scores, relevance, parents, components):
    scores, relevance = np.asarray(scores, float), np.asarray(relevance)
    if scores.ndim != 2 or scores.shape[1] != 8:
        raise ValueError('Require fixed eight-category gallery')
    result = retrieval_metrics(scores, relevance, ks=(1, 5, 8))
    ap = aggregate(result['per_query_ap'], parents, components)
    recall = {k: aggregate(v, parents, components)
              for k, v in result['per_query_recall'].items()}
    supported = np.flatnonzero(relevance.any(axis=0))
    reverse = retrieval_metrics(scores.T[supported], relevance.T[supported], ks=(1, 5, 10))
    return {'ap': ap, 'recall': recall, 'per_query_ap': result['per_query_ap'],
            'per_query_recall': result['per_query_recall'], 'gallery_size': 8,
            'reverse_supported_category_indices': supported.tolist(),
            'reverse_unsupported_category_indices': np.flatnonzero(~relevance.any(axis=0)).tolist(),
            'reverse': reverse}


def candidate_paired_interval(model_ap, baseline_ap, parents, components):
    """Geography-only interval candidate, conditional on the five fitted maps."""
    model, base = np.asarray(model_ap, float), np.asarray(baseline_ap, float)
    if model.ndim != 2 or model.shape != base.shape or model.shape[0] != 5:
        raise ValueError('Require five aligned seed by query arrays')
    if not np.isfinite(model).all() or not np.isfinite(base).all():
        raise ValueError('Non-finite AP')
    if (model < 0).any() or (model > 1).any() or (base < 0).any() or (base > 1).any():
        raise ValueError('AP outside unit range')
    difference = (model-base).mean(axis=0)
    result = aggregate(difference, parents, components)
    values = np.array(list(result['component_means'].values()))
    result['seed_component_mean_differences'] = [
        aggregate(row, parents, components)['component_mean'] for row in model-base]
    result['inference_status'] = 'candidate_requires_admitted_hierarchy_qualification'
    if len(values) < 3:
        result.update(interval=None, one_sided_p=None, reason='fewer_than_three_components')
        return result
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    if se <= 1e-15:
        result.update(interval=None, one_sided_p=None, reason='degenerate_component_variance')
        return result
    effect = result['component_mean']
    radius = float(t.ppf(.975, len(values)-1)*se)
    result.update(interval=[effect-radius, effect+radius], standard_error=se,
                  degrees_of_freedom=len(values)-1,
                  one_sided_p=float(t.sf(effect/se, len(values)-1)))
    return result


def holm_four(pvalues):
    p = np.asarray(pvalues, float)
    if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Require four finite family p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(4)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order]*np.arange(4, 0, -1)))
    return adjusted
