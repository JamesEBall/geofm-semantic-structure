"""Explicit unavailable slots in the fixed primary family. Author: James Edward Ball."""
import math
from .semantic_evaluation_v1 import holm_four

SLOTS = ('H1_clay', 'H1_prithvi', 'H3_clay', 'H3_prithvi')


def primary_family(h3):
    if set(h3) != {'clay', 'prithvi'}:
        raise ValueError('Require both primary model records')
    records = [dict(endpoint=name, status='unavailable', observed_p=None,
                    reason='population_RSA_inference_not_qualified', numeric_holm_input=1.)
               for name in SLOTS[:2]]
    for model in ['clay', 'prithvi']:
        item = h3[model]
        p = item.get('one_sided_p')
        qualified = item.get('admitted_hierarchy_qualified') is True
        if qualified and p is not None:
            p = float(p)
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid qualified p-value')
            records.append(dict(endpoint='H3_'+model, status='conditional_model_based',
                                observed_p=p, numeric_holm_input=p))
        else:
            records.append(dict(endpoint='H3_'+model, status='unavailable', observed_p=None,
                                numeric_holm_input=1., reason=item.get('reason') or
                                'admitted_hierarchy_or_component_statistic_unqualified'))
    adjusted = holm_four([r['numeric_holm_input'] for r in records])
    for record, value in zip(records, adjusted):
        available = record['observed_p'] is not None
        record['adjusted_p'] = float(value) if available else None
        record['reject_zero_at_0_05'] = bool(value < .05) if available else None
    return records
