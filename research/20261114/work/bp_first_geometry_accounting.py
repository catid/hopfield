"""Candidate H2bn complete-interval checks and fixed finite endpoints."""
import math
from fractions import Fraction


def batch_cost(receipt):
    total = receipt['answer_seconds']
    components = receipt['components']
    required = {'input_transfer', 'hidden_readout_and_flags', 'LP96_complete_prefix',
                'LP96_answer_return', 'LP96_original_margin_and_return', 'final_numerical_acceptance'}
    if receipt['mode'] == 'BP_geometry_LP96':
        required |= {'BP_native_solve', 'BP_cap_flags', 'BP_answer_return',
                     'BP_original_margin_and_return', 'geometry_screen_and_indices',
                     'LP_subset_gather', 'subset_scatter_and_provenance'}
    elif receipt['mode'] == 'LP96':
        required.add('model_provenance')
    else:
        raise ValueError('Unknown cost arm')
    if not required <= components.keys():
        raise ValueError('Missing required answer operation')
    values = [total, receipt['unattributed_in_segment_seconds'], *components.values()]
    if not all(math.isfinite(x) and x >= 0 for x in values) or total == 0:
        raise ValueError('Invalid measured interval')
    if abs(total - sum(components.values()) - receipt['unattributed_in_segment_seconds']) > 1e-8:
        raise ValueError('Incomplete or overlapping answer accounting')
    if receipt['mode'] == 'BP_geometry_LP96' and receipt['actual_subset_inputs'] == 0:
        if any(components[k] != 0 for k in ['LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return']):
            raise ValueError('Empty subset claims LP execution')
    if not 0 <= receipt['actual_subset_inputs'] <= receipt['n']:
        raise ValueError('Invalid subset cardinality')
    return total


def paired_endpoint(pairs):
    if len(pairs) != 4 or sorted(p['pair'] for p in pairs) != list(range(4)):
        raise ValueError('Exactly four fixed pairs required')
    values = []
    for pair in sorted(pairs, key=lambda p: p['pair']):
        control, candidate = pair['LP96_complete_answer_seconds'], pair['candidate_complete_answer_seconds']
        if not all(math.isfinite(v) and v > 0 for v in [control, candidate]):
            raise ValueError('Invalid cohort answer cost')
        values.append(Fraction(1) - Fraction(candidate) / Fraction(control))
    ordered = sorted(values)
    median = (ordered[1] + ordered[2]) / 2
    return {'paired_savings': [float(v) for v in values], 'median_saving': float(median),
            'minimum_saving': .10, 'exact_median_fraction': str(median),
            'decision': 'supported' if median >= Fraction(1, 10) else 'falsified'}


def preservation(control, candidate):
    if len(control) != 4096 or len(candidate) != 4096 or sum(control) != 4095:
        raise ValueError('Source joint cohort changed')
    if any(v not in [False, True, 0, 1] for v in [*control, *candidate]):
        raise ValueError('Nonbinary outcome')
    gains = sum(bool(b) and not bool(a) for a, b in zip(control, candidate))
    losses = sum(bool(a) and not bool(b) for a, b in zip(control, candidate))
    return {'joint_inputs': 4096, 'control_joint_successes': 4095,
            'candidate_joint_successes': sum(candidate), 'gains': gains, 'losses': losses,
            'decision': 'supported' if losses == 0 else 'falsified'}
