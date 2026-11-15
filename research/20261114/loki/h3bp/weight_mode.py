"""Uniform weight on every remaining candidate; no source or case weighting."""
def uniform_weights(candidates):
    assert candidates==sorted(set(candidates)) and 1<=len(candidates)<=16
    weights=[1 for _ in candidates]
    return weights,dict(mode='uniform_over_all_remaining_candidates',candidates=candidates,integer_weights=['1' for _ in candidates],weight_sum=len(candidates),no_candidate_excluded=True)
