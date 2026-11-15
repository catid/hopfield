"""Label-free numerical acceptance for H2bj source fallback diagnostics."""
import numpy as np


def accepted(hidden_good, finite, stopped, native_cap, block_cap, margin):
    """Accept only converged finite original-objective certificates.

    No tokens, labels, shortest distances, graph metrics or input identifiers are
    accepted by this interface. The certificate belongs to the stated model;
    callers must preserve that provenance when routing to another model.
    """
    arrays = [np.asarray(x) for x in
              (hidden_good, finite, stopped, native_cap, block_cap, margin)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Certificate fields must have identical shapes')
    if any(x.dtype != np.bool_ for x in arrays[:5]):
        raise TypeError('Certificate and cap flags must be Boolean')
    hidden, finite, stopped, native_cap, block_cap, margin = arrays
    return (hidden & finite & stopped & ~native_cap & ~block_cap
            & np.isfinite(margin) & (margin >= -1e-6))


def chosen_model(lp_accepted):
    """0 denotes preserved LP; 1 denotes the explicitly attributed BP fallback."""
    flags = np.asarray(lp_accepted)
    if flags.dtype != np.bool_:
        raise TypeError('LP acceptance must be Boolean')
    return (~flags).astype(np.uint8)
