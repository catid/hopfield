"""Nested, perfectly precise thinning of supplied corruption locations."""
import numpy as np
from corruption_censor_math import mask_from_indices, score


def threshold(percent):
    assert type(percent) is int and 0 <= percent <= 100
    return percent * (1 << 64) // 100


def draws(seed, bank, rows=600):
    assert type(seed) is int and seed >= 0 and type(bank) is int and bank >= 0
    assert type(rows) is int and rows > 0
    return np.random.PCG64(np.random.SeedSequence([seed, bank])).random_raw(rows*9).reshape(rows, 9)


def revealed_mask(cells, raw, percent):
    full = mask_from_indices(cells)
    assert raw.dtype == np.uint64 and raw.shape == cells.shape
    limit = threshold(percent)
    chosen = np.ones_like(raw, bool) if percent == 100 else raw < np.uint64(limit)
    partial = np.zeros_like(full)
    partial[np.arange(len(cells))[:, None], cells] = chosen
    assert not np.any(partial & ~full)
    return partial


def independent_mask(cells, raw, percent):
    assert type(percent) is int and 0 <= percent <= 100
    assert raw.dtype == np.uint64 and raw.shape == cells.shape
    full = mask_from_indices(cells)
    partial = np.zeros_like(full)
    # Python integers avoid unsigned threshold coercion and handle 2**64.
    for r, row in enumerate(cells):
        for k, cell in enumerate(row):
            if 100*int(raw[r, k]) < percent*(1 << 64) - (percent*(1 << 64) % 100):
                partial[r, int(cell)] = True
    return partial


def independent_draws(seed, bank, rows=600):
    generator = np.random.PCG64(np.random.SeedSequence([seed, bank]))
    return np.asarray([generator.random_raw(9) for _ in range(rows)], dtype=np.uint64)
