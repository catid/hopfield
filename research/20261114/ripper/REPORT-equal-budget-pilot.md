# H1ab equal-budget implementation pilot

The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.

All 49,920 masks, 161,740,800 log-LR values and 60,000 nested calibration pairs were checked. Every arm stops querying at its declared N and receives exactly 57,500 total noisy labels.

learned_small: upper bounds 1.4102%–2.1207%; coverage 6/6; fallback 0. Diagnostics do not select or change models.
learned_large: upper bounds 1.0204%–1.3184%; coverage 6/6; fallback 0. Diagnostics do not select or change models.

All allocations, the 5 pp primary margin, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.
