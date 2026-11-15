# H1ac rate-transfer implementation pilot

The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.

All 53,280 masks, 172,627,200 log-LR values and 6,000 calibration boards at both fixed noise rates were checked. Each arm receives 55,500 training queries; each learned arm also receives 2,000 calibration queries.

learned_01: upper bounds 1.4765%–2.5585%; coverage 6/6; fallback 0. Diagnostics do not select or change models.
learned_05: upper bounds 5.6948%–7.0052%; coverage 6/6; fallback 0. Diagnostics do not select or change models.

All allocations, the 5 pp margin, 70% floor, noise rates, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.
