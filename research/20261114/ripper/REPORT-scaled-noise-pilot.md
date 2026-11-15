# H1ad scaled-noise implementation pilot

The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.

All 79,920 masks, 258,940,800 log-LR values and 6,000 calibration boards at fixed 5% noise were checked. Short trajectories were verified as literal prefixes of the long trajectories. Each learned model pays 2,000 calibration queries in addition to its training queries.

learned_short: upper bounds 6.1118%–7.1840%; coverage 6/6; fallback 0. Diagnostics do not select or change models.
learned_long: upper bounds 6.1118%–7.1840%; coverage 6/6; fallback 0. Diagnostics do not select or change models.

All allocations, the 5 pp margin, 70% floor, noise rates, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.
