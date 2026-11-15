# H1aa repeated-query calibration: implementation pilot

The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, and support audits. It contains no solver inference and is not pooled with fresh evidence.

All 27,000 masks, 87,480,000 log-LR values and 60,000 repeated-query pairs were checked. Learned bounds span 1.0647% to 1.2193%; all six cover the true 1% rate, with no fallback. Coverage diagnostics do not select or modify models.

M10000, N1500, 5 pp noninferiority, streams, error budgets and 32 fresh banks were fixed before this pilot. The learned arm costs 57,500 queries per bank; each known-parameter control costs 37,500. No outcome-dependent setting changes were made.
