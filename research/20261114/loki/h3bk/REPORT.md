# H3bk: incomplete after exact-rational serialization error

The frozen run is incomplete. Its final exact aggregate numerator or denominator exceeded Python's default 4,300-decimal-digit integer-to-string conversion limit. All 125 group receipts were written, but the whole-study result could not be serialized; no primary decision is reported. This is an implementation/output error, not an infeasibility or a failed 10% scientific hypothesis.

The original source and protocol remain unchanged. All partial catalogs, proofs, receipts, the traceback and failed whole-study record are retained. Their integrity can be checked, but they are not presented as an independently validated scientific result. The run used 1.864954 aggregate CPU seconds / 1.873281 wall seconds, 8,558 memo states and 50,439 split attempts. Zero new draws, labels or GPU work.

A corrected serialization environment must receive a new frozen study directory and preserve the same threshold and scope. Run `upstream/.venv/bin/python research/20261114/loki/h3bk/verify_incomplete.py --root SNAPSHOT` for integrity-only verification.
