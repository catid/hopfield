"""Bind H2bm source bytes without parsing any real geometry outcome fields."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


here = Path(__file__).resolve().parent
root = here.parents[2]
dest = root / 'artifacts/research/20261114/work/bp_geometry_decomposition'
source = root / 'artifacts/research/20261114/work/cross_certificate_screen'
proposal = here / 'bp-geometry-decomposition-proposal.json'
protocol = here / 'bp-geometry-decomposition-protocol.json'
assert not protocol.exists() and not (dest / 'protocol.json').exists()
p = json.loads(proposal.read_text())
gate_path = dest / 'synthetic-gate.json'
gate = json.loads(gate_path.read_text())
assert gate['status'] == 'passed' and gate['synthetic_only']
assert gate['incomplete_receipt_preserved'] and gate['existing_output_refused'] and gate['watchdog_signal_incomplete_receipt']
for name, digest in gate['source_sha256'].items():
    assert sha(here / name) == digest, name
p['status'] = 'frozen_before_first_geometry_aggregate'
p['frozen_utc'] = datetime.now(timezone.utc).isoformat()
p['review'] = 'Root reviewed predicate, BFS, cohort identity, denominator and complete-negative audit; authorized freezing and execution after separate process CPU/wall accounting, overwrite refusal and watchdog failure receipts were gated.'
p['proposal_sha256'] = sha(proposal)
p['synthetic_gate_sha256'] = sha(gate_path)
p['analysis_source_sha256'] = {name: sha(here / name) for name in [
    'bp_geometry_checks.py', 'preflight_bp_geometry_decomposition.py',
    'analyze_bp_geometry_decomposition.py', 'freeze_bp_geometry_decomposition.py',
    'audit_connectivity_screen.py']}
p['sealed_source_sha256'] = {name: sha(source / name) for name in [
    'protocol.json', 'audit.json', 'sealed-margins.json', 'per-model-input.csv',
    'public-fixtures/manifest.json', 'public-fixtures/source-audit.json']}
p['source_public_verification_sha256'] = sha(root / 'results/research/work/cross-certificate-screen/public-verification.json')
p['original_geometry_definition_sha256'] = sha(here / 'audit_connectivity_screen.py')
p['freeze_reads_real_outcome_fields'] = False
payload = json.dumps(p, indent=2) + '\n'
protocol.write_text(payload)
(dest / 'protocol.json').write_text(payload)
print(json.dumps({'status': p['status'], 'frozen_utc': p['frozen_utc'],
                  'protocol_sha256': sha(protocol), 'outcome_aggregate_read': False}, indent=2))
