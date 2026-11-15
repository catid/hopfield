"""Independent saved-state audit plus frozen paired input-cluster inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import time
import numpy as np
import audit_noisy_pl_long as audit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/logical_completion'
PUBLIC = ROOT/'results/research/foureyes/logical_completion'
PROTOCOL = HERE/'logical_completion.json'
audit.OUT = OUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    p = json.loads(PROTOCOL.read_text());prepared = json.loads((OUT/'prepared.json').read_text())
    assert prepared['protocol_sha256'] == sha(PROTOCOL)
    for path, digest in (p['source_sha256'] | p['frozen_inputs']).items():
        assert sha(ROOT/path) == digest
    jobs = json.loads((OUT/'queue.json').read_text());assert all(j['status'] == 'complete' for j in jobs)
    with np.load(ROOT/'upstream/data/full-test/full-all.npz', allow_pickle=False) as data:
        lo, hi = p['rows_half_open']
        np.savez_compressed(OUT/'audit-target.npz', solutions=data['solutions'][lo:hi].astype(np.int8)-1,
            free=~data['clue_mask'][lo:hi].astype(bool), puzzle=np.arange(lo, hi))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(audit.audit_job, jobs))
    lookup = {r['job_id']: r for r in results}
    rows = []
    for mapping in prepared['mappings']:
        r = lookup[mapping['actual']]
        assert r['support_sha256'] == mapping['support_sha256']
        rows.append(dict(**mapping, accuracy_pct=float(r['good'].mean()*100), wrong_stops=int((r['stopped'] & ~r['good']).sum())))
    comparisons = []
    for source in ['adamw', 'paper_pl', 'long_pl']:
        banks = sorted({r['bank'] for r in rows if r['source_arm'] == source})
        outcomes = {(r['bank'], r['arm']): lookup[r['actual']]['good'].astype(np.int8) for r in rows if r['source_arm'] == source}
        delta = np.stack([outcomes[b, 'completed']-outcomes[b, 'original'] for b in banks]).mean(0)
        rng = np.random.default_rng(20261115555)
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()*100 for _ in range(5000)])
        lower, upper = np.quantile(boot, [.025, .975]).tolist()
        wrong = sum(r['wrong_stops'] for r in rows if r['source_arm'] == source)
        comparisons.append(dict(source_arm=source, fixed_models=len(banks), original_accuracy_pct=float(np.mean([outcomes[b, 'original'].mean() for b in banks])*100),
            completed_accuracy_pct=float(np.mean([outcomes[b, 'completed'].mean() for b in banks])*100), gain_pp=float(delta.mean()*100),
            paired_input95_interval_pp=[lower, upper], logical_wrong_stops=wrong,
            decision=('falsified' if wrong or upper < 1 else 'supported' if lower >= 1 else 'inconclusive') if source == 'adamw' else 'descriptive'))
    PUBLIC.mkdir(parents=True, exist_ok=True)
    ids = sorted(lookup)
    np.savez_compressed(PUBLIC/'audited-outcome-bits.npz', job_ids=np.array(ids),
        valid=np.packbits(np.stack([lookup[k]['good'] for k in ids]), axis=1),
        stopped=np.packbits(np.stack([lookup[k]['stopped'] for k in ids]), axis=1))
    evidence = {str(PROTOCOL.relative_to(ROOT)): sha(PROTOCOL), str((OUT/'prepared.json').relative_to(ROOT)): sha(OUT/'prepared.json')}
    for r in results:
        evidence.update(r['input_sha256'])
    summary = dict(status='validated', protocol_sha256=sha(PROTOCOL), comparisons=comparisons, model_mappings=prepared['mappings'],
        outcomes=rows, logical_cells=len(rows), unique_searches=len(jobs), boards_audited=10000*len(jobs), unique_inputs=10000,
        input_sha256=evidence, outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'), audit_seconds=time.perf_counter()-start,
        search_seconds=sum(r['search_seconds'] for r in results), limits=[p['selection_scope'], p['input_scope'], p['scientific_change'], 'Intervals condition on the selected fixed models and one search seed.'])
    (PUBLIC/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (PUBLIC/'per_model.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(comparisons=comparisons, boards_audited=summary['boards_audited'], audit_seconds=summary['audit_seconds'])))


if __name__ == '__main__':
    main()
