"""CPU validation of H3c saved boards, exact checkpoint parity and timing evidence."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3c'
OUT = ROOT / 'artifacts/research/20261114/loki/h3c'
OLD = ROOT / 'artifacts/research/20261114/loki'
PUBLIC = ROOT / 'results/research/loki/h3c'
sys.path.insert(0, str(ROOT / 'upstream'))
from cram import nqueens_data as D
spec = importlib.util.spec_from_file_location('h3_analyze', SRC.parent / 'analyze.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
HASHES = {}


def sha(path):
    path = Path(path)
    with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))] = digest
    return digest
def read(path): sha(path); return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def record(arm, updates, seed, inputs, solutions, keys, protocol):
    cached = updates == 4000
    path = OLD / f'h3b/primary/{arm}/s{seed}/final.json' if cached else OUT / f'search/{arm}/u{updates}/s{seed}/result.json'
    records = path.parent / ('final.npz' if cached else 'records.npz')
    rec = read(path); assert rec['status'] == 'complete'
    expected_protocol = SRC.parent / 'h3b/protocol.json' if cached else SRC / 'protocol.json'
    expected_source = SRC.parent / 'h3b/driver.py' if cached else SRC / 'driver.py'
    assert rec['protocol_sha256'] == sha(expected_protocol)
    assert rec['source_sha256'] == sha(expected_source)
    assert rec['model_sha256'] == sha(ROOT / rec['model_path'])
    assert rec['records_sha256'] == sha(records)
    z = dict(np.load(records)); boards = np.unpackbits(z['boards'], axis=-1)[..., :100].astype(bool)
    assert np.array_equal(z['inputs'], inputs)
    valid = np.array([[board.tobytes() in keys and board[given].all() for board in samples]
                      for given, samples in zip(inputs, boards)])
    counts = np.array([solutions[:, given].all(1).sum() for given in inputs]); denom = np.minimum(counts, 20)
    found = np.array([min(len({board.tobytes() for board in samples[ok]}), cap)
                      for samples, ok, cap in zip(boards, valid, denom)])
    n20 = valid[np.arange(len(inputs)), z['energy'].argmin(1)]
    for key, value in [('valid', valid), ('found', found), ('denom', denom), ('solution_counts', counts),
                       ('n20', n20), ('n1', valid[:, 0])]: assert np.array_equal(z[key], value), key
    assert ((z['moves'] >= 0) & (z['moves'] <= 20000)).all()
    assert (z['moves'][~z['stopped']] == 20000).all()
    assert np.array_equal(z['flop'], 20201 + 514 * z['moves'])
    assert int((z['stopped'] & ~valid).sum()) == rec['wrong_stops']
    assert int((~z['stopped']).sum()) == rec['budget_failures']
    assert rec['n20_accuracy_pct'] == 100 * float(n20.mean())
    assert rec['coverage_pct'] == 100 * float(found.sum() / denom.sum())
    row = dict(arm=arm, updates=updates, seed=seed, cached=cached, unique_inputs=len(inputs), chains=20,
        n20_correct=int(n20.sum()), n1_correct=int(valid[:, 0].sum()), valid_chains=int(valid.sum()),
        coverage_numerator=int(found.sum()), coverage_denominator=int(denom.sum()),
        wrong_stops=rec['wrong_stops'], capped_chains=rec['budget_failures'], mean_moves=rec['mean_moves'],
        search_seconds=rec['search_seconds'], model_sha256=rec['model_sha256'], records_sha256=rec['records_sha256'],
        result_path=str(path.relative_to(ROOT)), result_sha256=sha(path), protocol_sha256=rec['protocol_sha256'])
    return row, z


def main():
    started = time.perf_counter(); torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    p = read(SRC / 'protocol.json'); validation = read(OUT / 'validation.json')
    assert validation['status'] == 'passed' and all(validation['positive_scale_invariance_checks'])
    assert validation['protocol_sha256'] == sha(SRC / 'protocol.json')
    for group in ['source_sha256', 'data_sha256']:
        for path, digest in p[group].items(): assert sha(ROOT / path) == digest, path
    for arm in p['arms']:
        parity = read(OUT / f'parity-{arm}.json'); assert parity['status'] == 'passed' and all(parity['equality'].values())
        snapshot = OUT / f'fits/{arm}/u4000/model.pt'; cache = p['cache_models'][arm]
        assert sha(snapshot) == parity['snapshot_model_sha256']
        assert sha(ROOT / cache['path']) == parity['cache_model_sha256'] == cache['sha256']
        a = torch.load(snapshot, weights_only=False); b = torch.load(ROOT / cache['path'], weights_only=False)
        for key in a:
            assert torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key], (arm, key)
        for updates in p['checkpoints']:
            path = OUT / f'fits/{arm}/u{updates}'; fit = read(path / 'fit.json')
            assert fit['updates'] == updates and fit['schedule_denominator'] == 4000 and fit['rows'] == 97554
            assert fit['model_sha256'] == sha(path / 'model.pt')
            assert fit['protocol_sha256'] == sha(SRC / 'protocol.json') and fit['source_sha256'] == sha(SRC / 'driver.py')
    timing = read(OUT / 'timing.json'); assert timing['status'] == 'complete'
    assert timing['protocol_sha256'] == sha(SRC / 'protocol.json') and timing['source_sha256'] == sha(SRC / 'driver.py')
    assert all(all(row['parity'].values()) for row in timing['records'])
    timed = {}; warmups = {}
    for arm in p['arms']:
        timed[arm] = [row['seconds'] for row in timing['records'] if row['arm'] == arm and not row['warmup']]
        warmups[arm] = [row['seconds'] for row in timing['records'] if row['arm'] == arm and row['warmup']]
        assert len(timed[arm]) == 3 and len(warmups[arm]) == 1
    ratio = float(np.median(timed['adam-nodecay']) / np.median(timed['sgd-pre']))
    inputs = np.load(ROOT / p['search']['input_file'])['inputs'] == 2
    solutions = np.asarray(D.completions(np.zeros(100, bool), 10), bool); assert len(solutions) == 724
    keys = {row.tobytes() for row in solutions}; rows = []; per_seed = []; arrays = {}
    models = [('adamw-decay', 4000)] + [(arm, u) for arm in p['arms'] for u in p['checkpoints']]
    for arm, updates in models:
        checked = [record(arm, updates, seed, inputs, solutions, keys, p) for seed in p['search']['seeds']]
        recs, zs = zip(*checked); per_seed.extend(recs)
        arr = {key: np.stack([z[key] for z in zs]) for key in ['n20', 'n1', 'found']}; denom = zs[0]['denom']
        arrays[arm, updates] = (arr, denom)
        rows.append(dict(arm=arm, updates=updates, unique_inputs=len(inputs), search_seeds=p['search']['seeds'],
            n20_accuracy_pct=float(arr['n20'].mean()*100), n1_accuracy_pct=float(arr['n1'].mean()*100),
            coverage_pct=float(arr['found'].sum(axis=1).mean()/denom.sum()*100),
            mean_moves=float(np.mean([r['mean_moves'] for r in recs])),
            wrong_stops=sum(r['wrong_stops'] for r in recs), capped_chains=sum(r['capped_chains'] for r in recs),
            cached_inference=updates == 4000))
    a, denom = arrays['adam-nodecay', 1000]; b, _ = arrays['sgd-pre', 4000]
    stats = A.bootstrap(a, b, denom); regressions = (b['n20'] & ~a['n20']).any(0)
    upper = A.cp_upper(int(regressions.sum()), len(regressions)); accuracy_safe = upper < .001
    supported = accuracy_safe and ratio <= .5
    falsified = stats['accuracy_one_sided_95_upper_pp'] < -.1 or ratio > .5
    assert not (supported and falsified)
    decision = 'supported' if supported else 'falsified' if falsified else 'inconclusive'
    primary = dict(decision=decision, arm='adam-nodecay', updates=1000, control='sgd-pre', control_updates=4000,
        any_regression_inputs=int(regressions.sum()), regression_probability_exact95_upper=upper,
        conservative_accuracy95_lower_pp=-100*upper, accuracy_safeguard_passed=accuracy_safe,
        median_complete_fit_time_ratio=ratio, timing_target_passed=ratio <= .5, **stats)
    sha(__file__); sha(SRC.parent / 'analyze.py')
    summary = dict(status='validated', utc=dt.datetime.now(dt.timezone.utc).isoformat(), hypothesis='H3c',
        decision=decision, primary=primary, rows=rows, timing=dict(samples_seconds=timed, warmups_seconds=warmups,
        median_seconds={arm: float(np.median(values)) for arm, values in timed.items()}, scope=p['timing']['scope']),
        independent_validation=dict(saved_records=len(per_seed), saved_boards=len(per_seed)*len(inputs)*20,
            method='Membership in independently enumerated 724 valid complete solutions, preserved givens, exact per-seed counts, stopping flags, move/FLOP budgets and all hashes.'),
        limits=p['limits'], input_sha256=HASHES, audit_seconds=time.perf_counter()-started,
        new_training_or_inference=False)
    save(OUT / 'summary.json', summary); save(PUBLIC / 'summary.json', summary)
    for dest in [OUT / 'per_seed.csv', PUBLIC / 'per_seed.csv']:
        with dest.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_seed[0])); writer.writeheader(); writer.writerows(per_seed)
    names = {'adamw-decay': 'BP AdamW, decay 0.01 (cached)', 'adam-nodecay': 'BP Adam, no decay', 'sgd-pre': 'SGD + geometry preconditioner'}
    lines = ['# H3c: fixed training prefixes', '',
        f"**Frozen primary hypothesis: {decision}.** No-decay Adam after 1,000 updates was compared with preconditioned SGD after 4,000 updates, on all 6,513 unique n=10 official test inputs, with 20 chains and three paired search seeds. Both retain the original 4,000-update cosine schedule. The fixed 1,000-update checkpoint was specified before any prefix search outcome; no checkpoint was selected from this curve.", '',
        '| Method | Updates | N20 accuracy (%) | Coverage (%) | Mean moves / chain | Wrong stops | Capped chains |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {names[row['arm']]} | {row['updates']:,} | {row['n20_accuracy_pct']:.6f} | {row['coverage_pct']:.6f} | {row['mean_moves']:.2f} | {row['wrong_stops']:,} | {row['capped_chains']:,} |")
    lines += ['', f"Primary accuracy difference: {stats['accuracy_difference_pp']:+.6f} percentage points; one-sided 95% bootstrap endpoints [{stats['accuracy_one_sided_95_lower_pp']:.6f}, {stats['accuracy_one_sided_95_upper_pp']:.6f}]. The exact upper bound on the fraction of inputs with any paired regression is {100*upper:.6f}% ({int(regressions.sum())} observed inputs). The noninferiority margin is 0.1 point; search repeats are clustered inside each input.", '',
        f"Median complete in-memory fit: Adam at 1,000 updates {np.median(timed['adam-nodecay']):.6f} seconds; SGD at 4,000 updates {np.median(timed['sgd-pre']):.6f} seconds. Ratio {ratio:.6f}; the frozen target was at most 0.5. All three timings and warmups appear in the JSON. Both methods were timed on Loki GPU0 after one warmup each; other campaign work waited until timing finished.", '',
        p['timing']['scope'], '',
        'The two newly reconstructed 4,000-update trajectories exactly match all cached learned tensors and calibration scalars. Their previous searches are reused without rerunning. Other checkpoints are prefixes of these same trajectories, each with its own training-derived temperature and stopping statistics. Earlier snapshots add no state update and were excluded from timed prefix fits.', '',
        'This is an adaptive research follow-up on a previously used official test set. It provides evidence conditional on the frozen settings and cohort, not untouched external confirmation or broad optimizer superiority. There is one deterministic fitted trajectory per optimizer; the three seeds vary search only. All secondary checkpoints and failures are reported.', '',
        'Positive radial rescaling is an analytic inference invariant when energy, temperature unit, stopping threshold and tolerance scale together. A small CPU power-of-two fixture verified identical boards, stopping, moves, FLOPs and scaled energies. Keeping an absolute tolerance fixed changes the relative tolerance; this was not treated as a useful new GPU hypothesis.', '',
        '[Frozen protocol](protocol.json), [runner](driver.py), [auditor](analyze.py), [tracked summary](../../../../results/research/loki/h3c/summary.json), [per-seed counts](../../../../results/research/loki/h3c/per_seed.csv).']
    (SRC / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (PUBLIC / 'README.md').write_text('# H3c: fixed training prefixes\n\n' +
        f"The frozen primary hypothesis was **{decision}**. No-decay Adam at 1,000 updates changed N20 accuracy by {stats['accuracy_difference_pp']:+.6f} percentage points relative to preconditioned SGD at 4,000 updates. The observed median complete in-memory fit-time ratio was {ratio:.6f}.\n\n" +
        'This adaptive follow-up reuses the prior official test set. The checkpoint and margins were fixed before new prefix outcomes; all secondary checkpoints are reported. One deterministic fitted model per trajectory, three repeated search seeds per input.\n\n' +
        '[Full report and protocol](../../../../research/20261114/loki/h3c/REPORT.md), [summary and hashes](summary.json), [per-seed integer counts](per_seed.csv).\n')
    print(json.dumps(dict(status='validated', decision=decision, primary=primary, rows=len(rows),
                          per_seed_rows=len(per_seed), hashed_files=len(HASHES), audit_seconds=summary['audit_seconds'])))


if __name__ == '__main__': main()
