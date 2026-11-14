"""H3d saved-output reanalysis with independent certificates and selector audit."""
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
SRC = ROOT / 'research/20261114/loki/h3d'
OUT = ROOT / 'artifacts/research/20261114/loki/h3d'
PUBLIC = ROOT / 'results/research/loki/h3d'
sys.path[:0] = [str(SRC), str(ROOT / 'upstream')]
import driver
from cram import nqueens_data as D
spec = importlib.util.spec_from_file_location('h3analysis', SRC.parent / 'analyze.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
HASHES = {}


def sha(path):
    path = Path(path)
    with path.open('rb') as handle: result = hashlib.file_digest(handle, 'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))] = result; return result
def read(path): sha(path); return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def certificates(boards, inputs, flags, model):
    # All saved models are exactly the declared Ising parameterization.
    tensor = model['S'].numpy().astype(np.float64)
    assert not np.count_nonzero(tensor[:, :, 0, :]) and not np.count_nonzero(tensor[:, :, :, 0])
    assert not np.count_nonzero(model['B'][:, 0].numpy())
    J = tensor[:, :, 1, 1]; h = model['B'][:, 1].numpy().astype(np.float64)
    assert np.array_equal(J, J.T) and not np.count_nonzero(np.diag(J))
    flat = boards.reshape(-1, 100); given = np.repeat(inputs, 20, axis=0)
    gaps = []
    for low in range(0, len(flat), 4096):
        x = flat[low:low+4096].astype(np.float64)
        delta = (1 - 2*x) * (x @ J + h)
        delta[given[low:low+4096]] = np.inf
        gaps.append(delta.min(axis=1) - (float(model['delta']) - .001))
    gaps = np.concatenate(gaps).reshape(flags.shape)
    stopped = gaps[flags]; capped = gaps[~flags]
    return dict(stopped_count=int(flags.sum()), strict_negative_count=int((stopped < 0).sum()),
        below_numerical_allowance_count=int((stopped < -.001).sum()),
        minimum_stopped_gap=float(stopped.min()) if len(stopped) else None,
        maximum_stopped_deficit=float(max(0., -stopped.min())) if len(stopped) else None,
        capped_returned_board_satisfies_direct_certificate=int((capped >= 0).sum()),
        gap_quantiles_stopped=np.quantile(stopped, [0, .01, .5, .99, 1]).tolist() if len(stopped) else None,
        definition='Float64 min free-cell one-flip margin minus learned delta plus 0.001; strict negative deficits and the predefined 0.001 allowance are reported separately.')


def measure_selector(energy, flags):
    for _ in range(10): energy.argmin(axis=1); driver.choose(energy, flags)
    values = {'original': [], 'certificate_first': []}
    for repeat in range(101):
        order = ['original', 'certificate_first'] if repeat % 2 == 0 else ['certificate_first', 'original']
        for name in order:
            start = time.perf_counter_ns()
            result = energy.argmin(axis=1) if name == 'original' else driver.choose(energy, flags)
            values[name].append((time.perf_counter_ns() - start) / 1e6)
            assert len(result) == 6513
    original, new = [float(np.median(values[key])) for key in ['original', 'certificate_first']]
    return dict(milliseconds=values, medians_ms=dict(original=original, certificate_first=new),
        median_overhead_ms=new-original, median_ratio=new/original, warmups_each=10, repeats_each=101,
        input_rows=6513, chains=20, scope='CPU selector only; no search, I/O, rule checker, model evaluation or certificate recomputation included.')


def main():
    start = time.perf_counter(); torch.set_num_threads(1); PUBLIC.mkdir(parents=True, exist_ok=True)
    p = read(SRC / 'protocol.json')
    for path, digest in p['source_sha256'].items(): assert sha(ROOT / path) == digest, path
    solutions = np.asarray(D.completions(np.zeros(100, bool), 10), bool); assert len(solutions) == 724
    keys = {row.tobytes() for row in solutions}; records = []; evidence = {}; diagnostics = {}; timing = None
    for case in p['cases']:
        assert sha(ROOT / case['record']) == case['record_sha256']
        assert sha(ROOT / case['arrays']) == case['arrays_sha256']
        assert sha(ROOT / case['model']) == case['model_sha256']
        rec = read(ROOT / case['record']); assert rec['status'] == 'complete'
        z = dict(np.load(ROOT / case['arrays']))
        boards = np.unpackbits(z['boards'], axis=-1)[..., :100].astype(bool)
        valid = np.array([[row.tobytes() in keys and row[given].all() for row in samples]
                          for given, samples in zip(z['inputs'], boards)])
        assert np.array_equal(valid, z['valid'])
        energy, flags = z['energy'], z['stopped']
        assert np.isfinite(energy).all()
        original = energy.argmin(axis=1); selected = driver.choose(energy, flags)
        literal = []
        for values, stopped in zip(energy, flags):
            candidates = np.flatnonzero(stopped)
            if not len(candidates): candidates = np.arange(len(values))
            literal.append(int(candidates[np.argmin(values[candidates])]))
        assert np.array_equal(selected, literal)
        idx = np.arange(len(energy)); old_ok = valid[idx, original]; new_ok = valid[idx, selected]
        assert np.array_equal(old_ok, z['n20'])
        has_cert = flags.any(axis=1)
        assert flags[idx[has_cert], selected[has_cert]].all()
        assert np.array_equal(selected[~has_cert], original[~has_cert])
        model = torch.load(ROOT / case['model'], weights_only=False)
        diagnostic = certificates(boards, z['inputs'], flags, model)
        name = f"{case['arm']}/u{case['updates']}/s{case['seed']}"; diagnostics[name] = diagnostic
        assert int((flags & ~valid).sum()) == rec['wrong_stops']
        row = dict(arm=case['arm'], updates=case['updates'], seed=case['seed'], unique_inputs=len(energy),
            original_correct=int(old_ok.sum()), certificate_first_correct=int(new_ok.sum()),
            recovered_inputs=int((new_ok & ~old_ok).sum()), regressed_inputs=int((old_ok & ~new_ok).sum()),
            changed_selection_inputs=int((original != selected).sum()), inputs_with_stopped_chain=int(has_cert.sum()),
            original_failures_with_stopped_chain=int((~old_ok & has_cert).sum()),
            wrong_selected_stopped=int((~new_ok & flags[idx, selected]).sum()),
            fallback_inputs=int((~has_cert).sum()), fallback_correct=int((new_ok & ~has_cert).sum()),
            coverage_numerator=int(z['found'].sum()), coverage_denominator=int(z['denom'].sum()),
            source_records_sha256=case['arrays_sha256'], model_sha256=case['model_sha256'])
        records.append(row)
        evidence.setdefault((case['arm'], case['updates']), []).append(dict(original=old_ok, selected=new_ok,
            found=z['found'], denom=z['denom']))
        identity_path = OUT / f"selections/{case['arm']}/u{case['updates']}/s{case['seed']}.npz"
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(identity_path, original_index=original, selected_index=selected, original_valid=old_ok, selected_valid=new_ok)
        sha(identity_path)
        if case['arm'] == 'adamw-decay' and case['seed'] == 0:
            timing = measure_selector(energy, flags)
    rows = []; primary = None
    for (arm, updates), zs in evidence.items():
        original = {'n20': np.stack([z['original'] for z in zs]), 'found': np.stack([z['found'] for z in zs])}
        selected = {'n20': np.stack([z['selected'] for z in zs]), 'found': original['found']}
        stats = A.bootstrap(selected, original, zs[0]['denom'])
        current = [r for r in records if r['arm'] == arm and r['updates'] == updates]
        row = dict(arm=arm, updates=updates, unique_inputs=6513, seeds=[0, 1, 2],
            original_accuracy_pct=100*float(original['n20'].mean()), certificate_first_accuracy_pct=100*float(selected['n20'].mean()),
            recovered_inputs=sum(r['recovered_inputs'] for r in current),
            regressed_inputs=sum(r['regressed_inputs'] for r in current),
            wrong_selected_stopped=sum(r['wrong_selected_stopped'] for r in current),
            fallback_inputs=sum(r['fallback_inputs'] for r in current), **stats)
        rows.append(row)
        if arm == 'adamw-decay': primary = row
    allowance_violations = sum(v['below_numerical_allowance_count'] for v in diagnostics.values())
    decision = 'supported' if primary['accuracy_one_sided_95_lower_pp'] >= .2 else 'falsified' if primary['accuracy_one_sided_95_upper_pp'] < .2 else 'inconclusive'
    sha(__file__); sha(SRC.parent / 'analyze.py')
    summary = dict(status='validated', utc=dt.datetime.now(dt.timezone.utc).isoformat(), hypothesis='H3d', decision=decision,
        primary=primary, rows=rows, certificate_diagnostics=diagnostics,
        certificate_interpretation='passes_predeclared_numerical_allowance' if not allowance_violations else 'rejected_numerical_allowance_violations',
        stopped_flags_beyond_numerical_allowance=allowance_violations,
        maximum_stopped_deficit=max(v['maximum_stopped_deficit'] or 0. for v in diagnostics.values()),
        selector_timing=timing, additional_search_moves=0, additional_search_flops=0,
        independent_saved_board_checks=len(records)*6513*20, limits=p['limits'],
        input_sha256=HASHES, audit_seconds=time.perf_counter()-start)
    save(OUT / 'summary.json', summary); save(PUBLIC / 'summary.json', summary)
    for path in [OUT / 'per_seed.csv', PUBLIC / 'per_seed.csv']:
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    names = {'adamw-decay': 'BP AdamW, decay 0.01', 'adam-nodecay': 'BP Adam, no decay', 'sgd-pre': 'SGD + geometry preconditioner'}
    lines = ['# H3d: learned-certificate-first output selection', '',
        f"**Frozen primary hypothesis: {decision}.** The primary gain target was at least 0.2 percentage points for the 4,000-update AdamW-decay model. This reanalysis preserves every existing chain, weight, stopping threshold and search operation. It selects the lowest-energy stopped chain if one exists, otherwise the original lowest-energy returned chain. It uses learned flags and energies only; puzzle rules enter independent evaluation afterward.", '',
        '| Method | Updates | Original N20 (%) | Certificate-first N20 (%) | Gain (pp) | Recovered / regressed input-seed pairs |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {names[row['arm']]} | {row['updates']:,} | {row['original_accuracy_pct']:.6f} | {row['certificate_first_accuracy_pct']:.6f} | {row['accuracy_difference_pp']:+.6f} | {row['recovered_inputs']} / {row['regressed_inputs']} |")
    lines += ['', f"Primary one-sided 95% bootstrap endpoints for the gain are [{primary['accuracy_one_sided_95_lower_pp']:.6f}, {primary['accuracy_one_sided_95_upper_pp']:.6f}] percentage points. Unique inputs are resampled with their three search repeats clustered inside; these are not independent training seeds.", '',
        f"Every selection identity matched a literal row-by-row stopped-subset implementation. Original identities reproduced saved N20 correctness, and every no-stopped-chain fallback retained the exact original identity. The independent audit checked {len(records)*6513*20:,} returned boards against the complete 724-solution set and givens. Coverage, N1, search moves and search FLOPs remain unchanged because no search or returned chain changed.", '',
        f"Float64 full-board recomputation found {allowance_violations} stopped flags outside the predefined 0.001 absolute-energy allowance; the largest stopped-certificate deficit was {summary['maximum_stopped_deficit']:.9g}. Strict negative-gap counts and extrema are retained per record in the JSON. Capped outputs can be earlier best-energy boards, so their flags are not equated with a direct certificate on the returned state. The learned condition is not a general puzzle-correctness proof.", '',
        f"There is no additional search. Selection itself has a measured CPU cost: original median {timing['medians_ms']['original']:.6f} ms; certificate-first median {timing['medians_ms']['certificate_first']:.6f} ms; overhead {timing['median_overhead_ms']:.6f} ms for all 6,513 inputs with 20 chains (101 alternating measurements after ten warmups each). This excludes I/O and the diagnostic checker and is not a search-speedup claim.", '',
        'This is an adaptive follow-up on already-used official test outputs. The policy differs from the paper convention, so its results are a derivative experiment rather than a replacement reproduction score. No performance-based setting or checkpoint selection was performed inside this frozen reanalysis.', '',
        '[Frozen protocol](protocol.json), [selector](driver.py), [independent auditor](analyze.py), [tracked results](../../../../results/research/loki/h3d/summary.json), [per-seed counts](../../../../results/research/loki/h3d/per_seed.csv).']
    (SRC / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (PUBLIC / 'README.md').write_text('# H3d: learned-certificate-first output selection\n\n' +
        f"The frozen >=0.2-point primary gain hypothesis was **{decision}**. The observed AdamW-decay gain was {primary['accuracy_difference_pp']:+.6f} percentage points. No search was rerun; CPU selector overhead is measured separately.\n\n" +
        'This adaptive saved-output policy changes the paper selection convention. Learned stopped flags steer selection; independent rules are used only afterward to score results.\n\n' +
        '[Full report and protocol](../../../../research/20261114/loki/h3d/REPORT.md), [summary and hashes](summary.json), [per-seed counts](per_seed.csv).\n')
    print(json.dumps(dict(status='validated', decision=decision, primary=primary,
        max_certificate_deficit=summary['maximum_stopped_deficit'], certificate_allowance_violations=allowance_violations,
        selector_overhead_ms=timing['median_overhead_ms'], audit_seconds=summary['audit_seconds'])))


if __name__ == '__main__': main()
