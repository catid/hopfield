#!/usr/bin/env python3
"""Export validated Table7, D.2 and Figure4 artifacts without running a GPU."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from objective_queue import ROOT, digest, now, write


def read(path): return json.loads(Path(path).read_text())


def figure_curves(objectives, figure):
    config = read(ROOT / 'configs/figure4_jobs.json')
    states = {j['id']: j for j in figure['jobs']}
    ready = {}
    for job in objectives['jobs']:
        if job['task'] == 'sudoku' and job['family'] == 'table7_objectives' and job['status'] == 'complete':
            ready.setdefault(job['method'], {})[job['seed']] = job['result']['ready_s']
    rows = []
    cache_root = ROOT / 'artifacts/figure4/curve_cache'; cache_root.mkdir(parents=True, exist_ok=True)
    for job in config['jobs']:
        row = {k: job[k] for k in ('id', 'method', 'seed', 'lane', 'dense', 'run_relative')}
        row['status'] = states[job['id']]['status']
        if row['status'] != 'complete': rows.append(row); continue
        host = job['lane'].split(':')[0]
        run = ROOT / 'artifacts/figure4' / host / job['run_relative']
        execution_path = ROOT / 'artifacts/figure4' / host / 'jobs' / job['id'] / 'execution.json'
        execution = read(execution_path)
        overheads = ready.get(job['method'], {})
        if set(overheads) != {0, 1, 2}:
            row.update(status='incomplete_ready_times', reason='Standard final-run readiness times for all three seeds are required')
            rows.append(row); continue
        if job['dense']:
            proof = execution['validation'].get('clock_match', {})
            if not proof.get('matched'):
                row.update(status='incomplete_dense_clock_validation', reason='Dense and final checkpoint trajectories must match before borrowing the final clock')
                rows.append(row); continue
            clock = ROOT / 'artifacts/objectives' / host / job['clock_source']
            if not clock.exists() or digest(clock) != proof['final_log_sha256']:
                row.update(status='incomplete_dense_clock_collection'); rows.append(row); continue
        else: clock = run / 'log.jsonl'
        key = hashlib.sha256(json.dumps({'execution': digest(execution_path), 'ready': overheads,
                 'clock_sha256': digest(clock), 'policy': 'median_three_ready_times;linear_time_interpolation;v1'}, sort_keys=True).encode()).hexdigest()
        cache_path = cache_root / (job['id'] + '.json')
        if cache_path.exists() and read(cache_path).get('cache_key') == key:
            rows.append(read(cache_path)['row']); continue
        own = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
        own = [r for r in own if 'update' in r]
        timing = [json.loads(line) for line in clock.read_text().splitlines()]
        fit = {r['update']: r['fit_s'] for r in timing if 'update' in r}
        n = min(20, max(fit) - 1)
        startup = fit[1] - statistics.median(fit[k + 1] - fit[k] for k in range(1, n + 1))
        probes = ({r['update']: r['probe'] for r in own if 'probe' in r} if job['dense'] else
                  {int(k): v for k, v in read(run / 'probes.json').items()})
        assert set(probes) == set(execution['validation']['probe_updates'])
        assert all(p['n'] == 512 for p in probes.values())
        # The release supplies only three final-run overhead measurements. Use
        # a declared common median, preserving all three raw measurements.
        ready_s = statistics.median(overheads.values())
        row.update(status='complete', schedule_completed=execution['validation']['schedule_completed'],
                   completed_updates=execution['validation']['last_update'], n_probes=len(probes),
                   startup_s=startup, startup_estimate_later_updates=n, ready_s=ready_s,
                   ready_s_from_standard_seeds=overheads, clock_from=str(clock.relative_to(ROOT)),
                   curve=[{'update': k, 'solved': probes[k]['solved'], 'n': 512,
                           'solved_percent': 100 * probes[k]['solved'] / 512, 'fit_s': fit[k],
                           'paper_adjusted_fit_s': fit[k] - startup + ready_s} for k in sorted(probes)],
                   execution_sha256=digest(execution_path))
        write(cache_path, {'cache_key': key, 'row': row}); rows.append(row)
    out = {'generated_utc': now(), 'rows': rows, 'expected_runs': 100,
           'validated_runs': sum(r['status'] == 'complete' for r in rows),
           'complete': len(rows) == 100 and all(r['status'] == 'complete' for r in rows),
           'time_policy': 'Native fit clock minus per-run startup plus the median of three standard-final readiness measurements for that method. Dense runs borrow validated matched final-run clocks.',
           'plot_policy': 'Linear interpolation in elapsed seconds; zero before the first recorded probe, hold the final value afterwards; mean and25th/75th percentiles over exactly20seeds. This plotting/ready-time pooling convention is explicitly reconstructed because the release omits its final plotting script.',
           'hardware_note': 'Local Blackwell timings with host/GPU provenance; not H200 speed reproduction.'}
    write(ROOT / 'artifacts/figure4/curves.json', out)
    return out


def render(curves):
    methods = ['pl', 'mpf', 'cd', 'bptt1', 'bptt']
    groups = {m: [r for r in curves['rows'] if r['method'] == m and r['status'] == 'complete'] for m in methods}
    if not all(len(rows) == 20 for rows in groups.values()): return {'status': 'incomplete', 'reason': 'All five methods require20validated seeds'}
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT / 'artifacts/figure4/figures'; folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
    cached = folder / 'figure4.json'
    if cached.exists() and read(cached).get('inputs_sha256') == fingerprint: return read(cached)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    names = {'pl': 'Pseudolikelihood', 'mpf': 'Minimum probability flow', 'cd': 'Contrastive divergence',
             'bptt1': 'One-step BPTT', 'bptt': 'Full BPTT (60k schedule)'}
    stats = {}
    for method, rows in groups.items():
        xs = [np.array([p['paper_adjusted_fit_s'] for p in r['curve']]) for r in rows]
        ys = [np.array([p['solved_percent'] for p in r['curve']]) for r in rows]
        assert all((x > 0).all() and (np.diff(x) > 0).all() for x in xs)
        knots = np.unique(np.concatenate(xs))
        vals = np.stack([np.interp(knots, x, y, left=0., right=y[-1]) for x, y in zip(xs, ys)])
        mean = vals.mean(0); lo, hi = np.percentile(vals, [25, 75], axis=0)
        line, = ax.plot(knots, mean, label=names[method]); ax.fill_between(knots, lo, hi, color=line.get_color(), alpha=.18)
        hits = np.flatnonzero(mean >= 99.)
        t99 = None
        if len(hits):
            k = hits[0]; t99 = float(knots[k]) if k == 0 else float(knots[k-1] + (99-mean[k-1])/(mean[k]-mean[k-1])*(knots[k]-knots[k-1]))
            ax.plot(t99, 99., 'o', color=line.get_color())
        stats[method] = {'n_seeds': 20, 'mean_curve_T99_s': t99, 'final_mean_solved_percent': float(mean[-1]),
                         'all_schedules_completed': all(r['schedule_completed'] for r in rows)}
    ax.set_xscale('log'); ax.set_xlabel('Adjusted local training time (s)'); ax.set_ylabel('Solved development puzzles (%)')
    ax.set_ylim(0, 101); ax.grid(alpha=.2); ax.legend(); ax.set_title('CRAM objective curves: local reproduction, 20 seeds')
    for suffix in ('png', 'pdf', 'svg'): fig.savefig(folder / f'figure4.{suffix}', dpi=180)
    plt.close(fig)
    result = {'status': 'complete', 'inputs_sha256': fingerprint, 'methods': stats,
              'policy': curves['plot_policy'], 'hardware_note': curves['hardware_note']}
    write(cached, result); return result


def export():
    objectives = read(ROOT / 'artifacts/objectives/summary.json')
    figure = read(ROOT / 'artifacts/figure4/summary.json'); d2 = read(ROOT / 'artifacts/d2/summary.json')
    curves = figure_curves(objectives, figure); plot = render(curves)
    out = ROOT / 'artifacts/objective_studies'; out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, row in objectives['rows'].items():
        table.append({'recipe': name, 'status': row['status'], 'completed_seeds': len(row['completed_seeds']),
            'solved_percent_mean': row.get('solved_percent_mean'), 'solved_percent_sample_sd': row.get('solved_percent_sample_sd'),
            'T99_s': row.get('paper_adjusted_T99_s_median'), 'T99_common_update': row.get('T99_update_common_mean'),
            'FLOPs99': row.get('T99_fullcount_flop'), 'n_test_per_seed': row.get('n_test_per_seed')})
    with (out / 'table7.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    d2_valid = len(d2['jobs']) == 9 and all(j['status'] == 'complete' and j.get('probe_n') == 512 and
                  j.get('test_evaluated') is False for j in d2['jobs'])
    complete = objectives.get('scope_complete', False) and d2_valid and curves['complete'] and plot['status'] == 'complete'
    limits = [r['id'] for r in curves['rows'] if r['status'] == 'complete' and not r['schedule_completed']]
    limits += [j['id'] for j in d2['jobs'] if j['status'] == 'complete' and not j.get('schedule_completed')]
    limits += [j['id'] for j in objectives['jobs'] + objectives.get('external_jobs', [])
               if j['status'] == 'complete' and (j['final']['stop'] != 'schedule' or
                   max(r['update'] for r in j['final']['curve']) != j['final']['updates'])]
    summary = {'generated_utc': now(), 'status': 'complete' if complete and not limits else 'incomplete',
        'all_required_inputs_validated': complete, 'incomplete_schedules': limits,
        'table7': {'complete': objectives.get('scope_complete', False), 'managed': objectives['status_counts'],
                   'external': objectives.get('external_status_counts'), 'rows': table},
        'd2': {'complete': d2_valid, 'status_counts': d2['status_counts']},
        'figure4': {'complete': curves['complete'], 'validated_runs': curves['validated_runs'], 'expected_runs': 100, 'plot': plot},
        'scope_note': 'Supplementary ALT-free/Adam Maze curves are legacy release artifacts, not a reported arXiv-v1 figure/claim, and are excluded from required paper reproduction.'}
    write(out / 'summary.json', summary)
    lines = ['# Objective-study results', '', f'Status: **{summary["status"]}**. Updated {summary["generated_utc"]}.', '',
             'Table 7 uses the first common probe update where the three-run mean reaches 99%; individual seed crossings remain in the JSON evidence. Published frozen choices were reused, without a local tuning sweep. All measured timings are local Blackwell results.', '',
             '| Recipe | Status | Solved (%) | Adjusted T99 (s) | FLOPs99 |', '|---|---|---:|---:|---:|']
    for r in table:
        accuracy = '' if r['status'] != 'complete' else f'{r["solved_percent_mean"]:.5f} ± {r["solved_percent_sample_sd"]:.5f}'
        lines.append(f'| {r["recipe"]} | {r["status"]} | {accuracy} | {r["T99_s"] if r["T99_s"] is not None else "—"} | {r["FLOPs99"] if r["FLOPs99"] is not None else "—"} |')
    lines += ['', f'Figure 4: {curves["validated_runs"]}/100 curves have validated plotting inputs. D.2: {d2["status_counts"]}.', '',
              'A complete report requires all 84 Table 7 fits/tests, all 9 D.2 development-only fits, all 100 Figure 4 curves, matched dense clocks, and operation counts. Missing, failed, unmatched or truncated inputs keep the report incomplete.', '', summary['scope_note']]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'figure4_validated': curves['validated_runs'], 'output': str(out)}))
    return summary


if __name__ == '__main__': export()
