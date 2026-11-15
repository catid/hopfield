"""Synthetic H1bv three-cycle, paired-bound and batching equivalence gates."""
import json, platform
import numpy as np
from fixed_270_confirm_study import ROOT, HERE, PREFIXES, sha, save, make_masks, observe
from audit_fixed_270_confirm import reference_masks, reference_observe, complete_gate, literal_account, aggregate
from balanced_mask_math import make_balanced_masks
from stability_stopping_study import make_masks as parent_four_cycle
from audit_balanced_mask_math import verify_incidence


def seed_audit(design):
    values = set()
    files = {}
    def scan(value, context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, context)
        elif context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            files[str(path.relative_to(ROOT))] = sha(path)
            scan(data)
    proposed = {name: set(range(design['seeds'][name], design['seeds'][name]+64)) for name in ['source', 'balanced_mask', 'noise']}
    proposed['solver'] = {design['seeds']['solver_base']+41*index+degree for index in range(128) for degree in range(1, 41)}
    assert sum(len(v) for v in proposed.values()) == len(set().union(*proposed.values())) == 5312
    assert not values & set().union(*proposed.values())
    result = dict(status='passed', frozen_protocol_sha256=files, historical_explicit_seed_count=len(values),
                  new_ranges={name: [min(v), max(v)] for name, v in proposed.items()}, new_counts={name: len(v) for name, v in proposed.items()}, collisions=[],
                  scope='Disjoint from all explicit integer seed-valued fields in frozen protocol snapshots; does not expand every historical formula or prove PRNG independence.')
    save(HERE/'fixed-270-confirm-seed-audit.json', result)
    return result


def endpoint_fixture(short_success, short_unknown, full_success, full_unknown):
    rows = []
    for bank in range(64):
        for pi, n in enumerate(PREFIXES):
            successes, unknown = (short_success, short_unknown) if pi == 0 else (full_success, full_unknown)
            kind = 'success' if bank < successes else 'unresolved' if bank < successes+unknown else 'resolved_failure_nonexact'
            rows.append(dict(index=2*bank+pi, seed=bank, rows=n, classification=kind, selected_degree=20,
                             fixed_degree20=dict(classification=kind), top810=dict(classification=kind)))
    return aggregate(rows), rows


def main():
    assert platform.node() == 'ripper'
    path = HERE/'fixed-270-confirm-preflight.json'
    assert not path.exists() and not (HERE/'protocol-fixed-270-confirm.json').exists()
    design = json.loads((HERE/'fixed-270-confirm-proposal.json').read_text())
    seeds = seed_audit(design)
    benchmark = HERE/'fixed-270-preparation-benchmark.json'
    timings = json.loads(benchmark.read_text())
    assert timings['status'] == 'passed' and timings['all_outputs_identical_within_kind']
    assert timings['script_sha256'] == sha(HERE/'benchmark_fixed_270_preparation.py')
    records = {(r['kind'], r['mode']): r for r in timings['results']}
    assert records['masks', 'batch']['median_wall_seconds'] < records['masks', 'fresh']['median_wall_seconds']
    assert records['counts', 'persistent']['median_wall_seconds'] < records['counts', 'fresh']['median_wall_seconds']
    incidence = verify_incidence()
    for seed in [985001, 985002, 985003]:
        mask = make_masks(seed)
        assert mask.shape == (270, 81) and np.array_equal(mask, reference_masks(seed))
        assert np.array_equal(mask[:180], make_balanced_masks(seed)) and np.array_equal(mask, parent_four_cycle(seed)[:270])
        assert (mask.sum(1) == 72).all()
        for n in PREFIXES:
            actual = mask[:n].astype(np.int64).T @ mask[:n].astype(np.int64)
            k = n//90
            assert np.array_equal(actual, 71*k*np.ones((81, 81), np.int64)+9*k*np.eye(81, dtype=np.int64))
        assert literal_account(mask) == dict(full_physical_noisy_union=19440, prefix180_deployment=12960, additional270_queries=6480)
    rng = np.random.default_rng(985004)
    clean = rng.integers(0, 9, (270, 81), dtype=np.int8)
    noise = dict(iid_uniforms=rng.random((270, 81)), iid_offsets=rng.integers(1, 9, (270, 81), dtype=np.int8))
    expected = {}
    for n in PREFIXES:
        produced = observe(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        expected[n] = reference_observe(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        for key, value in expected[n].items():
            assert np.array_equal(produced[key], value)
    assert np.array_equal(expected[180]['values'], expected[270]['values'][:180])
    gates = complete_gate(mask, clean, noise, expected)
    endpoints = []
    for short, short_unknown, full, full_unknown, expected_decisions in [
            (53, 0, 61, 0, ('supported', 'supported')),
            (53, 1, 60, 1, ('inconclusive', 'inconclusive')),
            (55, 0, 60, 0, ('falsified', 'falsified')),
            (55, 1, 63, 0, ('supported', 'inconclusive'))]:
        result, _ = endpoint_fixture(short, short_unknown, full, full_unknown)
        actual = result['endpoints']
        assert (actual['exact270']['decision'], actual['paired_gain']['decision']) == expected_decisions
        assert actual['paired_gain']['lower'] == full-short-short_unknown and actual['paired_gain']['upper'] == full+full_unknown-short
        endpoints.append(actual)
    _, rows = endpoint_fixture(61, 0, 61, 0)
    rows[1]['classification'] = 'resolved_failure_nonexact'
    rows[127]['classification'] = 'success'
    regression = aggregate(rows)
    assert regression['paired_changes']['prefix180_wins'] == [0] and regression['paired_changes']['full270_wins'] == [63]
    assert regression['endpoints']['paired_gain']['lower'] == regression['endpoints']['paired_gain']['upper'] == 0
    worker_jobs = [list(range(worker, 64, 8)) for worker in range(8)]
    assert sorted(job for part in worker_jobs for job in part) == list(range(64)) and all(len(part) == 8 for part in worker_jobs)
    dependencies = json.loads((HERE/'stability-stopping-preflight.json').read_text())['source_sha256']
    files = {ROOT/rel for rel in dependencies}
    files |= {HERE/name for name in ['fixed_270_confirm_study.py', 'audit_fixed_270_confirm.py', 'preflight_fixed_270_confirm.py', 'run_fixed_270_confirm.py',
                                     'benchmark_fixed_270_preparation.py', 'stability_stopping_study.py']}
    result = dict(status='passed', host=platform.node(), scope='Unrelated synthetic labels/masks/paired outcomes and startup timing only; no study IDs/masks/noise or LP outcomes.',
                  incidence=incidence, parent180_and270_mask_prefixes_identical=True, exact_exposure_matrices=True, literal_noise_nesting=True,
                  full_label_gates=gates, endpoint_cases=endpoints, paired_regression_retained=True, worker_bank_partition=worker_jobs,
                  benchmark_sha256=sha(benchmark), seed_audit_sha256=sha(HERE/'fixed-270-confirm-seed-audit.json'),
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sorted(files)})
    save(path, result)
    print(json.dumps(dict(status='passed', source_files=len(files), seed_ranges=seeds['new_ranges'], batching_equivalence_and_median_speed_gate=True)))


if __name__ == '__main__':
    main()


