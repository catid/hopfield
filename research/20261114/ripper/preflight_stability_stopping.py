"""Synthetic-only H1bu balanced-prefix, stopping, endpoint and seed gates."""
import json, platform, time
import numpy as np
from stability_stopping_study import ROOT, HERE, PREFIXES, sha, save, make_masks, observe, available_graph, stopping_rule
from audit_stability_stopping import reference_masks, reference_stopping, reference_observe, complete_gate, literal_account, aggregate
from balanced_mask_math import make_balanced_masks
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
    proposed = {name: set(range(design['seeds'][name], design['seeds'][name]+32)) for name in ['source', 'balanced_mask', 'noise']}
    proposed['solver'] = {design['seeds']['solver_base']+41*index+degree for index in range(128) for degree in range(1, 41)}
    assert sum(len(v) for v in proposed.values()) == len(set().union(*proposed.values())) == 5216
    assert not values & set().union(*proposed.values())
    result = dict(status='passed', frozen_protocol_sha256=files, historical_explicit_seed_count=len(values),
                  new_ranges={name: [min(v), max(v)] for name, v in proposed.items()}, new_counts={name: len(v) for name, v in proposed.items()}, collisions=[],
                  scope='Disjoint from all explicit integer seed-valued fields in frozen protocol snapshots; does not expand every historical formula or prove PRNG independence.')
    save(HERE/'stability-stopping-seed-audit.json', result)
    return result


def cycle_graph(step):
    matrix = np.zeros((81, 81), bool)
    for cell in range(81):
        other = (cell+step) % 81
        matrix[cell, other] = matrix[other, cell] = True
    ii, jj = np.triu_indices(81, 1)
    return 2*matrix[ii, jj].astype(np.int64)


def toy_selector(degree, primal):
    values = [0, 3]
    for d in range(1, 40):
        values.append(2*values[-1]-values[-2]+(40 if d == degree else 1))
    records = [dict(certificate=dict(certified=True, primal_objective_twice=values[d])) for d in range(1, 41)]
    arrays = {f'd{degree:02d}_primal_twice': primal}
    return arrays, records


def endpoint_fixture(successes, unknown, stops):
    rows = []
    stopped = []
    for bank in range(32):
        kind = 'success' if bank < successes else 'unresolved' if bank < successes+unknown else 'resolved_failure_nonexact'
        n = stops[bank]
        for pi, prefix in enumerate(PREFIXES):
            rows.append(dict(index=4*bank+pi, seed=bank, rows=prefix, classification=kind, selected_degree=20,
                             fixed_degree20=dict(classification=kind), top810=dict(classification=kind)))
        stopped.append(dict(seed=bank, classification=kind, stop_rows=n, fixed360_classification=kind,
                            early_wrong_stable=n < 360 and kind == 'resolved_failure_nonexact', available_prefixes=[True]*4))
    return aggregate(rows, stopped)


def main():
    assert platform.node() == 'ripper'
    path = HERE/'stability-stopping-preflight.json'
    assert not path.exists() and not (HERE/'protocol-stability-stopping.json').exists()
    design = json.loads((HERE/'stability-stopping-proposal.json').read_text())
    seeds = seed_audit(design)
    incidence = verify_incidence()
    mask_checks = []
    for seed in [983001, 983002, 983003]:
        mask = make_masks(seed)
        assert np.array_equal(mask, reference_masks(seed)) and np.array_equal(mask[:180], make_balanced_masks(seed))
        assert mask.shape == (360, 81) and (mask.sum(1) == 72).all()
        for n in PREFIXES:
            product = mask[:n].astype(np.int64).T @ mask[:n].astype(np.int64)
            cycles = n//90
            assert np.array_equal(product, 71*cycles*np.ones((81, 81), np.int64)+9*cycles*np.eye(81, dtype=np.int64))
        for stop in [180, 270, 360]:
            q = literal_account(mask, stop)
            assert q['full_physical_noisy_union'] == 25920 and q['virtual_stopped_queries'] == 72*stop
        mask_checks.append(dict(seed=seed, first180_frozen_helper_identical=True, all_four_exposure_matrices_exact=True))
    rng = np.random.default_rng(983004)
    clean = rng.integers(0, 9, (360, 81), dtype=np.int8)
    noise = dict(iid_uniforms=rng.random((360, 81)), iid_offsets=rng.integers(1, 9, (360, 81), dtype=np.int8))
    expected = {}
    for n in PREFIXES:
        actual = observe(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        expected[n] = reference_observe(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        for key, value in expected[n].items():
            assert np.array_equal(actual[key], value)
    for n in PREFIXES:
        assert np.array_equal(expected[n]['values'], expected[360]['values'][:n])
    gates = complete_gate(mask, clean, noise, expected)
    a, b, c = [cycle_graph(step) for step in [1, 2, 4]]
    cases = []
    for graphs, expected_stop in [([a, a, b, b], 180), ([a, b, b, c], 270), ([a, b, c, c], 360),
                                  ([None, a, a, b], 270), ([a, None, a, a], 360), ([a, b, c, None], 360), ([None]*4, 360)]:
        result = stopping_rule(graphs)
        assert result == reference_stopping(graphs) and result['stop_rows'] == expected_stop
        cases.append(result)
    arrays, records = toy_selector(2, a)
    assert np.array_equal(available_graph(arrays, records), a)
    records[10]['certificate']['certified'] = False
    assert available_graph(arrays, records) is None
    arrays, records = toy_selector(1, a//2)
    assert available_graph(arrays, records) is None
    endpoints = []
    for successes, unknown, stops, expected in [
            (32, 0, [180]*16+[360]*16, ('supported', 'supported')),
            (30, 1, [270]*32, ('inconclusive', 'supported')),
            (30, 0, [180]*15+[360]*17, ('falsified', 'falsified')),
            (31, 1, [360]*32, ('supported', 'falsified'))]:
        result = endpoint_fixture(successes, unknown, stops)
        assert (result['endpoints']['stopped_exact']['decision'], result['endpoints']['virtual_rows']['decision']) == expected
        assert result['endpoints']['stopped_exact']['upper'] == successes+unknown
        endpoints.append(result['endpoints'])
    wrong = endpoint_fixture(31, 0, [180]*32)
    assert wrong['early_wrong_stable_banks'] == [31] and wrong['endpoints']['stopped_exact']['lower'] == 31
    before = time.perf_counter()
    make_masks(983005)
    mask_seconds = time.perf_counter()-before
    dependencies = json.loads((HERE/'higher-noise-preflight.json').read_text())['source_sha256']
    files = {ROOT/rel for rel in dependencies}
    files |= {HERE/name for name in ['stability_stopping_study.py', 'audit_stability_stopping.py', 'preflight_stability_stopping.py', 'run_stability_stopping.py',
                                     'higher_noise_study.py', 'audit_higher_noise.py', 'balanced_mask_math.py', 'audit_balanced_mask_math.py']}
    result = dict(status='passed', host=platform.node(), scope='Synthetic masks, IID arrays and controller/endpoint fixtures only; no selected study IDs or LP outcomes.',
                  incidence=incidence, mask_checks=mask_checks, literal_noise_and_observation_prefixes=True, complete_label_gates=gates,
                  controller_cases=cases, selector_certification_and_integrality_cases=True, endpoint_cases=endpoints,
                  early_wrong_stable_retained=True, virtual_and_full_physical_costs_separate=True,
                  synthetic_four_cycle_mask_seconds=mask_seconds, seed_audit_sha256=sha(HERE/'stability-stopping-seed-audit.json'),
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sorted(files)})
    save(path, result)
    print(json.dumps(dict(status='passed', source_files=len(files), seed_ranges=seeds['new_ranges'], synthetic_four_cycle_mask_seconds=mask_seconds)))


if __name__ == '__main__':
    main()
