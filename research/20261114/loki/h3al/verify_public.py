"""Public-only, independent grouped-union DP and complete integer census audit."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def grouped_union_dp(frequency, maximum):
    """Count selected subsets by draw size and union using binomial convolutions."""
    width = len(frequency)
    dp = [[0] * width for _ in range(maximum + 1)]
    dp[0][0] = 1
    processed = 0
    for mask, count in enumerate(frequency):
        if not count:
            continue
        choices = [math.comb(count, j) for j in range(min(count, maximum) + 1)]
        nxt = [[0] * width for _ in range(maximum + 1)]
        for k in range(min(processed, maximum) + 1):
            for union, weight in enumerate(dp[k]):
                if weight:
                    nxt[k][union] += weight
                    for j in range(1, min(count, maximum-k) + 1):
                        nxt[k+j][union | mask] += weight * choices[j]
        processed += count
        dp = nxt
    return dp


def histogram(columns):
    """Integer multiplicities, rather than the experiment's Boolean union."""
    h = np.zeros(226**3, np.int64)
    cells = 15*np.arange(15)[None, :] + columns.astype(np.int64)
    for k in (1, 2, 3):
        indices = np.array(list(itertools.combinations(range(15), k)))
        powers = np.array([226**(2-j) for j in range(k)], np.int64)
        encoded = ((cells[:, indices]+1)*powers).sum(-1).ravel()
        h += np.bincount(encoded, minlength=226**3)
    assert h.sum() == 575*len(columns)
    return h


def verify(root, require_summary=True):
    start = time.perf_counter()
    R = Path(root)
    P = R/'results/research/loki/h3al'
    S = R/'research/20261114/loki/h3al'
    read = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    p = read(S/'protocol.json')
    raw = read(P/'study-result.json')
    gate = read(P/'capacity-gate.json')
    source = read(P/'source-outcomes.json')
    seal = read(P/'filtering-completeness-seal.json')
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    check('complete stage', raw['status'] == gate['status'] == 'complete_verified')
    check('full budget', gate['aggregate_cpu_seconds'] <= 120 and gate['wall_seconds'] <= 180 and not gate['monitor_expired'])
    check('protocol identity', raw['protocol_sha256'] == source['protocol_sha256'] == sha(S/'protocol.json'))
    check('stage identities', raw['source_sha256'] == sha(S/'study.py') and source['source_sha256'] == sha(S/'source_outcomes.py'))
    check('source sealed before target filtering', source['utc'] < seal['utc'] < raw['utc'])
    check('source receipt identity', raw['source_outcomes_sha256'] == sha(P/'source-outcomes.json'))
    check('gate and stage results', all(gate[k] == raw[k] for k in raw if k != 'utc'))
    for rel, digest in p['source_sha256'].items():
        check('frozen source '+rel, sha(R/rel) == digest)
    for rel, digest in p['target_data_sha256'].items():
        check('frozen target '+rel, sha(R/rel) == digest)
    population = np.load(P/'source-population-n13.npy')
    check('population identity', sha(P/'source-population-n13.npy') == p['source_population']['sha256'])
    check('population shape', population.shape == (1932, 169) and population.dtype == bool and (population.sum(1) == 13).all())
    observed = []
    for board in population:
        offsets = set()
        for a, b in itertools.combinations(np.flatnonzero(board), 2):
            dr, dc = int(b)//13-int(a)//13, int(b)%13-int(a)%13
            assert dr and dc and dr != abs(dc) and (dr, abs(dc)) not in ((1, 3), (3, 1))
            offsets.add((dr, dc))
        observed.append(offsets)
    check('all1932 source labels independently valid', True)
    # A separate tiny enumeration validates the auditor's DP, not the runner.
    fixture_masks = [0, 1, 3, 1, 4, 5, 0, 7]
    dp = grouped_union_dp(np.bincount(fixture_masks, minlength=8).tolist(), 8)
    for k in range(9):
        literal = [0]*8
        for selection in itertools.combinations(fixture_masks, k):
            union = 0
            for m in selection:
                union |= m
            literal[union] += 1
        check(f'audit DP exhaustive draw{k}', literal == dp[k])
    supports = {r['group']: np.load(P/f"support{r['group']}/F.npz")['F'] for r in source['supports']}
    membership = {group: [] for group in supports}
    cache = {}
    all_seen = set().union(*observed)
    for bank, cfg in enumerate(p['models']):
        b = source['banks'][bank]
        model_path = R/cfg['model_path'].replace('artifacts/research/20261114/loki', 'results/research/loki')
        design_path = R/cfg['source_design_path'].replace('artifacts/research/20261114/loki', 'results/research/loki')
        check(f'bank{bank}inherited identities', sha(model_path) == p['source_data_sha256'][cfg['model_path']] == b['base_model_sha256'] and sha(design_path) == p['source_data_sha256'][cfg['source_design_path']])
        model = np.load(model_path)
        base = model['F']
        finite = {tuple(x) for x in np.load(design_path)['offsets']}
        check(f'bank{bank}positive support model', not model['h'].any() and (model['J'][base.astype(bool)] > 0).all() and not model['J'][~base.astype(bool)].any())
        classes = sorted((dr, dc) for dr, dc in finite & all_seen if base[max(0, -dc), 15*dr+max(0, dc)])
        check(f'bank{bank}observable classes', classes == [tuple(x) for x in b['classes']] and len(classes) == b['class_count'] <= 8)
        masks = [sum(1 << i for i, offset in enumerate(classes) if offset in seen) for seen in observed]
        frequency = np.bincount(masks, minlength=1 << len(classes)).tolist()
        patterns = np.load(P/f'b{bank}/source-patterns.npz')
        check(f'bank{bank}all joint source masks', np.array_equal(masks, patterns['masks']) and np.array_equal(frequency, patterns['frequency']) and np.array_equal(classes, patterns['classes']) and frequency == b['frequency'])
        check(f'bank{bank}pattern hash', sha(P/f'b{bank}/source-patterns.npz') == b['source_patterns_sha256'])
        key = tuple(frequency)
        if key not in cache:
            cache[key] = grouped_union_dp(frequency, 64)
        for dose in p['doses']:
            masses = cache[key][dose]
            check(f'bank{bank}dose{dose}every exact mass', masses == list(map(int, b['draw_masses'][str(dose)])) and all(m >= 0 for m in masses) and sum(masses) == int(b['total_masses'][str(dose)]) == math.comb(1932, dose))
        for mask, group in enumerate(b['mask_to_support']):
            F = base.copy()
            for a, c in itertools.combinations(range(225), 2):
                offset = (c//15-a//15, c%15-a%15)
                if offset in classes and mask & (1 << classes.index(offset)):
                    F[a, c] = F[c, a] = 0
            check(f'bank{bank}mask{mask}complete support', np.array_equal(F, supports[group]))
            membership[group].append(dict(bank=bank, mask=mask))
    check('80 masks retained', sum(len(b['mask_to_support']) for b in source['banks']) == source['total_bank_masks'] == 80)
    check('32 distinct complete supports', len(supports) == source['unique_supports'] == 32 and len({F.tobytes() for F in supports.values()}) == 32)
    for record in source['supports']:
        group = record['group']
        check(f'support{group}identity and all members', record['members'] == membership[group] and sha(P/f'support{group}/F.npz') == record['F_npz_sha256'] and hashlib.sha256(supports[group].tobytes()).hexdigest() == record['support_sha256'])
    ref = R/p['evaluation']['reference_root']
    truth = np.load(ref/'truth/cache.npz')['columns']
    trueF = np.load(ref/'truth/F.npy')
    universe = np.load(ref/'input-universe.npz')
    keys, clues = universe['eligible_keys'], universe['eligible_clues']
    with tempfile.TemporaryDirectory(prefix='h3al-public-') as temp:
        temp = Path(temp)
        binary = temp/'truth'
        subprocess.run(['g++', '-O3', '-std=c++17', '-DNDEBUG', '-Wall', '-Wextra', '-Werror', str(R/'research/20261114/loki/h3aj/truth_camel.cpp'), '-o', str(binary)], check=True)
        for n, expected in ((13, np.flatnonzero(population).reshape(-1, 13) % 13), (15, truth)):
            output = temp/f'n{n}.bin'
            result = subprocess.run([str(binary), str(n), str(output)], capture_output=True, text=True, check=True)
            complete = np.fromfile(output, np.uint8).reshape(-1, n)
            check(f'N{n}independent complete enumeration', len(complete) == int(result.stdout) == len(expected) and np.array_equal(np.unique(complete, axis=0), np.unique(expected, axis=0)))
    for a, b in itertools.combinations(range(225), 2):
        dr, dc = abs(a//15-b//15), abs(a%15-b%15)
        assert trueF[a, b] == trueF[b, a] == (not dr or not dc or dr == dc or (dr, dc) in ((1, 3), (3, 1)))
    check('independent literal true graph', not np.diag(trueF).any())
    check('complete filtering proof for all32 supports before scoring', all((F >= trueF).all() for F in supports.values()))
    check('filtering receipt', seal['status'] == 'passed' and seal['supports'] == 32 and seal['all_repaired_F_contain_true_F'] and seal['reference_cache_sha256'] == sha(ref/'truth/cache.npz') and seal['reference_universe_sha256'] == sha(ref/'input-universe.npz'))
    h = histogram(truth)
    check('complete reference key census', np.array_equal(np.flatnonzero(h), universe['all_keys']) and np.array_equal(h[h > 0], universe['all_true_counts']) and np.array_equal(h[keys], universe['eligible_true_counts']))
    check('same exact exclusions', np.array_equal(np.setdiff1d(universe['all_keys'], universe['excluded_keys']), keys) and len(keys) == 831605 and len(truth) == 26960)
    check('key clue decoding', np.array_equal(clues, (keys % 226 > 0).astype(int) + (keys//226 % 226 > 0) + (keys//226**2 > 0)))
    cells = 15*np.arange(15)[None, :] + truth.astype(np.int64)
    ii, jj = np.triu_indices(225, 1)
    capacities = {}
    for group, F in supports.items():
        counts = np.zeros(len(truth), np.int64)
        for i, j in itertools.combinations(range(15), 2):
            counts += F[cells[:, i], cells[:, j]]
        valid = counts == 0
        h = histogram(truth[valid])
        flags = h[keys] > 0
        capacity = np.load(P/f'support{group}/capacity.npz')
        check(f'support{group}packed complete state and key census', np.array_equal(valid, np.unpackbits(capacity['valid_state_bits'], bitorder='little')[:len(valid)]) and np.array_equal(flags, np.unpackbits(capacity['eligible_capacity_bits'], bitorder='little')[:len(flags)]))
        r = raw['support_results'][group]
        num = int(flags.sum())
        false = int(((F[ii, jj] > 0) & (trueF[ii, jj] == 0)).sum())
        check(f'support{group}metrics and rational threshold', r['valid_zero_F_states'] == int(valid.sum()) and r['true_states'] == len(truth) and r['eligible_inputs'] == len(keys) and r['covered_eligible_inputs'] == num and r['capacity_pct'] == 100*float(flags.mean()) and r['valid_state_retention_pct'] == 100*float(valid.mean()) and r['meets_95_percent'] == (20*num >= 19*len(keys)) and r['false_forbidden_edges'] == false and r['missed_true_edges'] == 0 and r['F_contains_true_F'] and r['capacity_sha256'] == sha(P/f'support{group}/capacity.npz'))
        for k in (1, 2, 3):
            part = clues == k
            check(f'support{group}clue{k}census', r['per_clue'][str(k)] == dict(eligible_inputs=int(part.sum()), covered_eligible_inputs=int(flags[part].sum())))
        capacities[group] = (num, false)
    probabilities = []
    for row in raw['rows']:
        b = source['banks'][row['bank']]
        dose = row['dose']
        masses = cache[tuple(b['frequency'])][dose]
        mapping = b['mask_to_support']
        denominator = math.comb(1932, dose)
        favorable = sum(m for mask, m in enumerate(masses) if 20*capacities[mapping[mask]][0] >= 19*len(keys))
        expected = sum(m*capacities[mapping[mask]][0] for mask, m in enumerate(masses))
        exact = sum(m for mask, m in enumerate(masses) if capacities[mapping[mask]][1] == 0)
        fraction = Fraction(favorable, denominator)
        check(f"bank{row['bank']}dose{dose}probabilities", int(row['favorable_draw_subsets']) == favorable and int(row['total_draw_subsets']) == denominator and int(row['success_probability_numerator']) == fraction.numerator and int(row['success_probability_denominator']) == fraction.denominator and row['success_probability_pct'] == 100*float(fraction) and row['expected_capacity_pct'] == 100*float(Fraction(expected, denominator*len(keys))) and row['exact_graph_probability_pct'] == 100*float(Fraction(exact, denominator)) and row['unchanged_support_probability_pct'] == 100*float(Fraction(masses[0], denominator)))
        primary = dose == 64
        check(f"bank{row['bank']}dose{dose}decision", row['primary'] == primary and row['decision'] == (('supported' if 10*favorable >= 9*denominator else 'falsified') if primary else 'descriptive'))
        if primary:
            probabilities.append(fraction)
    joint = math.prod(probabilities)
    check('per-fit conjunction primary', len(probabilities) == 4 and len(raw['rows']) == 20 and raw['primary'] == [r for r in raw['rows'] if r['primary']] and raw['decision'] == ('supported' if all(q >= Fraction(9, 10) for q in probabilities) else 'falsified'))
    check('descriptive independent stream product', Fraction(int(raw['independent_stream_joint']['numerator']), int(raw['independent_stream_joint']['denominator'])) == joint and raw['independent_stream_joint']['probability_pct'] == 100*float(joint))
    if require_summary:
        summary = read(P/'summary.json')
        check('summary mirrors evidence', summary['status'] == 'validated' and summary['rows'] == raw['rows'] and summary['primary'] == raw['primary'] and summary['support_results'] == raw['support_results'] and summary['independent_stream_joint'] == raw['independent_stream_joint'] and summary['protocol_sha256'] == sha(S/'protocol.json'))
        check('audit sources', summary['audit_source_sha256'] == sha(S/'analyze.py') and summary['public_verifier_source_sha256'] == sha(S/'verify_public.py'))
        for rel, digest in summary['public_sha256'].items():
            check('public export '+rel, sha(P/rel) == digest)
        check('figure input identities', all(sha(R/rel) == digest for rel, digest in read(P/'plot-provenance.json')['input_sha256'].items()))
    return dict(status='passed', checks=len(checks), all_source_joint_masks_and_80_deletions_replayed=True, all_integer_masses_match_independent_grouped_union_DP=True, all_32_complete_state_and_integer_key_censuses_rebuilt=True, N13_and_N15_true_universes_independently_reenumerated=True, every_rational_threshold_and_probability_recomputed=True, elapsed_seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    print(json.dumps(verify(parser.parse_args().root)))
