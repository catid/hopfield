"""Recompute every finite likelihood, selected rule and complete target census."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def graph(n, orbits):
    F = np.zeros((n*n, n*n), np.int32)
    orbit_set = {tuple(sorted(o)) for o in orbits}
    for u in range(n*n):
        for v in range(u+1, n*n):
            a, b = abs(u//n-v//n), abs(u%n-v%n)
            F[u, v] = F[v, u] = a == 0 or b == 0 or a == b or tuple(sorted((a, b))) in orbit_set
    return F


def witnesses(columns, orbits):
    lookup = {tuple(o): j for j, o in enumerate(orbits)}
    hits = np.zeros((len(columns), len(orbits)), bool)
    for k, board in enumerate(columns):
        for a, b in itertools.combinations(range(len(board)), 2):
            orbit = tuple(sorted((b-a, abs(int(board[b])-int(board[a])))))
            if orbit in lookup:
                hits[k, lookup[orbit]] = True
    return hits


def histogram(columns):
    cells = 13*np.arange(13)[None, :] + columns.astype(np.int64)
    counts = np.zeros(170**3, np.int64)
    for width in (1, 2, 3):
        subsets = np.array(list(itertools.combinations(range(13), width)))
        keys = ((cells[:, subsets]+1)*np.array([170**(2-j) for j in range(width)])).sum(-1)
        counts += np.bincount(keys.ravel(), minlength=len(counts))
    assert counts.sum() == 377*len(columns)
    return counts


def cross_study(root, source):
    from fractions import Fraction
    from decimal import Decimal, localcontext
    R=Path(root); P=R/'results/research/loki/h3au'; oldP=R/'results/research/loki/h3at'; read=lambda p:json.loads(p.read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    old=read(oldP/'source-selection.json'); results=read(oldP/'capacity-gate.json'); rows=[]
    for bank in source['banks']:
        index=bank['index']; control=P/f"support{bank['consistency_control_group']}/model.npz"; previous=old['banks'][index]; oldcontrol=oldP/f"support{previous['consistency_control_group']}/model.npz"; true=(2,3) if index<4 else (2,5); truecid=next(c['candidate'] for c in source['candidates'] if c['orbits']==[list(true)]); imports=[]
        for member in previous['maximizers']:
            cid=member['candidate']; path=oldP/f"support{member['group']}/model.npz"; odds=Fraction(int(bank['scores'][truecid]['MAP_integer_score_without_common3']),int(bank['scores'][cid]['MAP_integer_score_without_common3']))
            with localcontext() as context:
                context.prec=50; decimal=str(Decimal(odds.numerator)/Decimal(odds.denominator))
            imports.append(dict(candidate=cid,model_path=str(path.relative_to(R)),model_sha256=sha(path),old_ML_to_true_MAP_weight_odds_numerator=str(odds.numerator),old_ML_to_true_MAP_weight_odds_denominator=str(odds.denominator),old_ML_to_true_MAP_weight_odds_decimal=decimal,equals_any_new_MAP_F=any(np.array_equal(np.load(path)['F'],np.load(P/f"support{m['group']}/model.npz")['F']) for m in bank['maximizers'])))
        rows.append(dict(index=index,all_new_MAP_orbits=[source['candidates'][m['candidate']]['orbits'] for m in bank['maximizers']],prior_ML_models=imports,control_F_identical=bool(np.array_equal(np.load(control)['F'],np.load(oldcontrol)['F'])),prior_control_model_path=str(oldcontrol.relative_to(R)),prior_control_model_sha256=sha(oldcontrol)))
    return dict(scope='Clearly labeled imported H3at ML/control measurements on the same already seen banks; new MAP and controls are freshly evaluated. Posterior-odds explanation is descriptive, not prior tuning.',source_selection_path=str((oldP/'source-selection.json').relative_to(R)),source_selection_sha256=sha(oldP/'source-selection.json'),prior_results_path=str((oldP/'capacity-gate.json').relative_to(R)),prior_results_sha256=sha(oldP/'capacity-gate.json'),imported_rows=results['rows'],banks=rows)


def verify(root, require_summary=True):
    start = time.perf_counter()
    R = Path(root); S = R/'research/20261114/loki/h3au'; P = R/'results/research/loki/h3au'
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    p = read(S/'protocol.json'); g = read(P/'capacity-gate.json'); raw = read(P/'study-result.json'); source = read(P/'source-selection.json')
    check('post-hoc same-bank control graph comparisons',read(P/'posthoc-control-comparison.json')==cross_study(R,source))
    check('complete within all frozen budgets', g['status'] == raw['status'] == 'complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds'] <= 120 and g['wall_seconds'] <= 180 and p['budget']['GPUs'] == 0)
    check('sealed source selection precedes target score', p['frozen_utc'] < source['utc'] < raw['utc'] and source['status'] == 'complete_source_only')
    check('protocol and source receipt identities', sha(S/'protocol.json') == source['protocol_sha256'] == g['protocol_sha256'] and raw['source_selection_sha256'] == sha(P/'source-selection.json') and all(g[k] == raw[k] for k in raw if k != 'utc') and source['source_sha256'] == sha(S/'source.py') and raw['source_sha256'] == sha(S/'study.py'))
    for category in ('source_sha256', 'source_data_sha256', 'target_data_sha256'):
        for rel, digest in p[category].items():
            check(category+' '+rel, sha(R/rel) == digest)
    orbits = [tuple(o) for o in p['candidate_prior']['orbits']]
    check('fixed 55 orbit prior', orbits == list(itertools.combinations(range(1, 12), 2)))
    candidates = [chosen for width in (0, 1, 2) for chosen in itertools.combinations(range(55), width)]
    check('all 1541 candidates retained', len(source['candidates']) == len(candidates) == 1541)
    bits = np.load(P/'candidate-validity-bits.npz')
    bitarray=bits['bits']
    check('reused complete source evidence hashes', source['candidate_validity_bits_sha256']==sha(P/'candidate-validity-bits.npz')==sha(R/p['normalizers']['bits_path']) and source['source_ordinary_columns_sha256']==sha(P/'source-ordinary-columns.npy')==sha(R/p['normalizers']['ordinary_path']) and source['inherited_source_selection_sha256']==sha(R/p['normalizers']['selection_path']) and source['all1541_frozen_normalizations_rechecked'] and source['independent_literal_normalizer_verification_reused_from']=='H3at')
    from fractions import Fraction
    check('declared exact hierarchical prior sums to one', sum(Fraction(1,3*math.comb(55,len(chosen))) for chosen in candidates)==1 and [sum(len(chosen)==k for chosen in candidates) for k in (0,1,2)]==[1,55,1485])
    supports = {r['group']: np.load(P/f"support{r['group']}/model.npz")['F'] for r in source['supports']}
    check('all distinct supports within cap', set(supports) == set(range(len(supports))) and len(supports) <= 3084)
    members = {group: [] for group in supports}
    expected_rows = []; expected_banks = []; source_populations = {}
    with tempfile.TemporaryDirectory(prefix='h3au-public-') as tmp:
        tmp = Path(tmp); binary = tmp/'literal'
        subprocess.run(['g++', '-O3', '-std=c++17', '-DNDEBUG', '-Wall', '-Wextra', '-Werror', str(S/'reverse_literal.cpp'), '-o', str(binary)], check=True)
        def enumerate_F(F, n, label):
            inp = tmp/f'{label}.bin'; out = tmp/f'{label}-columns.bin'
            F.astype(np.uint8).tofile(inp)
            result = subprocess.run([str(binary), str(n), str(inp), str(out)], check=True, capture_output=True, text=True)
            columns = np.fromfile(out, np.uint8).reshape(-1, n)
            assert len(columns) == int(result.stdout)
            return columns
        ordinary = enumerate_F(graph(12, []), 12, 'ordinary12')
        check('complete ordinary N12 population', len(ordinary) == 14200 and np.array_equal(ordinary, np.load(P/'source-ordinary-columns.npy')) and bitarray.shape == (1541, 1775))
        incidence = witnesses(ordinary, orbits)
        for cid, chosen in enumerate(candidates):
            item = source['candidates'][cid]; rules = [orbits[j] for j in chosen]
            F = graph(12, rules); columns = enumerate_F(F, 12, f'candidate{cid}')
            filtered = ordinary[~incidence[:, chosen].any(1)] if chosen else ordinary
            check(f'candidate{cid}complete literal and witness populations', np.array_equal(columns, filtered) and np.array_equal(columns, ordinary[np.unpackbits(bitarray[cid], bitorder='little')[:len(ordinary)].astype(bool)]))
            expected = dict(candidate=cid, orbit_indices=list(chosen), orbits=[list(o) for o in rules], complete_source_cardinality=len(columns), source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(), source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest(), orbit_count=len(chosen), prior_class_probability_numerator='1', prior_class_probability_denominator='3', models_in_class=math.comb(55,len(chosen)), prior_per_model_denominator=str(3*math.comb(55,len(chosen))))
            check(f'candidate{cid}all normalization metadata', item == expected)
            for family, cfg in enumerate(p['families']):
                if rules == [tuple(cfg['leaper'])]: source_populations[family] = columns
        inherited = read(R/'research/20261114/loki/h3ao/protocol.json')
        for index, cfg in enumerate(p['banks']):
            b = source['banks'][index]; d = np.load(R/cfg['design_path']); population = source_populations[index//4]
            draw = np.random.default_rng(inherited['sampling']['stream_seeds'][index]).choice(len(population), 64, replace=False)[:16]
            columns = population[draw]; labels = np.zeros((16, 144), bool)
            labels[np.arange(16)[:, None], 12*np.arange(12)[None, :]+columns] = True
            check(f'bank{index}exact original uniform distinct prefix', np.array_equal(d['population_indices'], draw) and np.array_equal(d['labels'], labels) and len(set(map(bytes, columns))) == 16)
            hits = witnesses(columns, orbits); evidence = np.load(P/f'b{index}/source-evidence.npz')
            check(f'bank{index}all sample orbit witnesses', np.array_equal(columns, evidence['sample_columns']) and np.array_equal(hits, evidence['sample_orbit_incidence']) and b['source_evidence_sha256'] == sha(P/f'b{index}/source-evidence.npz'))
            scores = []
            for cid, chosen in enumerate(candidates):
                count = source['candidates'][cid]['complete_source_cardinality']
                violations = int(hits[:, chosen].any(1).sum()) if chosen else 0
                check(f'bank{index}candidate{cid}minimum sample cardinality', violations != 0 or count >= 16)
                denominator=math.comb(count,16) if violations==0 else None; groups=math.comb(55,len(chosen)); score=groups*denominator if violations==0 else None
                scores.append(dict(candidate=cid,sample_violations=violations,consistent=violations==0,complete_source_cardinality=count,orbit_count=len(chosen),models_in_class=groups,likelihood_numerator='1' if violations==0 else '0',likelihood_denominator=str(denominator) if violations==0 else None,posterior_weight_numerator='1' if violations==0 else '0',posterior_weight_denominator=str(3*score) if violations==0 else None,MAP_integer_score_without_common3=str(score) if violations==0 else None))
            check(f'bank{index}all1541 exact posterior weights',scores==b['scores'])
            best=min(int(r['MAP_integer_score_without_common3']) for r in scores if r['consistent'])
            winners=[r['candidate'] for r in scores if r['consistent'] and int(r['MAP_integer_score_without_common3'])==best]
            check(f'bank{index}every tied MAP maximizer retained',winners==[r['candidate'] for r in b['maximizers']] and b['unique_maximum']==(len(winners)==1) and b['best_MAP_integer_score_without_common3']==str(best))
            unwitnessed = np.flatnonzero(~hits.any(0)).tolist()
            check(f'bank{index}complete consistency envelope', b['consistency_orbit_indices'] == unwitnessed and np.array_equal(supports[b['consistency_control_group']], graph(13, [orbits[j] for j in unwitnessed])))
            arms = [dict(candidate=r['candidate'], group=r['group'], arm='MAP_maximizer') for r in b['maximizers']] + [dict(candidate=None, group=b['consistency_control_group'], arm='consistency_envelope')]
            for arm in arms:
                group = arm['group']; rules = [orbits[j] for j in (candidates[arm['candidate']] if arm['candidate'] is not None else unwitnessed)]
                check(f"bank{index}{arm['arm']}{arm['candidate']}full graph", np.array_equal(supports[group], graph(13, rules)))
                members[group].append(dict(index=index, arm=arm['arm'], candidate=arm['candidate']))
                expected_rows.append(dict(index=index, family=index//4, bank=index%4, **arm))
            expected_banks.append(dict(index=index, family=index//4, maximizers=len(winners), unique=len(winners) == 1, maximizer_and_consistency_control_F_identical=all(np.array_equal(supports[r['group']], supports[b['consistency_control_group']]) for r in b['maximizers'])))
        check('exactly eight banks and every logical outcome', len(source['banks']) == 8 and len(g['rows']) == len(expected_rows))
        references = {}
        for family, cfg in enumerate(p['families']):
            Ftrue = graph(13, [tuple(cfg['leaper'])]); truth = enumerate_F(Ftrue, 13, f'truth{family}'); folder = P/f'family{family}'
            check(f'family{family}complete truth and graph', np.array_equal(truth, np.load(R/cfg['target_path'])) and np.array_equal(truth, np.load(folder/'true-columns.npy')) and np.array_equal(Ftrue, np.load(folder/'true-F.npy')))
            counts = histogram(truth); keys = np.flatnonzero(counts); universe = np.load(folder/'input-universe.npz')
            check(f'family{family}full integer clue census', np.array_equal(keys, universe['keys']) and np.array_equal(counts[keys], universe['counts']))
            references[family] = truth, Ftrue, keys, set(map(bytes, truth))
        infos = []
        for item in source['supports']:
            group = item['group']; folder = P/f'support{group}'; model = np.load(folder/'model.npz'); F = supports[group]
            check(f'support{group}uniform model and membership', np.array_equal(F.astype(np.float32), model['J']) and not model['h'].any() and model['q'] == 13 and model['unit'] == 1 and item['members'] == members[group] and item['model_sha256'] == sha(folder/'model.npz') and all(m['index']//4 == item['family'] for m in members[group]))
            truth, Ftrue, keys, truekeys = references[item['family']]; columns = enumerate_F(F, 13, f'support{group}'); valid = np.array([bytes(row) in truekeys for row in columns]); flags = histogram(columns[valid])[keys] > 0; cap = np.load(folder/'capacity.npz'); ai, aj = np.triu_indices(169, 1)
            check(f'support{group}complete valid and invalid states', np.array_equal(columns, cap['columns']) and np.array_equal(valid, cap['valid_state_mask']))
            check(f'support{group}all input capacities', np.array_equal(np.packbits(flags, bitorder='little'), cap['key_capacity_bits']))
            info = dict(group=group, family=item['family'], complete_zero_F_states=len(columns), valid_zero_F_states=int(valid.sum()), invalid_zero_F_states=int((~valid).sum()), true_states=len(truth), input_keys=len(keys), covered_keys=int(flags.sum()), capacity_pct=100*float(flags.mean()), meets_95_percent=20*int(flags.sum()) >= 19*len(keys), exact_graph=bool(np.array_equal(F, Ftrue)), false_forbidden_edges=int(((F[ai, aj] > 0)&(Ftrue[ai, aj] == 0)).sum()), missed_true_edges=int(((F[ai, aj] == 0)&(Ftrue[ai, aj] > 0)).sum()), capacity_sha256=sha(folder/'capacity.npz'))
            check(f'support{group}all reported metrics', info == g['support_results'][group]); infos.append(info)
        for i, row in enumerate(expected_rows):
            info = infos[row['group']]; row.update({k: v for k, v in info.items() if k not in ('group', 'family')}); row['primary'] = row['arm'] == 'MAP_maximizer'; row['decision'] = ('supported' if row['meets_95_percent'] and row['invalid_zero_F_states'] == 0 else 'falsified') if row['primary'] else 'descriptive'
            check(f'logical outcome{i}all fields and decision', row == g['rows'][i])
        for bank in expected_banks:
            rows = [r for r in expected_rows if r['index'] == bank['index'] and r['primary']]; bank['decision'] = 'supported' if all(r['decision'] == 'supported' for r in rows) else 'falsified'; bank['all_maximizers_exact_graph'] = all(r['exact_graph'] for r in rows)
        check('all bank conjunctions and control identities', expected_banks == g['bank_decisions'])
        check('frozen overall conjunction', g['decision'] == ('supported' if all(b['decision'] == 'supported' for b in expected_banks) else 'falsified'))
    if require_summary:
        s = read(P/'summary.json')
        check('summary matches all raw outcomes', s['status'] == 'validated' and all(s[k] == g[k] for k in ('decision', 'rows', 'bank_decisions', 'support_results')) and s['protocol_sha256'] == sha(S/'protocol.json') and s['audit_source_sha256'] == sha(S/'analyze.py') and s['verifier_source_sha256'] == sha(S/'verify_public.py'))
        for rel, digest in s['public_sha256'].items(): check('export '+rel, sha(P/rel) == digest)
        figure = read(P/'plot-provenance.json')
        check('figure provenance inputs and outputs', all(sha(R/rel) == digest for rel, digest in figure['input_sha256'].items()) and all(sha(P/rel) == digest for rel, digest in figure['output_sha256'].items()))
    return dict(status='passed', checks=len(checks), all1541_complete_source_candidate_populations_reenumerated=True, all8_original_uniform_source_draws_replayed=True, all12328_exact_posterior_weights_and_every_tie_verified=True, all_full_target_censuses_and_decisions_verified=True, all_consistency_control_identities_verified=True, elapsed_seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    print(json.dumps(verify(parser.parse_args().root)))
