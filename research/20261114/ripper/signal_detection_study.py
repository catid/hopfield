"""H1bt: prospective conditional-randomization signal detection; no graph confidence claim."""
from datetime import datetime, timezone
import argparse, hashlib, json, os, shutil
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from run import ROOT, HERE, sha, save
from partial_degree_study import npz, zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from laplace_rate_math import rate_costs
from marginal_shuffle_study import observed_histograms, shuffle_observed, observe, concentration

OUT = ROOT/'artifacts/research/20261114/ripper/signal_detection'
PUBLIC = ROOT/'results/research/ripper/signal_detection'
PARENT = ROOT/'results/research/ripper/higher_noise'
RAW_PARENT = ROOT/'artifacts/research/20261114/ripper/higher_noise'
PROTOCOL = HERE/'protocol-signal-detection.json'
ARMS = ['uniform', 'balanced']
B = 31


def grid_index(case, arm_index, draw):
    assert 0 <= case < 64 and 0 <= arm_index < 2 and 0 <= draw <= B
    return (2*case+arm_index)*32+draw


def shuffle_seed(p, case, arm_index, draw):
    assert 1 <= draw <= B
    return p['shuffle_seed_base']+62*case+31*arm_index+draw-1


def sample_null(seed):
    return np.random.default_rng(seed).integers(0, 9, size=(180, 81), dtype=np.int8)


def randomization_test(observed, shuffles):
    assert len(shuffles) == B
    if observed is None or any(value is None for value in shuffles):
        return dict(numerator=None, denominator=32, greater_or_equal=None, signal=None)
    exceed = sum(value['numerator']*observed['denominator'] >= observed['numerator']*value['denominator'] for value in shuffles)
    return dict(numerator=1+exceed, denominator=32, greater_or_equal=exceed, signal=exceed == 0)


def verify(raw=False):
    p = json.loads(PROTOCOL.read_text())
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel
    if raw:
        for rel, digest in p['raw_input_sha256'].items():
            assert sha(ROOT/rel) == digest, rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate = HERE/'signal-detection-preflight.json'
    preflight = json.loads(gate.read_text())
    assert preflight['status'] == 'passed'
    seed_receipt = HERE/'signal-detection-seed-audit.json'
    assert sha(seed_receipt) == preflight['seed_audit_sha256']
    assert json.loads(seed_receipt.read_text())['status'] == 'passed'
    for rel, digest in preflight['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    proposal = HERE/'signal-detection-proposal.json'
    design = json.loads(proposal.read_text())
    parent = json.loads((HERE/'protocol-higher-noise.json').read_text())
    inputs = [gate, proposal, HERE/'signal-detection-seed-audit.json', HERE/'protocol-higher-noise.json', PARENT/'summary.json']
    raw = {}
    for bank in range(32):
        inputs.append(PARENT/f's{bank}/masks.npz')
        for arm in ARMS:
            inputs += [PARENT/f's{bank}/{arm}-{suffix}' for suffix in ['models.npz', 'receipt.json']]
            path = RAW_PARENT/f's{bank}/{arm}-observations.npz'
            receipt = json.loads((PARENT/f's{bank}/{arm}-receipt.json').read_text())
            assert sha(path) == receipt['observations_sha256']
            raw[str(path.relative_to(ROOT))] = sha(path)
    p = dict(design, frozen_utc=datetime.now(timezone.utc).isoformat(),
             status='frozen_before_new_IID_null_boards_and_all_3968_new_shuffle_draws',
             source_sha256=preflight['source_sha256'], input_sha256={str(f.relative_to(ROOT)): sha(f) for f in inputs}, raw_input_sha256=raw,
             degrees=parent['degrees'], candidate_degrees=parent['candidate_degrees'], costs=parent['costs'],
             selector=parent['selector'], objective=parent['objective'], proof=parent['proof'], prior=parent['prior'],
             parent_solver=parent['solver'], solver=dict(parent['solver'], random_seed_base=design['solver_seed_base']),
             parent_protocol_sha256=sha(HERE/'protocol-higher-noise.json'))
    save(PROTOCOL, p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), new_LPs=161280, archived_LPs_rechecked=2560)))


def fit_grid(p, case, ai, draw, mask, labels, directory):
    directory.mkdir(parents=True, exist_ok=False)
    index = grid_index(case, ai, draw)
    observed = observe(mask, labels)
    cost = rate_costs(observed['joint_count'], observed['equal_count'])
    arrays = dict(cost=cost, joint_count=observed['joint_count'], equal_count=observed['equal_count'])
    arrays.update({'d00_'+key: value for key, value in zero_witness(81, cost).items()})
    ii, jj = np.triu_indices(81, 1)
    records = []
    for degree in range(1, 41):
        log = directory/f'd{degree:02d}.log'
        x, y, solver = solve(81, degree, ii, jj, cost, p['solver_seed_base']-2026111502+index*41+degree, log)
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
            certificate, witness = attempt_certificate(81, degree, ii, jj, cost, x, y)
        else:
            certificate = dict(certified=False, reason='solver_status_or_validity')
            witness = {key: np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
        arrays.update({f'd{degree:02d}_'+key: value for key, value in dict(primal_values=x, node_dual_values=y, **witness).items()})
        records.append(dict(degree=degree, solver=solver, certificate=certificate, log_sha256=sha(log)))
    np.savez_compressed(directory/'models.npz', **arrays)
    save(directory/'receipt.json', dict(case=case, bank=case % 32, cohort='real' if case < 32 else 'null', arm=ARMS[ai], draw=draw, index=index,
         protocol_sha256=sha(PROTOCOL), records=records, models_sha256=sha(directory/'models.npz'),
         labels_bytes_sha256=hashlib.sha256(observed['values'].tobytes()).hexdigest()))


def case_job(case):
    p = verify(True)
    bank = case % 32
    directory = OUT/f'c{case:02d}'
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PARENT/f's{bank}/masks.npz', directory/'masks.npz')
    masks = npz(directory/'masks.npz')
    base = {}
    if case >= 32:
        base['null_board'] = sample_null(p['null_seed_base']+bank)
    histograms = {}
    for ai, arm in enumerate(ARMS):
        if case < 32:
            original = npz(RAW_PARENT/f's{bank}/{arm}-observations.npz')
            assert np.array_equal(original['given'], masks[arm])
            labels = original['values']
        else:
            labels = np.where(masks[arm], base['null_board'], -1).astype(np.int8)
        histograms[arm] = observed_histograms(masks[arm], labels)
        base[arm+'_histogram'] = histograms[arm]
    np.savez_compressed(directory/'base.npz', **base)
    complete = 0
    for ai, arm in enumerate(ARMS):
        for draw in range(0 if case >= 32 else 1, 32):
            labels = np.where(masks[arm], base['null_board'], -1).astype(np.int8) if draw == 0 else shuffle_observed(masks[arm], histograms[arm], shuffle_seed(p, case, ai, draw))
            fit_grid(p, case, ai, draw, masks[arm], labels, directory/arm/f'b{draw:02d}')
            complete += 1
            save(directory/'progress.json', dict(case=case, complete_grids=complete, complete_LPs=40*complete, status='fitting'))
    from audit_signal_detection import audit_case
    audited = audit_case(case)
    save(directory/'audit.json', audited)
    save(directory/'progress.json', dict(case=case, complete_grids=complete, complete_LPs=40*complete, status='audited'))
    print(json.dumps(dict(case=case, status='audited', complete_LPs=40*complete)), flush=True)


def summary():
    from audit_signal_detection import aggregate
    p = verify()
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    audits = []
    files = {}
    for case in range(64):
        directory = OUT/f'c{case:02d}'
        audited = json.loads((directory/'audit.json').read_text())
        assert audited['status'] == 'validated' and audited['protocol_sha256'] == sha(PROTOCOL) and audited['auditor_sha256'] == sha(HERE/'audit_signal_detection.py')
        for name, digest in audited['artifact_sha256'].items():
            assert sha(directory/name) == digest
            destination = PUBLIC/f'c{case:02d}'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory/name, destination)
            files[f'c{case:02d}/{name}'] = digest
        audits.append(audited)
    rows = [row for audited in audits for row in audited['rows']]
    outcomes = aggregate(rows)
    all_grids = [g for row in rows for g in [row['observed_grid']]+row['shuffle_grids']]
    literal_accounting = {key: sum(a['query_accounting'][key] for a in audits) for key in audits[0]['query_accounting']}
    assert literal_accounting == dict(shuffled_assignments=51425280, new_IID_generated_cells=466560, new_IID_observed_union=460721, historical_real_union_reused=460721)
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), scope=p['scope'], rows=rows, **outcomes, raw_audits=audits,
                  query_accounting=p['query_accounting'], literal_synthetic_accounting=literal_accounting, certified_LPs=sum(g['certified_grid_LPs'] for g in all_grids),
                  mutations_rejected=sum(g['mutations_rejected'] for g in all_grids), analytic_zero_witnesses_checked=len(all_grids),
                  artifact_sha256=files, all_cases_retained=True)
    assert len(all_grids) == 4096
    save(OUT/'audit.json', result)
    save(PUBLIC/'summary.json', result)
    print(json.dumps(dict(endpoints=result['endpoints'], grouped=result['grouped'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'case', 'summary'])
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    case_job(args.case) if args.action == 'case' else globals()[args.action]()
