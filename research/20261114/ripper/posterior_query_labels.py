"""H1bz: H1bw query records scored with exact posterior expected equalities."""
from datetime import datetime, timezone
import argparse, json, os, shutil
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from run import ROOT, HERE, sha, save
from partial_degree_study import zero_witness, npz
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate, top_count_support
from posterior_query_math import score
PARENT = ROOT/'results/research/ripper/repeat_query_tradeoff'
OUT = ROOT/'artifacts/research/20261114/ripper/posterior_query_labels'
PUBLIC = ROOT/'results/research/ripper/posterior_query_labels'
PROTOCOL = HERE/'protocol-posterior-query-labels.json'
ARMS = ['single270', 'triple90']
EXPOSURE = dict(single270=213, triple90=71)


def read(path):
    return json.loads(path.read_text())


def arm_queries(records, arm):
    if arm == 'single270':
        return records['first'][None], records['given']
    assert arm == 'triple90'
    return np.stack([records['first'][:90], records['repeat1'], records['repeat2']]), records['given'][:90]


def verify(raw=False):
    p = read(PROTOCOL)
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel
    if raw:
        for rel, digest in p['raw_input_sha256'].items():
            assert sha(ROOT/rel) == digest, rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    path = HERE/'posterior-query-labels-preflight.json'
    gate = read(path)
    assert gate['status'] == 'passed' and not gate['study_posterior_scores_read']
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    seeds = HERE/'posterior-query-labels-seed-audit.json'
    assert read(seeds)['status'] == 'passed' and sha(seeds) == gate['seed_audit_sha256']
    math_gate = HERE/'posterior-query-math-check.json'
    assert read(math_gate)['status'] == 'passed' and sha(math_gate) == gate['math_gate_sha256']
    parent = read(HERE/'protocol-repeat-query-tradeoff.json')
    prior = read(PARENT/'summary.json')
    assert [row['success'] for row in prior['grouped']] == [64, 50] and len(prior['rows']) == 128
    assert all(row['selected_degree'] == 20 and row['classification'] != 'unresolved' for row in prior['rows'])
    proposal = HERE/'posterior-query-labels-proposal.json'
    inputs = [path, seeds, proposal, math_gate, HERE/'protocol-repeat-query-tradeoff.json', PARENT/'summary.json',
              ROOT/'results/research/ripper/repeat_query_tradeoff_public_manifest.json']
    for seed in range(64):
        inputs += [PARENT/f's{seed}/{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]
        inputs.append(PARENT/f's{seed}/masks.npz')
    raw_parent = ROOT/'artifacts/research/20261114/ripper/repeat_query_tradeoff'
    raw_inputs = [raw_parent/'input_manifest.json', raw_parent/'source-audit.json']
    for seed in range(64):
        raw_inputs += [raw_parent/f'training-s{seed}.npz']
        raw_inputs += [raw_parent/f's{seed}/{name}' for name in ['potential_noise.npz', 'masks.npz', 'single270-observations.npz', 'triple90-observations.npz']]
    inputs += [PARENT/'input_manifest.json', PARENT/'source-audit.json']
    design = read(proposal)
    p = dict(design, frozen_utc=datetime.now(timezone.utc).isoformat(), status='frozen_before_query_record_reconstruction_posterior_scores_and_new_LPs',
             solver=dict(parent['solver'], random_seed_base=design['solver_seed_base']), source_sha256=gate['source_sha256'],
             input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs},
             raw_input_sha256={str(path.relative_to(ROOT)):sha(path) for path in raw_inputs})
    save(PROTOCOL, p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), input_files=len(inputs))), flush=True)


def bank(seed, p):
    directory = OUT/f's{seed}'
    assert directory.exists()
    seal = read(OUT/'query-record-seal.json')
    assert seal['protocol_sha256'] == sha(PROTOCOL) and seal['all64_query_records_validated_before_scoring']
    assert seal['record_sha256'][str(seed)] == sha(directory/'query-records.npz')
    records_input = npz(directory/'query-records.npz')
    ii, jj = np.triu_indices(81, 1)
    for arm_index, arm in enumerate(ARMS):
        index = 2*seed+arm_index
        source = PARENT/f's{seed}/{arm}-models.npz'
        assert sha(source) == p['input_sha256'][str(source.relative_to(ROOT))]
        values, given = arm_queries(records_input, arm)
        arrays = score(values, given)
        cost = arrays['cost']
        assert (arrays['joint_count'] == EXPOSURE[arm]).all()
        arrays['top810_support'] = top_count_support(np.ones(3240, bool), cost)
        arrays.update({'d00_'+key:value for key, value in zero_witness(81, cost).items()})
        records = []
        for degree in range(1, 41):
            log = directory/f'{arm}-d{degree:02d}.log'
            x, y, solver = solve(81, degree, ii, jj, cost, p['solver']['random_seed_base']-2026111502+index*41+degree, log)
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
                certificate, witness = attempt_certificate(81, degree, ii, jj, cost, x, y)
            else:
                certificate = dict(certified=False, reason='solver_status_or_validity')
                witness = {key:np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
            arrays.update({f'd{degree:02d}_'+key:value for key, value in dict(primal_values=x, node_dual_values=y, **witness).items()})
            records.append(dict(degree=degree, solver=solver, certificate=certificate, log_sha256=sha(log)))
        np.savez_compressed(directory/f'{arm}-models.npz', **arrays)
        save(directory/f'{arm}-receipt.json', dict(index=index, seed=seed, arm=arm, protocol_sha256=sha(PROTOCOL),
             records=records, models_sha256=sha(directory/f'{arm}-models.npz'), parent_models_sha256=sha(source),
             query_records_sha256=sha(directory/'query-records.npz'), query_seal_sha256=sha(OUT/'query-record-seal.json')))
    from audit_posterior_query_labels import audit_bank
    checked = audit_bank(seed, False, p)
    save(directory/'audit.json', checked)
    print(json.dumps(dict(seed=seed, complete=True)), flush=True)


def worker(worker_id):
    assert 0 <= worker_id < 8
    p = verify()
    for seed in range(worker_id, 64, 8):
        bank(seed, p)


def summary():
    from audit_posterior_query_labels import aggregate
    p = verify()
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    rows, audits, files = [], [], {}
    for seed in range(64):
        directory = OUT/f's{seed}'
        checked = read(directory/'audit.json')
        assert checked['status'] == 'validated' and checked['protocol_sha256'] == sha(PROTOCOL)
        assert checked['auditor_sha256'] == sha(HERE/'audit_posterior_query_labels.py')
        for name, digest in checked['artifact_sha256'].items():
            assert sha(directory/name) == digest
        audits.append(checked)
        rows += checked['rows']
        target = PUBLIC/f's{seed}'
        target.mkdir()
        for name in ['audit.json', 'query-records.npz', 'query-record-audit.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]:
            shutil.copy2(directory/name, target/name)
            files[f's{seed}/{name}'] = sha(target/name)
    for name in ['query-record-seal.json']:
        shutil.copy2(OUT/name, PUBLIC/name)
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), scope=p['scope'], rows=rows, **aggregate(rows),
                  new_certified_LPs=sum(row['certified_grid_LPs'] for row in rows),
                  archived_selected_certificates=128, mutations_rejected=sum(row['mutations_rejected']+3 for row in rows),
                  analytic_zero_proofs=128, query_accounting=p['query_accounting'], artifact_sha256=files, all128_arms_retained=True)
    save(OUT/'audit.json', result)
    save(PUBLIC/'summary.json', result)
    print(json.dumps(dict(endpoints=result['endpoints'], grouped=result['grouped'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'worker', 'summary'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        from prepare_posterior_queries import prepare
        prepare()
    else:
        worker(args.seed) if args.action == 'worker' else globals()[args.action]()
