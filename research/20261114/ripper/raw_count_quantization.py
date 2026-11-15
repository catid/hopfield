"""H1by: all H1bw grids refit with unrounded raw equality-count costs."""
from datetime import datetime, timezone
import argparse, json, os, shutil
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from run import ROOT, HERE, sha, save
from partial_degree_study import zero_witness, npz
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate, top_count_support
PARENT = ROOT/'results/research/ripper/repeat_query_tradeoff'
OUT = ROOT/'artifacts/research/20261114/ripper/raw_count_quantization'
PUBLIC = ROOT/'results/research/ripper/raw_count_quantization'
PROTOCOL = HERE/'protocol-raw-count-quantization.json'
ARMS = ['single270', 'triple90']
EXPOSURE = dict(single270=213, triple90=71)


def read(path):
    return json.loads(path.read_text())


def raw_cost(joint, equal, expected):
    assert joint.shape == equal.shape == (3240,)
    assert joint.dtype == equal.dtype == np.int64
    assert (joint == expected).all() and ((equal >= 0) & (equal <= joint)).all()
    return equal.copy()


def verify():
    p = read(PROTOCOL)
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    path = HERE/'raw-count-quantization-preflight.json'
    gate = read(path)
    assert gate['status'] == 'passed' and not gate['study_rawcount_scores_read']
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    seeds = HERE/'raw-count-quantization-seed-audit.json'
    assert read(seeds)['status'] == 'passed' and sha(seeds) == gate['seed_audit_sha256']
    parent = read(HERE/'protocol-repeat-query-tradeoff.json')
    prior = read(PARENT/'summary.json')
    assert [row['success'] for row in prior['grouped']] == [64, 50] and len(prior['rows']) == 128
    assert all(row['selected_degree'] == 20 and row['classification'] != 'unresolved' for row in prior['rows'])
    proposal = HERE/'raw-count-quantization-proposal.json'
    inputs = [path, seeds, proposal, HERE/'protocol-repeat-query-tradeoff.json', PARENT/'summary.json',
              ROOT/'results/research/ripper/repeat_query_tradeoff_public_manifest.json']
    for seed in range(64):
        inputs += [PARENT/f's{seed}/{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]
    design = read(proposal)
    p = dict(design, frozen_utc=datetime.now(timezone.utc).isoformat(), status='frozen_before_study_rawcount_objectives_and_new_LPs',
             solver=dict(parent['solver'], random_seed_base=design['solver_seed_base']), source_sha256=gate['source_sha256'],
             input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs})
    save(PROTOCOL, p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), input_files=len(inputs))), flush=True)


def bank(seed, p):
    directory = OUT/f's{seed}'
    directory.mkdir()
    ii, jj = np.triu_indices(81, 1)
    for arm_index, arm in enumerate(ARMS):
        index = 2*seed+arm_index
        source = PARENT/f's{seed}/{arm}-models.npz'
        assert sha(source) == p['input_sha256'][str(source.relative_to(ROOT))]
        with np.load(source, allow_pickle=False) as z:
            joint, equal = z['joint_count'], z['equal_count']
        cost = raw_cost(joint, equal, EXPOSURE[arm])
        arrays = dict(cost=cost, joint_count=joint, equal_count=equal,
                      top810_support=top_count_support(np.ones(3240, bool), cost))
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
             records=records, models_sha256=sha(directory/f'{arm}-models.npz'), parent_models_sha256=sha(source)))
    from audit_raw_count_quantization import audit_bank
    checked = audit_bank(seed, False, p)
    save(directory/'audit.json', checked)
    print(json.dumps(dict(seed=seed, complete=True)), flush=True)


def worker(worker_id):
    assert 0 <= worker_id < 8
    p = verify()
    for seed in range(worker_id, 64, 8):
        bank(seed, p)


def summary():
    from audit_raw_count_quantization import aggregate
    p = verify()
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    rows, audits, files = [], [], {}
    for seed in range(64):
        directory = OUT/f's{seed}'
        checked = read(directory/'audit.json')
        assert checked['status'] == 'validated' and checked['protocol_sha256'] == sha(PROTOCOL)
        assert checked['auditor_sha256'] == sha(HERE/'audit_raw_count_quantization.py')
        for name, digest in checked['artifact_sha256'].items():
            assert sha(directory/name) == digest
        audits.append(checked)
        rows += checked['rows']
        target = PUBLIC/f's{seed}'
        target.mkdir()
        for name in ['audit.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]:
            shutil.copy2(directory/name, target/name)
            files[f's{seed}/{name}'] = sha(target/name)
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), scope=p['scope'], rows=rows, **aggregate(rows),
                  new_certified_LPs=sum(row['certified_grid_LPs'] for row in rows),
                  archived_selected_certificates=128, mutations_rejected=sum(row['mutations_rejected']+3 for row in rows),
                  analytic_zero_proofs=128, query_accounting=p['query_accounting'], artifact_sha256=files, all128_arms_retained=True)
    save(OUT/'audit.json', result)
    save(PUBLIC/'summary.json', result)
    print(json.dumps(dict(endpoints=result['endpoints'], grouped=result['grouped'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'worker', 'summary'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    worker(args.seed) if args.action == 'worker' else globals()[args.action]()
