"""H1bx: exact objective gaps for every frozen H1bw selected graph."""
from pathlib import Path
import argparse, datetime, json, os, platform, resource, signal, subprocess, sys, time
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
WALL_ORIGIN = time.monotonic()
import numpy as np
from run import ROOT, HERE, sha, save
from audit_degree_prior import check_certificate
PARENT = ROOT/'results/research/ripper/repeat_query_tradeoff'
PUBLIC = ROOT/'results/research/ripper/repeated_objective_gap'
PROTOCOL = HERE/'protocol-repeated-objective-gap.json'
ARMS = ['single270', 'triple90']


def read(path):
    return json.loads(Path(path).read_text())


def truth_graph():
    ii, jj = np.triu_indices(81, 1)
    return (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))


def evaluate(n, degree, cost, proof, truth):
    ii, jj = np.triu_indices(n, 1)
    assert truth.dtype == bool and truth.shape == cost.shape
    incidence = np.zeros(n, np.int64)
    np.add.at(incidence, ii, truth)
    np.add.at(incidence, jj, truth)
    assert (incidence == degree).all()
    assert check_certificate(n, degree, ii, jj, cost, *proof)
    optimum = sum(int(c)*int(x) for c, x in zip(cost, proof[0], strict=True))
    true_cost = 2*sum(int(c) for c in cost[truth])
    gap = true_cost-optimum
    assert gap >= 0
    exact = bool(np.array_equal(proof[0], 2*truth.astype(np.int64)))
    integral = bool(np.isin(proof[0], [0, 2]).all())
    if exact:
        assert gap == 0
    return dict(optimum_twice=optimum, truth_cost_twice=true_cost, gap_twice=gap,
                exact=exact, integral=integral,
                truth_objective_status='cooptimal' if gap == 0 else 'strictly_suboptimal')


def aggregate(rows):
    assert len(rows) == 128 and [row['index'] for row in rows] == list(range(128))
    assert [row['arm'] for row in rows] == ARMS*64
    failed = [row for row in rows if row['arm'] == 'triple90' and not row['exact']]
    assert len(failed) == 14
    strict = sum(row['gap_twice'] > 0 for row in failed)
    grouped = []
    for arm in ARMS:
        part = [row for row in rows if row['arm'] == arm]
        grouped.append(dict(arm=arm, total=len(part), exact=sum(row['exact'] for row in part),
                            strict=sum(row['gap_twice'] > 0 for row in part),
                            cooptimal_nonexact=sum(row['gap_twice'] == 0 and not row['exact'] for row in part),
                            zero_gap=sum(row['gap_twice'] == 0 for row in part),
                            selected_fractional=sum(not row['integral'] for row in part)))
    assert [row['exact'] for row in grouped] == [64, 50]
    return dict(denominator=14, required_strict=10, strict_failures=strict, cooptimal_failures=14-strict,
                decision='supported' if strict >= 10 else 'falsified', grouped=grouped,
                all128_retained=True, selected_certificates_rechecked=128, new_LPs=0, new_labels=0,
                parent_physical_noisy_union=2073600, official_noisy_label_sequence=14687319)


def verify():
    p = read(PROTOCOL)
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not PUBLIC.exists()
    gate_path = HERE/'repeated-objective-gap-preflight.json'
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['study_gaps_read'] is False
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    parent = read(PARENT/'summary.json')
    assert len(parent['rows']) == 128 and all(row['selected_degree'] == 20 for row in parent['rows'])
    assert [row['success'] for row in parent['grouped']] == [64, 50]
    assert all(row['classification'] != 'unresolved' for row in parent['rows'])
    inputs = [gate_path, HERE/'repeated-objective-gap-proposal.json', HERE/'protocol-repeat-query-tradeoff.json',
              PARENT/'summary.json', ROOT/'results/research/ripper/repeat_query_tradeoff_public_manifest.json']
    for seed in range(64):
        inputs += [PARENT/f's{seed}/{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]
    p = dict(read(HERE/'repeated-objective-gap-proposal.json'), status='frozen_before_true_graph_objective_gap_calculations',
             frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_sha256=gate['source_sha256'],
             input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs})
    save(PROTOCOL, p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), input_files=len(inputs))))


def main():
    assert platform.node() == 'ripper'
    assert not PUBLIC.exists()
    def expired(*_):
        raise TimeoutError('Frozen H1bx60CPU/90wall ceiling')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 90-(time.monotonic()-WALL_ORIGIN)))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    used = int(usage.ru_utime+usage.ru_stime)
    resource.setrlimit(resource.RLIMIT_CPU, (used+60, used+61))
    rows = []
    try:
        p = verify()
        PUBLIC.mkdir(parents=True)
        prior = read(PARENT/'summary.json')['rows']
        truth = truth_graph()
        for index, old in enumerate(prior):
            seed, arm_index = divmod(index, 2)
            arm = ARMS[arm_index]
            with np.load(PARENT/f's{seed}/{arm}-models.npz', allow_pickle=False) as z:
                cost = z['cost']
                proof = [z['d20_'+key] for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
            result = evaluate(81, 20, cost, proof, truth)
            assert result['exact'] == (old['classification'] == 'success')
            rows.append(dict(index=index, seed=seed, arm=arm, degree=20, parent_classification=old['classification'], **result))
            if (index+1) % 16 == 0:
                save(PUBLIC/'progress.json', dict(completed=index+1, planned=128, pid=os.getpid()))
        save(PUBLIC/'rows.json', rows)
        result = dict(status='pending_independent_audit', protocol_sha256=sha(PROTOCOL), rows_sha256=sha(PUBLIC/'rows.json'), **aggregate(rows))
        save(PUBLIC/'summary.json', result)
        with (PUBLIC/'independent-audit.json').open('x') as log:
            subprocess.run([sys.executable, str(HERE/'audit_repeated_objective_gap.py'), '--initial'], stdout=log, check=True)
        usages = [resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)]
        cpu = sum(u.ru_utime+u.ru_stime for u in usages)
        wall = time.monotonic()-WALL_ORIGIN
        assert cpu < 60 and wall < 90
        result.update(status='complete', aggregate_cpu_seconds=cpu, wall_seconds=wall,
                      independent_audit_sha256=sha(PUBLIC/'independent-audit.json'),
                      runtime_scope='From process start, including imports, input hashing and first independent auditor; final reaped child CPU included.')
        save(PUBLIC/'summary.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        save(PUBLIC/'failure.json', dict(error=repr(error), partial_rows=rows, partial_results_usable=False))
        if (PUBLIC/'summary.json').exists():
            incomplete = read(PUBLIC/'summary.json')
            incomplete['status'] = 'incomplete'
            save(PUBLIC/'summary.json', incomplete)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else main()
