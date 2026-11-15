"""H4aq: exact voting-objective diagnosis of every H4ap bank."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from objective_gap import evaluate

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'results/research/foureyes/bootstrap_consensus'
BASE = ROOT / 'results/research/foureyes/prefix_budget/degree'
PUBLIC = ROOT / 'results/research/foureyes/consensus_gap'
PROTOCOL = HERE / 'consensus-gap.json'
FAMILIES = ['cycle81', 'power2', 'power3', 'cliques9']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def decision(strict):
    assert isinstance(strict, int) and 0 <= strict <= 31
    return 'supported' if strict >= 25 else 'falsified'


def aggregate(rows):
    assert len(rows) == 256 and [r['index'] for r in rows] == list(range(256))
    losses = [r for r in rows if r['baseline_exact'] and not r['consensus_exact']]
    assert len(losses) == 31
    strict = sum(r['gap_twice'] > 0 for r in losses)
    groups = []
    for family in FAMILIES:
        part = [r for r in rows if r['family'] == family]
        lost = [r for r in part if r['baseline_exact'] and not r['consensus_exact']]
        assert len(part) == 64
        groups.append(dict(family=family, losses=len(lost),
                           strict_losses=sum(r['gap_twice'] > 0 for r in lost),
                           tied_losses=sum(r['gap_twice'] == 0 for r in lost),
                           strict_integral_losses=sum(r['gap_twice'] > 0 and r['consensus_integral'] for r in lost),
                           all_strict=sum(r['gap_twice'] > 0 for r in part)))
    return dict(denominator=31, required_strict=25, strict_losses=strict,
                tied_losses=31-strict, decision=decision(strict), groups=groups,
                baseline_exact=sum(r['baseline_exact'] for r in rows),
                consensus_exact=sum(r['consensus_exact'] for r in rows),
                all_strict=sum(r['gap_twice'] > 0 for r in rows))


def freeze():
    assert not PROTOCOL.exists() and not PUBLIC.exists()
    gate_path = HERE / 'consensus-gap-preflight.json'
    gate = read(gate_path)
    assert gate['status'] == 'passed'
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    prior_audit = HERE / 'consensus-gap-parent-audit.json'
    audit = read(prior_audit)
    assert audit['status'] == 'passed' and audit['certified_LPs'] == 8448
    assert audit['mutations_rejected'] == 25344 and audit['unavailable_raw_dependencies'] == []
    parent = read(HERE / 'bootstrap-consensus.json')
    summary = read(PARENT / 'summary.json')
    assert len(summary['rows']) == 256
    assert sum(r['baseline_exact'] and r['classification'] != 'success' for r in summary['rows']) == 31
    assert all(r['classification'] != 'unresolved' for r in summary['rows'])
    inputs = [gate_path, prior_audit, HERE / 'consensus-gap-proposal.json',
              HERE / 'bootstrap-consensus.json', PARENT / 'summary.json', BASE / 'summary.json']
    for job in parent['jobs']:
        inputs += [PARENT / f"b{job['index']:03d}" / name for name in ['models.npz', 'receipt.json', 'audit.json']]
        inputs += [BASE / f"j{job['parent_prefix_index']:03d}" / name for name in ['models.npz', 'receipt.json']]
    inputs += [ROOT / f'results/research/foureyes/{name}/controls/graphs.npz'
               for name in ['cycle_power', 'varying_degree']]
    proposal = read(HERE / 'consensus-gap-proposal.json')
    protocol = dict(proposal, status='frozen_before_true_graph_vote_gap_calculation',
                    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_sha256=gate['source_sha256'],
                    input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    save(PROTOCOL, protocol)
    print('FROZEN', sha(PROTOCOL), flush=True)


def main():
    start = time.perf_counter()
    cpu = time.process_time()
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_start = child_usage.ru_utime + child_usage.ru_stime
    protocol = read(PROTOCOL)
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    def expired(*_):
        raise TimeoutError('H4aq frozen 60 CPU / 90 wall second limit')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.setitimer(signal.ITIMER_REAL, 90)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    used = int(usage.ru_utime + usage.ru_stime)
    resource.setrlimit(resource.RLIMIT_CPU, (used + 60, used + 61))
    rows = []
    try:
        for rel, digest in (protocol['source_sha256'] | protocol['input_sha256']).items():
            assert sha(ROOT / rel) == digest, rel
        jobs = read(HERE / 'bootstrap-consensus.json')['jobs']
        prior = read(PARENT / 'summary.json')['rows']
        graphs = {name: arrays(ROOT / f'results/research/foureyes/{name}/controls/graphs.npz')
                  for name in ['cycle_power', 'varying_degree']}
        for index, job in enumerate(jobs):
            data = arrays(PARENT / f'b{index:03d}/models.npz')
            old = arrays(BASE / f"j{job['parent_prefix_index']:03d}/models.npz")
            degree = job['inferred_degree']
            truth = graphs[job['parent']]['truths'][job['parent_index']]
            votes = np.stack([data[f'r{rep:02d}_primal_twice'] for rep in range(32)]).sum(0)
            assert np.array_equal(votes, data['votes_twice'])
            cost = 64-votes
            assert np.array_equal(cost, data['consensus_cost'])
            proof = [data['r32_' + k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
            result = evaluate(81, degree, cost, *proof, truth)
            base_proof = [old[f'd{degree:02d}_' + k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
            base_result = evaluate(81, degree, old['cost'], *base_proof, truth)
            baseline_exact = bool(np.array_equal(base_proof[0], 2*truth))
            exact = bool(np.array_equal(proof[0], 2*truth))
            integral = bool(np.isin(proof[0], [0, 2]).all())
            assert baseline_exact == job['baseline_exact'] == prior[index]['baseline_exact']
            assert exact == (prior[index]['classification'] == 'success')
            if baseline_exact:
                assert base_result['gap_twice'] == 0
            if exact:
                assert result['gap_twice'] == 0
            rows.append(dict(index=index, family=job['family'], group=job['group'],
                             degree=degree, baseline_exact=baseline_exact,
                             consensus_exact=exact, consensus_integral=integral, **result))
            if (index + 1) % 64 == 0:
                save(PUBLIC / 'progress.json', dict(completed=index+1, planned=256))
                print('GAPS_CHECKED', index+1, flush=True)
        save(PUBLIC / 'rows.json', rows)
        result = dict(status='complete', protocol_sha256=sha(PROTOCOL),
                      rows_sha256=sha(PUBLIC / 'rows.json'), **aggregate(rows),
                      certificates_rechecked=512, new_LPs=0, new_labels=0,
                      new_draws=0, new_GPU_seconds=0,
                      scientific_cpu_seconds=time.process_time()-cpu,
                      scientific_wall_seconds=time.perf_counter()-start)
        assert result['scientific_cpu_seconds'] < 60 and result['scientific_wall_seconds'] < 90
        save(PUBLIC / 'summary.json', result)
        with (PUBLIC / 'independent-audit.json').open('x') as stream:
            subprocess.run([sys.executable, str(HERE / 'audit_consensus_gap.py')],
                           stdout=stream, check=True)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result['scientific_cpu_seconds'] = (time.process_time()-cpu + child_usage.ru_utime
                                            + child_usage.ru_stime-child_cpu_start)
        result['scientific_wall_seconds'] = time.perf_counter()-start
        result['independent_audit_sha256'] = sha(PUBLIC / 'independent-audit.json')
        assert result['scientific_cpu_seconds'] < 60 and result['scientific_wall_seconds'] < 90
        save(PUBLIC / 'summary.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if (PUBLIC / 'summary.json').exists():
            incomplete = read(PUBLIC / 'summary.json')
            incomplete['status'] = 'incomplete'
            save(PUBLIC / 'summary.json', incomplete)
        save(PUBLIC / 'failure.json', dict(error=repr(error), partial_rows=rows,
                                         partial_results_usable=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.command == 'freeze' else main()
