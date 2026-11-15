"""Independent integer certificate, vote cost, truth and endpoint replay."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from audit_objective_gap import certify, reconstruct

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.perf_counter()
    protocol_path = HERE / 'consensus-gap.json'
    p = read(protocol_path)
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT / rel) == digest, rel
    public = ROOT / 'results/research/foureyes/consensus_gap'
    parent = ROOT / 'results/research/foureyes/bootstrap_consensus'
    base = ROOT / 'results/research/foureyes/prefix_budget/degree'
    rows = read(public / 'rows.json')
    summary = read(public / 'summary.json')
    old_rows = read(parent / 'summary.json')['rows']
    assert len(rows) == 256 and summary['protocol_sha256'] == sha(protocol_path)
    assert summary['rows_sha256'] == sha(public / 'rows.json')
    families = ['cycle81', 'power2', 'power3', 'cliques9']
    graphs = {}
    for name in ['cycle_power', 'varying_degree']:
        with np.load(ROOT / f'results/research/foureyes/{name}/controls/graphs.npz') as z:
            graphs[name] = {k: z[k] for k in z.files}
    groups = {f: dict(family=f, losses=0, strict_losses=0, tied_losses=0,
                      strict_integral_losses=0, all_strict=0) for f in families}
    losses = strict = mutations = exacts = baselines = all_strict = 0
    for i, row in enumerate(rows):
        family = families[i % 4]
        group = i // 4
        name = 'varying_degree' if family in ['cycle81', 'cliques9'] else 'cycle_power'
        gi = 2*group + int(family in ['cliques9', 'power3'])
        graph = reconstruct(graphs[name]['orders'][gi], family)
        degree = int(graph[0].sum())
        assert (graph.sum(0) == degree).all()
        truth = np.array([graph[a, b] for a in range(81) for b in range(a+1, 81)], bool)
        assert np.array_equal(truth, graphs[name]['truths'][gi])
        old_index = 12*group + 3*(i % 4) + 1
        with np.load(parent / f'b{i:03d}/models.npz') as z:
            voters = [z[f'r{rep:02d}_primal_twice'].tolist() for rep in range(32)]
            votes = [sum(v[e] for v in voters) for e in range(3240)]
            cost = np.array([64-v for v in votes], np.int64)
            assert np.array_equal(cost, z['consensus_cost']) and np.array_equal(votes, z['votes_twice'])
            proof = [z['r32_' + k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
        with np.load(base / f'j{old_index:03d}/models.npz') as z:
            old_cost = z['cost']
            old_proof = [z[f'd{degree:02d}_' + k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
        opt = certify(81, degree, cost, *proof)
        base_opt = certify(81, degree, old_cost, *old_proof)
        for c, witness in [(cost, proof), (old_cost, old_proof)]:
            for k in range(3):
                bad = [v.copy() for v in witness]
                bad[k][0] += 1
                try:
                    certify(81, degree, c, *bad)
                except AssertionError:
                    mutations += 1
                else:
                    raise AssertionError('Altered certificate accepted')
        actual = sum(2*int(cost[e]) for e in range(3240) if truth[e])
        gap = actual-opt
        assert gap >= 0
        exact = bool(np.array_equal(proof[0], truth.astype(np.int64)*2))
        integral = all(int(x) in [0, 2] for x in proof[0])
        baseline = bool(np.array_equal(old_proof[0], truth.astype(np.int64)*2))
        assert exact == (old_rows[i]['classification'] == 'success') and baseline == old_rows[i]['baseline_exact']
        if exact:
            assert gap == 0
        if baseline:
            assert base_opt == sum(2*int(old_cost[e]) for e in range(3240) if truth[e])
        assert row == dict(index=i, family=family, group=group, degree=degree,
                           baseline_exact=baseline, consensus_exact=exact,
                           consensus_integral=integral, optimum_twice=opt,
                           truth_cost_twice=actual, gap_twice=gap,
                           truth_objective_status='cooptimal' if gap == 0 else 'strictly_suboptimal')
        g = groups[family]
        exacts += exact
        baselines += baseline
        all_strict += gap > 0
        g['all_strict'] += int(gap > 0)
        if baseline and not exact:
            losses += 1
            strict += gap > 0
            g['losses'] += 1
            g['strict_losses'] += int(gap > 0)
            g['tied_losses'] += int(gap == 0)
            g['strict_integral_losses'] += int(gap > 0 and integral)
    assert losses == 31 and mutations == 1536
    expected = dict(denominator=31, required_strict=25, strict_losses=strict,
                    tied_losses=31-strict, decision='supported' if strict >= 25 else 'falsified',
                    groups=[groups[f] for f in families], baseline_exact=baselines,
                    consensus_exact=exacts, all_strict=all_strict,
                    certificates_rechecked=512, new_LPs=0, new_labels=0, new_draws=0, new_GPU_seconds=0)
    for key, value in expected.items():
        assert summary[key] == value, key
    assert summary['status'] == 'complete'
    assert summary['scientific_cpu_seconds'] < 60 and summary['scientific_wall_seconds'] < 90
    print(json.dumps(dict(status='passed', **expected, mutations_rejected=mutations,
                          wall_seconds=time.perf_counter()-started)))


if __name__ == '__main__':
    main()
