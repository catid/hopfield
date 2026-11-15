"""Public H4ac graph/vote/threshold replay; raw PL replay remains separate."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / 'results/research/foureyes/pooled_source_study'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strong(j):
    positive = np.sort(j[j > 0])
    n = len(positive)
    assert n > 0
    middle = positive[n // 2] if n % 2 else (positive[n // 2 - 1] + positive[n // 2]) / 2
    return (j > 0) & (j >= middle / 4), float(middle / 4)


def main():
    protocol = HERE / 'pooled_source_study.json'
    p = json.loads(protocol.read_text())
    s = json.loads((PUBLIC / 'summary.json').read_text())
    assert s['status'] == 'validated' and s['protocol_sha256'] == sha(protocol)
    assert s['auditor_sha256'] == sha(HERE / 'audit_pooled_source_study.py')
    available = unavailable = 0
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        if (ROOT / rel).exists():
            assert sha(ROOT / rel) == digest, rel
            available += 1
        else:
            assert rel.startswith('artifacts/'), rel
            unavailable += 1
    assert sha(PUBLIC / 'models.npz') == s['models_sha256']
    with np.load(PUBLIC / 'models.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / 'results/research/foureyes/canonical_confirm/models.npz', allow_pickle=False) as z:
        old = {k: z[k].copy() for k in ['canonical_weak_coefficients', 'adamw_support']}
    assert np.array_equal(arrays['groups'], np.arange(64).reshape(16, 4))
    assert arrays['arms'].tolist() == p['arms']
    assert arrays['supports'].shape == (16, 4, 3240) and arrays['supports'].dtype == bool
    for key in ['pooled_coefficients', 'pooled_gradient']:
        assert arrays[key].shape == (16, 3240) and arrays[key].dtype == np.float64
        assert np.isfinite(arrays[key]).all()
    assert arrays['pooled_counts'].shape == (16, 3240)
    assert arrays['pooled_counts'].dtype == np.int64
    assert ((arrays['pooled_counts'] >= 0) & (arrays['pooled_counts'] <= 2400)).all()
    individual = np.stack([strong(j)[0] for j in old['canonical_weak_coefficients']])
    ii, jj = np.triu_indices(81, 1)
    truth = (ii // 9 == jj // 9) | (ii % 9 == jj % 9) | ((ii // 27 == jj // 27) & (ii % 9 // 3 == jj % 9 // 3))
    assert truth.sum() == 810 and len(s['rows']) == 64
    for group, banks in enumerate(arrays['groups']):
        canonical_vote = individual[banks].sum(axis=0, dtype=np.uint8)
        adamw_vote = old['adamw_support'][banks].sum(axis=0, dtype=np.uint8)
        assert np.array_equal(arrays['canonical_vote_counts'][group], canonical_vote)
        assert np.array_equal(arrays['adamw_vote_counts'][group], adamw_vote)
        pooled, cut = strong(arrays['pooled_coefficients'][group])
        counts = arrays['pooled_counts'][group]
        median = np.median(counts)
        values = np.unique(counts)
        eligible = np.flatnonzero(values[1:] <= median)
        if len(eligible):
            index = eligible[np.argmax(np.diff(values)[eligible])]
            count_cut = int((values[index] + values[index + 1]) // 2)
        else:
            count_cut = 0
        rebuilt = np.stack([adamw_vote >= 3, canonical_vote >= 3, pooled, counts <= count_cut])
        assert np.array_equal(arrays['supports'][group], rebuilt)
        audit = s['pooled_fit_audits'][group]
        assert audit['group'] == group and audit['banks'] == banks.tolist()
        assert audit['pooled_threshold'] == cut and audit['restricted_count_threshold'] == count_cut
        j, gradient = arrays['pooled_coefficients'][group], arrays['pooled_gradient'][group]
        penalty = 9 / 30000
        residual = np.where(j > 0, gradient + penalty, np.where(j < 0, gradient - penalty, np.maximum(abs(gradient) - penalty, 0)))
        assert abs(float(abs(residual).max()) - audit['numpy_kkt_linf']) < 1e-15
        assert abs(audit['numpy_objective'] - audit['numpy_loss'] - penalty * float(abs(j).sum())) < 1e-12
        assert audit['updates'] == 256 and 0 < audit['wall_seconds'] < 30
        for arm, support in zip(p['arms'], rebuilt, strict=True):
            row = next(r for r in s['rows'] if r['group'] == group and r['arm'] == arm)
            missing, extra = int((truth & ~support).sum()), int((~truth & support).sum())
            assert row['banks'] == banks.tolist() and row['missing_true'] == missing and row['extra'] == extra
            assert row['retained'] == int(support.sum()) and row['all_true_retained'] == (missing == 0)
            assert row['exact_graph'] == (missing == extra == 0)
            assert row['support_sha256'] == hashlib.sha256(support.tobytes()).hexdigest()
    for arm, grouped in zip(p['arms'], s['grouped'], strict=True):
        rows = [r for r in s['rows'] if r['arm'] == arm]
        assert len(rows) == 16 and grouped['arm'] == arm
        for aggregate, item in [('exact_graphs', 'exact_graph'), ('all_true_retaining_groups', 'all_true_retained'), ('missing_true_total', 'missing_true'), ('extra_total', 'extra')]:
            assert grouped[aggregate] == sum(r[item] for r in rows)
    selected = s['grouped'][1]
    decision = 'supported' if selected['exact_graphs'] >= 15 and selected['missing_true_total'] == 0 else 'falsified'
    assert s['primary']['decision'] == decision
    assert s['primary']['exact_graphs'] == selected['exact_graphs']
    assert s['primary']['missing_true_total'] == selected['missing_true_total']
    assert s['new_source_rows'] == s['new_solver_inputs'] == s['inference_gpu_seconds'] == 0
    assert s['observation_instances'] == 38400 and s['all_groups_retained']
    for phase in ['gate', 'fit']:
        r = s['launch'][phase]
        assert r['status'] == 'complete' and r['returncodes'] == [0, 0, 0, 0]
        assert r['wall_seconds'] < p['budget'][phase + '_phase_wall_seconds']
    print(json.dumps(dict(status='passed',graphs=64,primary=s['primary'],checked_input_hashes=available,
                         excluded_raw_inputs=unavailable,scope='Public votes, thresholds, graph counts and stored-gradient KKT replay. Independent raw objective, history and source concatenation require the local artifacts.')))


if __name__ == '__main__':
    main()
