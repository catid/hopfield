"""Independent geometry construction and exhaustive oracle-edit verification."""
import hashlib
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from confidence_oracle_common import OUT, PROTOCOL, load_protocol


def main():
    p = load_protocol(require_models=False)
    manifest = json.loads((OUT/'models.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['preparation_sha256'] == sha(HERE/'prepare_confidence_oracle.py')
    assert manifest['policy_sha256'] == sha(HERE/'confidence_oracle_policy.py')
    # Enumerate row/column/box groups independently of the policy's pair formula.
    groups = []
    for r in range(9):
        groups.append([r*9+c for c in range(9)])
    for c in range(9):
        groups.append([r*9+c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            groups.append([(3*br+r)*9+3*bc+c for r in range(3) for c in range(3)])
    edges = set()
    for group in groups:
        for a in group:
            for b in group:
                if a < b:
                    edges.add((a, b))
    assert len(edges) == 810
    pairs = [(a, b) for a in range(81) for b in range(a+1, 81)]
    true = np.array([pair in edges for pair in pairs])
    rows = []
    prior = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    previous = json.loads((prior/'audit.json').read_text())
    for seed in p['seeds']:
        entries = [m for m in manifest['models'] if m['seed'] == seed]
        assert len(entries) == 4 and {m['arm'] for m in entries} == set(p['arms'])
        base_entry = entries[0]
        receipt_path = prior/f's{seed}/receipt.json'
        assert sha(receipt_path) == previous['artifact_sha256'][str(receipt_path.relative_to(ROOT))]
        receipt = json.loads(receipt_path.read_text())
        original = next(m for m in receipt['models'] if m['arm'] == 'evidence')
        assert base_entry['base_path'] == original['path'] and base_entry['base_sha256'] == original['sha256']
        assert sha(ROOT/base_entry['base_path']) == base_entry['base_sha256']
        base = np.load(ROOT/base_entry['base_path'])['weight16']
        assert np.all(base[true] > 0)
        for model in entries:
            assert model['base_path'] == base_entry['base_path'] and model['base_sha256'] == base_entry['base_sha256']
            path = ROOT/model['path']
            assert sha(path) == model['sha256']
            arrays = np.load(path)
            weight = arrays['weight16']
            expected = []
            changed_true = changed_extra = deleted_true = deleted_extra = 0
            for index, is_true in enumerate(true):
                old = int(base[index])
                value = old
                if model['arm'] == 'repair_true' and is_true and old == 1:
                    value = 16
                elif model['arm'] == 'soften_strong_extra' and not is_true and old == 16:
                    value = 1
                elif model['arm'] == 'delete_weak_extra' and not is_true and old == 1:
                    value = 0
                expected.append(value)
                changed_true += int(is_true and old != value)
                changed_extra += int(not is_true and old != value)
                deleted_true += int(is_true and old > 0 and value == 0)
                deleted_extra += int(not is_true and old > 0 and value == 0)
            assert np.array_equal(weight, expected) and np.array_equal(arrays['support'], weight > 0)
            assert hashlib.sha256(weight.tobytes()).hexdigest() == model['weight16_bytes_sha256']
            assert hashlib.sha256((weight > 0).tobytes()).hexdigest() == model['support_bytes_sha256']
            assert deleted_true == 0 and np.all(weight[true] > 0)
            if model['arm'] != 'delete_weak_extra':
                assert np.array_equal(weight > 0, base > 0)
            else:
                assert np.array_equal((base > 0) & (weight == 0), (~true) & (base == 1))
            rows.append({'seed': seed, 'arm': model['arm'], 'changed_true_edges': changed_true,
                         'changed_extra_edges': changed_extra, 'deleted_true_edges': deleted_true,
                         'deleted_extra_edges': deleted_extra, 'support_edges': int((weight > 0).sum()),
                         'true_edges_present': int((weight[true] > 0).sum()),
                         'positive_support_preserved': bool(np.array_equal(weight > 0, base > 0))})
    totals = {arm: {key: sum(row[key] for row in rows if row['arm'] == arm)
                   for key in ('changed_true_edges', 'changed_extra_edges', 'deleted_true_edges', 'deleted_extra_edges')}
              for arm in p['arms']}
    assert totals['repair_true']['changed_true_edges'] == 82
    assert totals['soften_strong_extra']['changed_extra_edges'] == 503
    assert totals['delete_weak_extra']['deleted_extra_edges'] == 19445
    report = {'status': 'validated', 'utc': now(), 'protocol_sha256': sha(PROTOCOL),
              'auditor_sha256': sha(__file__), 'models_manifest_sha256': sha(OUT/'models.json'),
              'model_count': 128, 'weight_entries_verified': 414720, 'independent_true_edges': 810,
              'deleted_true_edges': 0, 'totals': totals, 'rows': rows, 'new_queries': 0,
              'support_semantics': 'Repair and softening preserve positive support. Deletion expands the zero set by removing only extras; every true edge remains positive, so zero-energy validity remains sound.'}
    save(OUT/'model_audit.json', report)
    print(json.dumps({'status': report['status'], 'totals': totals, 'deleted_true_edges': 0}), flush=True)


if __name__ == '__main__':
    main()
