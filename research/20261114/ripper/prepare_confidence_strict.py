"""Reweight all fixed H1ai fits before accessing the fresh held-out cohort."""
from fractions import Fraction
import hashlib
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from confidence_strict_common import OUT, PROTOCOL, load_protocol
from confidence_strict_policy import weights


def main():
    p = load_protocol(require_models=False)
    assert not OUT.exists()
    OUT.mkdir()
    prior = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    previous = json.loads((prior/'audit.json').read_text())
    models = []
    for seed in p['seeds']:
        d = prior/f's{seed}'
        receipt_path = d/'receipt.json'
        assert sha(receipt_path) == previous['artifact_sha256'][str(receipt_path.relative_to(ROOT))]
        receipt = json.loads(receipt_path.read_text())
        for name in ('fit.npz', 'calibration.json', 'shuffle_permutation.npy'):
            assert sha(d/name) == receipt['artifact_sha256'][name]
        source = next(m for m in receipt['models'] if m['arm'] == 'evidence')
        assert sha(ROOT/source['path']) == source['sha256']
        old = np.load(ROOT/source['path'])['weight16']
        fit = np.load(d/'fit.npz')
        support, joint, equal = fit['support'], fit['joint_count'], fit['equal_count']
        calibration = json.loads((d/'calibration.json').read_text())['fit']
        eta = Fraction(calibration['eta_upper_numerator'], calibration['eta_upper_denominator'])
        permutation = np.load(d/'shuffle_permutation.npy')
        arrays = (support, joint, equal, permutation, old)
        for array in arrays:
            array.setflags(write=False)
        before = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
        assigned = weights(support, joint, equal, eta, permutation, old)
        assert before == [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
        dest = OUT/f's{seed}'
        dest.mkdir()
        for arm, weight in assigned.items():
            path = dest/f'{arm}-n1110.npz'
            np.savez_compressed(path, support=support, weight16=weight)
            models.append({'arm': arm, 'seed': seed, 'n': 1110, 'path': str(path.relative_to(ROOT)),
                           'sha256': sha(path), 'prior_receipt_sha256': sha(receipt_path),
                           'weight16_bytes_sha256': hashlib.sha256(weight.tobytes()).hexdigest(),
                           'support_bytes_sha256': hashlib.sha256(support.tobytes()).hexdigest()})
    save(OUT/'models.json', {'status': 'prepared_before_fresh_outcomes', 'utc': now(),
         'protocol_sha256': sha(PROTOCOL), 'preparation_sha256': sha(__file__),
         'policy_sha256': sha(HERE/'confidence_strict_policy.py'), 'new_queries': 0,
         'source_banks_retained': 32, 'models': models})
    print(json.dumps({'prepared_models': len(models), 'new_queries': 0}), flush=True)


if __name__ == '__main__':
    main()
