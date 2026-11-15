"""Independent Fraction/Decimal audit before any fresh-cohort inference."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from confidence_strict_common import OUT, PROTOCOL, load_protocol
from confidence_strict_policy import strong_ratio, strict_strong


def main():
    p = load_protocol(require_models=False)
    manifest = json.loads((OUT/'models.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['preparation_sha256'] == sha(HERE/'prepare_confidence_strict.py')
    assert manifest['policy_sha256'] == sha(HERE/'confidence_strict_policy.py')
    # Exact threshold boundary and adjacent integers, independent of observed fits.
    for denominator in (16, 160, 16000, 16*10**40):
        numerator = denominator//16
        assert strong_ratio(numerator, denominator)
        assert not strong_ratio(numerator+1, denominator)
        if numerator > 1:
            assert strong_ratio(numerator-1, denominator)
    for eta in (Fraction(1,20), Fraction(64095,10**6), Fraction(8,9)):
        assert not strict_strong(eta,0,0)  # LR=1 exceeds the strict 1/16 threshold.
    prior = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    previous = json.loads((prior/'audit.json').read_text())
    prior_protocol = json.loads((HERE/'protocol-confidence-budget-source.json').read_text())
    i, j = np.triu_indices(81,1)
    true = (i//9 == j//9) | (i%9 == j%9) | ((i//27 == j//27) & (i%9//3 == j%9//3))
    rows = []
    fraction_checks = decimal_checks = 0
    for seed in p['seeds']:
        d = prior/f's{seed}'
        receipt_path = d/'receipt.json'
        assert sha(receipt_path) == previous['artifact_sha256'][str(receipt_path.relative_to(ROOT))]
        receipt = json.loads(receipt_path.read_text())
        for name in ('fit.npz','calibration.json','shuffle_permutation.npy'):
            assert sha(d/name) == receipt['artifact_sha256'][name]
        fit = np.load(d/'fit.npz')
        support, joint, equal = fit['support'], fit['joint_count'], fit['equal_count']
        calibration = json.loads((d/'calibration.json').read_text())['fit']
        eta = Fraction(calibration['eta_upper_numerator'],calibration['eta_upper_denominator'])
        null = eta/4-Fraction(9,64)*eta*eta
        alternative = Fraction(1,9)
        a,b = (Fraction(1),Fraction(1)) if null >= alternative else (alternative/null,(1-alternative)/(1-null))
        old_expected = np.zeros(3240,np.uint8)
        strict_expected = np.zeros(3240,np.uint8)
        with localcontext() as context:
            context.prec = 80
            log_a = (Decimal(a.numerator)/a.denominator).ln()
            log_b = (Decimal(b.numerator)/b.denominator).ln()
            cut = -Decimal(16).ln()
            for index in np.flatnonzero(support):
                count, count_equal = int(joint[index]), int(equal[index])
                ratio = a**count_equal*b**(count-count_equal)
                old_expected[index] = 16 if ratio <= 1 else 1
                strict_expected[index] = 16 if ratio <= Fraction(1,16) else 1
                fraction_checks += 1
                log_ratio = count_equal*log_a+(count-count_equal)*log_b
                assert (ratio <= Fraction(1,16)) == (log_ratio <= cut)
                decimal_checks += 1
        permutation = np.random.default_rng(prior_protocol['streams']['weight_shuffle']+seed).permutation(3240)
        assert np.array_equal(np.load(d/'shuffle_permutation.npy'),permutation)
        shuffled = np.where(support,16,0).astype(np.uint8)
        retained = [int(index) for index in permutation if support[index]]
        shuffled[retained[:int((strict_expected == 1).sum())]] = 1
        expected = {'old_rule': old_expected, 'strict_rule': strict_expected, 'shuffled': shuffled}
        old_entry = next(m for m in receipt['models'] if m['arm'] == 'evidence')
        assert sha(ROOT/old_entry['path']) == old_entry['sha256']
        assert np.array_equal(np.load(ROOT/old_entry['path'])['weight16'], old_expected)
        for entry in [m for m in manifest['models'] if m['seed'] == seed]:
            assert entry['prior_receipt_sha256'] == sha(receipt_path)
            path = ROOT/entry['path']
            assert sha(path) == entry['sha256']
            z = np.load(path)
            weight = z['weight16']
            assert np.array_equal(weight,expected[entry['arm']])
            assert np.array_equal(z['support'],support) and np.array_equal(weight > 0,support)
            assert hashlib.sha256(weight.tobytes()).hexdigest() == entry['weight16_bytes_sha256']
            assert hashlib.sha256(support.tobytes()).hexdigest() == entry['support_bytes_sha256']
            rows.append({'seed': seed, 'arm': entry['arm'], 'missing_true_edges': int((true & ~support).sum()),
                         'extra_edges': int((~true & support).sum()), 'weak_true_edges': int((true & (weight == 1)).sum()),
                         'weak_extra_edges': int((~true & (weight == 1)).sum()), 'strong_extra_edges': int((~true & (weight == 16)).sum()),
                         'weak_weights': int((weight == 1).sum()), 'support_preserved': True})
        assert np.array_equal(np.sort(shuffled),np.sort(strict_expected))
    assert len(rows) == 96
    report = {'status': 'validated', 'utc': now(), 'protocol_sha256': sha(PROTOCOL),
              'auditor_sha256': sha(__file__), 'models_manifest_sha256': sha(OUT/'models.json'),
              'source_only': True, 'fresh_inputs_accessed': 0, 'model_count': 96,
              'independent_fraction_classifications': fraction_checks,
              'independent_decimal80_classifications': decimal_checks,
              'exact_boundary_and_LR1_checks': True, 'original_rule_independently_reproduced': True,
              'pre_observation_permutation_replayed': True, 'all32_banks_retained': True,
              'weight_entries_verified': 311040, 'rows': rows,
              'geometry_scope': 'Postfit descriptive audit only. No geometry enters assignment or bank selection.',
              'totals': {arm: {key:sum(row[key] for row in rows if row['arm'] == arm)
                         for key in ('missing_true_edges','extra_edges','weak_true_edges','weak_extra_edges','strong_extra_edges')}
                         for arm in p['arms']}}
    save(OUT/'model_audit.json',report)
    save(ROOT/'results/research/ripper/confidence_strict_source.json',report)
    print(json.dumps({'status':report['status'],'fraction_checks':fraction_checks,'decimal_checks':decimal_checks,'totals':report['totals']}),flush=True)


if __name__ == '__main__':
    main()
