"""Independent SciPy H2bo endpoints before fresh generation."""
import json
from scipy.stats import beta
import study as S
import audit as A
from bp_first_fresh_statistics import absolute, noninferiority


def main():
    n = 4096
    rows, absolutes, paired = [], [], []
    maximum = 0.
    def reference(k, alpha):
        return (float(beta.ppf(alpha, k, n-k+1)) if k else 0.,
                float(beta.ppf(1-alpha, k+1, n-k)) if k < n else 1.)
    for alpha in [.025, .05]:
        for k in [0, 1, 4, 8, 9, 20, 21, 40, 52, 64, 128, 2048, 4095, 4096]:
            actual, expected = A.bounds(k, n, alpha), reference(k, alpha)
            error = max(abs(x-y) for x, y in zip(actual, expected))
            assert error < 1e-10
            maximum = max(maximum, error)
            rows.append({'k': k, 'alpha': alpha, 'error': error})
    for failures in [0, 1, 4, 8, 9, 52]:
        actual = absolute(n-failures)
        lo, hi = reference(n-failures, .05)
        assert actual['decision'] == ('supported' if lo >= .999 else 'falsified' if hi < .999 else 'inconclusive')
        absolutes.append(actual)
    for gains, losses in [(0, 0), (0, 1), (0, 4), (0, 52), (4, 0), (21, 0), (40, 0), (52, 0), (64, 1), (128, 52), (2048, 2048)]:
        actual = noninferiority(gains, losses)
        gl, gu = reference(gains, .025)
        ll, lu = reference(losses, .025)
        lower, upper = 100*(gl-lu), 100*(gu-ll)
        assert abs(lower-actual['separately_one_sided95_net_lower_pp']) < 1e-8
        assert abs(upper-actual['separately_one_sided95_net_upper_pp']) < 1e-8
        assert actual['decision'] == ('supported' if lower >= -.1 else 'falsified' if upper < -.1 else 'inconclusive')
        paired.append(actual)
    invalid = [(absolute, (4096, 4095)), (absolute, (-1,)),
               (noninferiority, (4096, 1)), (noninferiority, (0, 0, 4095))]
    for function, args in invalid:
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid denominator/outcome accepted')
    target = S.OUT / 'bp_first_fresh_confirmation'
    target.mkdir(exist_ok=True)
    result = {'status': 'passed', 'bound_cases': rows, 'absolute_cases': absolutes,
              'paired_cases': paired, 'invalid_cohort_cases': len(invalid),
              'maximum_probability_error': maximum, 'fresh_outcomes_read': False,
              'source_sha256': {name: S.sha(S.HERE/name) for name in [
                  'bp_first_fresh_statistics.py', 'preflight_bp_first_fresh_statistics.py',
                  'fallback_confirmation_statistics.py', 'audit.py']}}
    S.save(target / 'statistics-gate.json', result)
    print(json.dumps({'status': 'passed', 'maximum_probability_error': maximum,
                      'invalid_cohort_cases': len(invalid), 'fresh_outcomes_read': False}, indent=2))


if __name__ == '__main__':
    main()
