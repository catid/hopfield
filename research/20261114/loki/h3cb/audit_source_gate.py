"""Independent source-only replay; prospective RNG APIs are forbidden here."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from common import ROOT, SRC, read, save, sha
from audit_sources import verify_prior


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prior = read(SRC/'prior-gate.json')
    seeds = read(SRC/'seed-audit.json')
    protocol = dict(normalizers=read(SRC.parent/'h3br/protocol.json')['normalizers'],
                    truths=prior['prior_rows'], fitted_prior_rows=prior['fitted_prior_rows'],
                    historical_protocol_sha256=seeds['historical_protocol_sha256'],
                    seed_bases=dict(permutation=2027700000, source=2027800000))
    docs = {name: read(SRC/name) for name in ['prior-gate.json', 'fitted-weight-gate.json', 'seed-audit.json', 'two-score-gate.json']}
    checks, rng_attempts = 0, 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    def deny(*args, **kwargs):
        nonlocal rng_attempts
        rng_attempts += 1
        raise AssertionError('Source-only audit attempted an RNG draw')
    original_rng = np.random.default_rng
    def replay(p, documents):
        def fixture_read(path):
            if path.parent == folder/'source-gates':
                return documents[path.name]
            return read(path)
        return verify_prior(ROOT, folder, p, fixture_read, sha, check)
    try:
        np.random.default_rng = deny
        replay(protocol, docs)
        passed_checks = checks
        mutations = {}
        for mode in ['uniform_ticket_multiplicity', 'uniform_sampling_row', 'fitted_prior_row',
                     'fitted_integer_weight', 'actual_uniform_mass', 'historical_range', 'historical_seed_digest', 'uniform_fitted_score', 'uniform_weight_receipt', 'ineligible_support_drop', 'T_MAP_score_role']:
            p, documents = copy.deepcopy(protocol), copy.deepcopy(docs)
            if mode == 'uniform_ticket_multiplicity':
                documents['prior-gate.json']['model_ticket_multiplicities'][0] += 1
            elif mode == 'uniform_sampling_row':
                p['truths'][0]['sampling_integer_weight'] = 1485
            elif mode == 'fitted_prior_row':
                p['fitted_prior_rows'][0]['prior_integer_weight'] = 1
            elif mode == 'fitted_integer_weight':
                documents['fitted-weight-gate.json']['integer_weight_receipt']['integer_weights'][0] = '1'
            elif mode == 'actual_uniform_mass':
                documents['fitted-weight-gate.json']['actual_uniform_prior_mass'][1] = documents['fitted-weight-gate.json']['fitted_mass'][1]
            elif mode == 'historical_range':
                documents['seed-audit.json']['expanded_known_truth_source_ranges'][-1]['seeds'] -= 1
            elif mode == 'uniform_fitted_score':
                documents['two-score-gate.json']['T_scores'][1] = documents['two-score-gate.json']['H_scores'][1]
            elif mode == 'uniform_weight_receipt':
                documents['two-score-gate.json']['T_fixture']['integer_weights'][1] = '1'
            elif mode == 'ineligible_support_drop':
                documents['two-score-gate.json']['T_scores'][0] = None
            elif mode == 'T_MAP_score_role':
                documents['two-score-gate.json']['selector_scores']['T'] = documents['two-score-gate.json']['T_scores']
            else:
                assert mode == 'historical_seed_digest'
                documents['seed-audit.json']['historical_protocol_sha256'][next(iter(seeds['historical_protocol_sha256']))] = '0'*64
            try:
                replay(p, documents)
            except AssertionError as error:
                expected = dict(uniform_ticket_multiplicity='runtime_prior_gate_identity',
                    uniform_sampling_row='all_exact_prior_tickets',fitted_prior_row='separate_fixed_fitted_hierarchy',
                    fitted_integer_weight='independent_separate_fitted_weight_gate',actual_uniform_mass='independent_separate_fitted_weight_gate',
                    historical_range='nearby_other_machine_solver_ranges',historical_seed_digest='historical_seed_protocols_bound',
                    uniform_fitted_score='independent_uniform_and_hierarchical_scores_and_support',
                    uniform_weight_receipt='independent_uniform_and_hierarchical_scores_and_support',
                    ineligible_support_drop='independent_uniform_and_hierarchical_scores_and_support',
                    T_MAP_score_role='independent_uniform_and_hierarchical_scores_and_support')
                assert str(error) == expected[mode], (mode,str(error))
                mutations[mode] = True
            else:
                raise AssertionError('Semantic source mutation accepted: '+mode)
        assert rng_attempts == 0
    finally:
        np.random.default_rng = original_rng
    result = dict(status='passed_independent_two_score_sampling_support_gate',
                  successful_replay_checks=passed_checks, total_checks_including_rejected_mutations=checks,
                  semantic_mutations_rejected=mutations, RNG_API_attempts=rng_attempts,
                  fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0,
                  source_sha256={name: sha(SRC/name) for name in ['audit_sources.py', 'audit_source_gate.py', 'source_gate.py', 'prior.py', 'weights.py']},
                  gate_sha256={name: sha(SRC/name) for name in docs},
                  elapsed_seconds=time.perf_counter()-started)
    save(folder/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    print(run(parser.parse_args().folder))
