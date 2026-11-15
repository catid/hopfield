"""Independent proof/cache checks of the separately executed baseline wrapper."""
import hashlib
import numpy as np
from common import SRC, read, save, sha, load
from baseline import OriginalBaseline
from ledger import canonical


def run(folder, budget):
    folder.mkdir(parents=True, exist_ok=False)
    first_audit = load(SRC.parent/'h3bi/verify_public.py', 'baseline_wrapper_first_audit')
    weighted_audit = load(SRC.parent/'h3bl/verify_public.py', 'baseline_wrapper_weighted_audit')
    start_states, start_work = budget.states, budget.splits
    checks = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    fixtures = [dict(name='two_uniform', n=2, scores=['1', '1'], signatures=list(range(4)), feasible=True),
                dict(name='six_weighted', n=6, scores=['1', '2', '4', '8', '16', '32'], signatures=list(range(64)), feasible=True),
                dict(name='six_infeasible', n=6, scores=['1']*6, signatures=[0, 1, 2, 4, 8, 16, 32, 63], feasible=False),
                dict(name='seventeen_information', n=17, scores=['1']*17, signatures=[0, (1<<17)-1], feasible=False)]
    rows = []
    for fixture in fixtures:
        n = fixture['n']
        pred = np.array([[bool(sig & (1<<cid)) for sig in fixture['signatures']] for cid in range(n)])
        pred.flags.writeable = False
        before_states, before_work = budget.states, budget.splits
        wrapper = OriginalBaseline(folder/fixture['name'], pred, fixture['scores'], 'a'*64, budget, 100000)
        row, route = wrapper.get(list(range(n)))
        states, work = budget.states, budget.splits
        again = wrapper.get(list(range(n)))
        check('identical_baseline_cache_zero_work', again == (row, route) and
              budget.states == states and budget.splits == work and len(wrapper.rows) == 1)
        sub = folder/fixture['name']/'group0'
        first, proof = read(sub/'firstlex/proof.json'), read(sub/'posterior/proof.json')
        catalog = read(sub/'catalog.json')['signatures'] if (sub/'catalog.json').exists() else None
        ns, nt, _ = first_audit.certificate(first, catalog, check, 'first_')
        if first['feasible']:
            xs, xt, _ = weighted_audit.weighted_certificate(proof, catalog, check, 'weighted_')
        else:
            xs = xt = 0
            check('shared_infeasibility_no_extra_search', proof == first)
        inputs = read(sub/'compiler-input.json')
        check('baseline_wrapper_identity', row['feasible'] == fixture['feasible'] == proof['feasible'] and
              row['firstlex_proof_sha256'] == sha(sub/'firstlex/proof.json') and
              row['proof_sha256'] == sha(sub/'posterior/proof.json') and
              row['compiler_input_sha256'] == sha(sub/'compiler-input.json') and
              row['cache_key'] == hashlib.sha256(canonical(inputs)).hexdigest() and
              inputs['depth'] == 4 and inputs['cross_arm_cache_reuse'] is False and
              inputs['shared_infeasibility_certificate'] == (not proof['feasible']))
        check('baseline_wrapper_work', ns == row['firstlex_memo_states'] and nt == row['firstlex_split_attempts'] and
              xs == row['posterior_memo_states'] and xt == row['posterior_split_attempts'] and
              ns+xs == row['memo_states'] == budget.states-before_states and
              nt+xt == row['work_steps'] == budget.splits-before_work)
        if proof['feasible']:
            for i in range(n):
                keys = [p['root_state'] for p in [proof, route]]
                paths = [[], []]
                for j, tree in enumerate([proof, route]):
                    while tree['states'][keys[j]]['reason'] != 'singleton_leaf':
                        node = tree['states'][keys[j]]
                        index = node['chosen_split']['board_index']
                        paths[j].append(index)
                        keys[j] = node['positive_state'] if pred[i, index] else node['negative_state']
                    check('correct_candidate_leaf', tree['states'][keys[j]]['mask'] == (1<<i))
                check('compact_baseline_exact_route', paths[0] == paths[1] and len(paths[0]) <= 4)
        rows.append(dict(fixture=fixture, result=row))
    result = dict(status='passed_independently_verified_baseline_wrapper_gate', checks=checks,
                  fixtures=rows, memo_states_added=budget.states-start_states,
                  work_steps_added=budget.splits-start_work,
                  source_sha256={name: sha(SRC/name) for name in ['baseline.py', 'baseline_compiler.py',
                     'baseline_firstlex.py', 'policy.py', 'weights.py', 'wrapper_gate.py']},
                  fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    save(folder/'summary.json', result)
    return result
