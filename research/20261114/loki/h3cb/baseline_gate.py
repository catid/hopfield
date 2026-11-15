"""Original baseline versus all-inline depth-four on fixed synthetic inputs."""
import gzip
import hashlib
import json
import time
from common import SRC, save, sha
from baseline_compiler import compile_tree as original
from baseline_firstlex import compile_tree as firstlex
from compiler import compile_tree as inline
from pruning_gate import inputs, paths
from verify_tree import weighted_certificate


def run(folder, budget):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    before_states, before_work = budget.states, budget.splits
    checks = cases = 0
    totals = dict(original_states=0, original_work=0, inline_states=0,
                  inline_work=0, firstlex_states=0, firstlex_work=0)
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    path = folder / 'synthetic-baseline.jsonl.gz'
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb',
          fileobj=raw, compresslevel=6, mtime=0) as stream:
        for fixture in inputs():
            if fixture['depth'] != 4:
                continue
            n = len(fixture['weights'])
            catalog = [dict(signature=s, lexfirst_board_index=i, board_count=1)
                       for i, s in enumerate(fixture['signatures'])]
            a, b = budget.states, budget.splits
            first = firstlex(list(range(n)), catalog, 4, budget)
            counters = dict(firstlex_states=budget.states-a, firstlex_work=budget.splits-b)
            a, b = budget.states, budget.splits
            reference = original(list(range(n)), fixture['weights'], catalog, 4, budget)
            counters.update(original_states=budget.states-a, original_work=budget.splits-b)
            a, b = budget.states, budget.splits
            proof = inline(list(range(n)), fixture['weights'], catalog, 4, budget)
            counters.update(inline_states=budget.states-a, inline_work=budget.splits-b)
            check('original_firstlex_and_weighted_feasibility', first['feasible'] == reference['feasible'])
            check('original_inline_feasibility_cost_paths',
                  proof['feasible'] == reference['feasible'] and
                  proof['optimal_cost'] == reference['optimal_cost'] and
                  paths(proof) == paths(reference))
            ns, nt, _ = weighted_certificate(proof, catalog, check, 'inline_')
            check('inline_charged_counts', ns == counters['inline_states'] and nt == counters['inline_work'])
            check('original_charged_counts', len(reference['states']) == counters['original_states'] and
                  sum(len(v['attempts']) for v in reference['states'].values()) == counters['original_work'])
            check('firstlex_charged_counts', len(first['states']) == counters['firstlex_states'] and
                  sum(len(v['attempts']) for v in first['states'].values()) == counters['firstlex_work'])
            record = dict(case=cases, input=fixture, firstlex=first, original=reference,
                          inline=proof, counters=counters)
            stream.write((json.dumps(record, sort_keys=True, separators=(',', ':'))+'\n').encode())
            for key, value in counters.items():
                totals[key] += value
            cases += 1
    assert cases == 4252
    result = dict(status='passed_fixed_synthetic_baseline_parity', cases=cases,
                  checks=checks, all_costs_feasibility_and_candidate_paths_identical=True,
                  memo_states_added=budget.states-before_states,
                  work_steps_added=budget.splits-before_work, **totals,
                  source_sha256={name: sha(SRC/name) for name in
                     ['baseline_compiler.py', 'baseline_firstlex.py', 'compiler.py',
                      'baseline_gate.py', 'pruning_gate.py', 'verify_tree.py']},
                  proof_ledger_sha256=sha(path), proof_ledger_bytes=path.stat().st_size,
                  fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0,
                  elapsed_seconds=time.perf_counter()-start)
    save(folder/'summary.json', result)
    return result
