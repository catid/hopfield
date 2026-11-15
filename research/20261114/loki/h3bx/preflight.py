"""Retained synthetic gates only; this entry point never draws a fresh cohort."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import time
from pathlib import Path
from common import SRC, save, sha, now
from compiler import Budget
import synthetic_gate
from restriction import gate as restriction_gate
from pruning_gate import run as pruning_gate
from cache_gate import run as cache_gate
from cost_gate import run as cost_gate
from baseline_gate import run as baseline_gate
from audit_gate import verify as verify_inline
from audit_baseline_gate import verify as verify_baseline
from wrapper_gate import run as wrapper_gate


def main(folder):
    assert not (SRC/'protocol.json').exists()
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    budget = Budget()
    previous = synthetic_gate.OUT
    try:
        synthetic_gate.OUT = folder
        toy = synthetic_gate.run(budget)
    finally:
        synthetic_gate.OUT = previous
    restriction = restriction_gate(toy['fixtures'])
    pruning = pruning_gate(folder/'pruning', budget)
    cache = cache_gate(folder/'cache', budget)
    cost = cost_gate(folder/'cost', budget)
    baseline = baseline_gate(folder/'baseline', budget)
    wrapper = wrapper_gate(folder/'wrapper', budget)
    save(folder/'cost-summary.json', cost)
    save(folder/'restriction-summary.json', restriction)
    independent_inline = verify_inline(folder, SRC)
    independent_baseline = verify_baseline(folder/'baseline')
    result = dict(status='passed_prefreeze_synthetic_gates', utc=now(),
                  folder=str(folder), new_truth_draws=0, new_source_draws=0,
                  actual_oracle_calls=0, synthetic_memo_states=budget.states,
                  synthetic_work_steps=budget.splits,
                  pruning_gate=pruning, cache_gate=cache, cost_gate=cost,
                  baseline_gate=baseline, restriction_gate=restriction,
                  wrapper_gate=wrapper,
                  independent_inline=independent_inline,
                  independent_baseline=independent_baseline,
                  scientific_source_sha256={p.name: sha(p) for p in sorted(SRC.glob('*.py'))},
                  elapsed_seconds=time.perf_counter()-start)
    save(folder/'preflight.json', result)
    print({key: value for key, value in result.items() if key not in
           ['pruning_gate', 'cache_gate', 'cost_gate', 'baseline_gate', 'restriction_gate', 'wrapper_gate',
            'scientific_source_sha256']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
