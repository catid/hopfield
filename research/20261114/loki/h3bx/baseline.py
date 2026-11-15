"""Original H3br depth-four policy with its own complete-proof cache."""
import hashlib
import time
import numpy as np
from common import SRC, save, sha, now
from baseline_compiler import compile_tree as weighted_tree
from baseline_firstlex import compile_tree as first_tree
from weights import integer_weights
from ledger import canonical
from policy import compact


class OriginalBaseline:
    def __init__(self, folder, predictions, scores, predictions_sha256, budget, max_bits):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=False)
        self.predictions = predictions
        self.scores = tuple(scores)
        self.predictions_sha256 = predictions_sha256
        self.budget = budget
        self.max_bits = max_bits
        self.cache = {}
        self.rows = []

    def get(self, candidates):
        candidates = list(candidates)
        assert candidates == sorted(set(candidates)) and len(candidates) >= 2
        key = tuple(candidates)
        if key in self.cache:
            return self.cache[key]
        group = len(self.rows)
        folder = self.folder / f'group{group}'
        folder.mkdir()
        before_states, before_work = self.budget.states, self.budget.splits
        begin_cpu, begin_wall = time.process_time(), time.perf_counter()
        catalog = None
        catalog_sha = None
        if len(candidates) > 16:
            self.budget.state()
            full = (1 << len(candidates)) - 1
            root = f'{full}:4'
            first = dict(candidates=candidates, depth_limit=4, root_state=root,
                         feasible=False, states={root: dict(mask=full,
                         depth_remaining=4, models=len(candidates), feasible=False,
                         reason='binary_information_bound', attempts=[])},
                         leaf_depths={}, worst_leaf_depth=None)
        else:
            codes = np.zeros(self.predictions.shape[1], np.uint32)
            for local, candidate in enumerate(candidates):
                codes |= self.predictions[candidate].astype(np.uint32) << local
            signatures, indices, counts = np.unique(codes, return_index=True, return_counts=True)
            catalog = [dict(signature=int(s), lexfirst_board_index=int(i), board_count=int(n))
                       for s, i, n in zip(signatures, indices, counts)]
            save(folder / 'catalog.json', dict(candidates=candidates,
                 universe_boards=self.predictions.shape[1], signatures=catalog))
            catalog_sha = sha(folder / 'catalog.json')
            first = first_tree(candidates, catalog, 4, self.budget)
        first_states = self.budget.states - before_states
        first_work = self.budget.splits - before_work
        save(folder / 'firstlex/proof.json', first)
        weights = None
        weights_sha = None
        if first['feasible']:
            weights, receipt = integer_weights(candidates, self.scores, self.max_bits)
            save(folder / 'posterior/weights.json', receipt)
            weights_sha = sha(folder / 'posterior/weights.json')
            proof = weighted_tree(candidates, weights, catalog, 4, self.budget)
            assert proof['feasible']
            assert int(proof['optimal_cost']) <= sum(w * first['leaf_depths'][str(c)]
                                                    for c, w in zip(candidates, weights))
        else:
            proof = first
        inputs = dict(policy='original_H3br_posterior', candidates=candidates,
                      weights=list(map(str, weights)) if weights is not None else None,
                      depth=4, catalog_sha256=catalog_sha,
                      predictions_sha256=self.predictions_sha256,
                      firstlex_compiler_sha256=sha(SRC / 'baseline_firstlex.py'),
                      posterior_compiler_sha256=sha(SRC / 'baseline_compiler.py'),
                      shared_infeasibility_certificate=not first['feasible'],
                      allowed_inputs_only=True, cross_arm_cache_reuse=False)
        save(folder / 'compiler-input.json', inputs)
        save(folder / 'posterior/proof.json', proof)
        row = dict(group=group, candidates=candidates, depth=4, utc=now(),
                   feasible=proof['feasible'], catalog_sha256=catalog_sha,
                   weights_sha256=weights_sha,
                   firstlex_proof_sha256=sha(folder / 'firstlex/proof.json'),
                   proof_sha256=sha(folder / 'posterior/proof.json'),
                   compiler_input_sha256=sha(folder / 'compiler-input.json'),
                   cache_key=hashlib.sha256(canonical(inputs)).hexdigest(),
                   firstlex_memo_states=first_states,
                   firstlex_split_attempts=first_work,
                   posterior_memo_states=self.budget.states-before_states-first_states,
                   posterior_split_attempts=self.budget.splits-before_work-first_work,
                   memo_states=self.budget.states-before_states,
                   work_steps=self.budget.splits-before_work,
                   CPU_seconds=time.process_time()-begin_cpu,
                   wall_seconds=time.perf_counter()-begin_wall)
        save(folder / 'result.json', row)
        self.rows.append(row)
        self.cache[key] = (row, compact(proof))
        return self.cache[key]
