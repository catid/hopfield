"""Independent full certificates and input bindings for the original baseline."""
import hashlib
import math
import numpy as np
from common import load


class OriginalAudit:
    def __init__(self, folder, source, rows, predictions, scores, predictions_sha,
                 check, read, sha, canonical):
        self.folder, self.source, self.rows = folder, source, rows
        self.pred, self.scores, self.pred_sha = predictions, scores, predictions_sha
        self.check, self.read, self.sha, self.canonical = check, read, sha, canonical
        self.first = load(source.parent/'h3bi/verify_public.py', 'original_audit_first')
        self.weighted = load(source.parent/'h3bl/verify_public.py', 'original_audit_weighted')
        self.lookup = {tuple(row['candidates']): row['group'] for row in rows}
        check('original_cache_unique', len(self.lookup) == len(rows) and
              [r['group'] for r in rows] == list(range(len(rows))))
        self.cache, self.used = {}, set()
        self.states = self.work = self.paths = 0

    def get(self, candidates, bound):
        group = self.lookup[tuple(candidates)]
        row = self.rows[group]
        self.used.add(group)
        check, read, sha = self.check, self.read, self.sha
        check('original_prebound_timestamp', row['utc'] <= bound['utc'])
        if group in self.cache:
            return row, self.cache[group]
        folder = self.folder/f'group{group}'
        first, proof = read(folder/'firstlex/proof.json'), read(folder/'posterior/proof.json')
        catalog = None
        catalog_sha = None
        if len(candidates) <= 16:
            codes = np.sum(self.pred[candidates].astype(np.uint32) *
                           (1 << np.arange(len(candidates), dtype=np.uint32))[:, None], axis=0, dtype=np.uint32)
            signatures, counts = np.unique(codes, return_counts=True)
            catalog = [dict(signature=int(s), lexfirst_board_index=int(np.flatnonzero(codes == s)[0]),
                            board_count=int(n)) for s, n in zip(signatures, counts)]
            catalog_sha = sha(folder/'catalog.json')
            check('original_complete_query_catalog', read(folder/'catalog.json') ==
                  dict(candidates=candidates, universe_boards=self.pred.shape[1], signatures=catalog))
        fs, ft, _ = self.first.certificate(first, catalog, check, 'original_firstlex_')
        weights = None
        weights_sha = None
        if first['feasible']:
            denominators = [int(self.scores[c]) for c in candidates]
            L, lengths = 1, []
            for denominator in denominators:
                L = math.lcm(L, denominator)
                lengths.append(L.bit_length())
            weights = [L//d for d in denominators]
            receipt = dict(score_denominators=list(map(str, denominators)), common_denominator=str(L),
                           LCM_intermediate_bit_lengths=lengths, integer_weights=list(map(str, weights)),
                           weight_bit_lengths=[w.bit_length() for w in weights], max_bits=100000)
            weights_sha = sha(folder/'posterior/weights.json')
            check('original_full_candidate_posterior', read(folder/'posterior/weights.json') == receipt and
                  proof['weights'] == list(map(str, weights)) and max(lengths) <= 100000 and
                  max(w.bit_length() for w in weights) <= 100000)
            ns, nt, _ = self.weighted.weighted_certificate(proof, catalog, check, 'original_weighted_')
            check('original_feasibility_and_expected_cost', proof['feasible'] and
                  int(proof['optimal_cost']) <= sum(w*first['leaf_depths'][str(c)] for c, w in zip(candidates, weights)))
        else:
            ns = nt = 0
            check('original_full_infeasibility_retained', proof == first)
        inputs = dict(policy='original_H3br_posterior', candidates=candidates,
                      weights=list(map(str, weights)) if weights is not None else None,
                      depth=4, catalog_sha256=catalog_sha, predictions_sha256=self.pred_sha,
                      firstlex_compiler_sha256=sha(self.source/'baseline_firstlex.py'),
                      posterior_compiler_sha256=sha(self.source/'baseline_compiler.py'),
                      shared_infeasibility_certificate=not first['feasible'],
                      allowed_inputs_only=True, cross_arm_cache_reuse=False)
        check('original_input_result_proof_seals', row == read(folder/'result.json') and
              row['group'] == group and row['candidates'] == first['candidates'] == proof['candidates'] == candidates and
              row['depth'] == first['depth_limit'] == proof['depth_limit'] == 4 and
              row['feasible'] == first['feasible'] == proof['feasible'] and
              row['catalog_sha256'] == catalog_sha and row['weights_sha256'] == weights_sha and
              row['firstlex_proof_sha256'] == sha(folder/'firstlex/proof.json') and
              row['proof_sha256'] == sha(folder/'posterior/proof.json') and
              row['compiler_input_sha256'] == sha(folder/'compiler-input.json') and
              read(folder/'compiler-input.json') == inputs and
              row['cache_key'] == hashlib.sha256(self.canonical(inputs)).hexdigest())
        check('original_all_state_work_counts', row['firstlex_memo_states'] == fs and
              row['firstlex_split_attempts'] == ft and row['posterior_memo_states'] == ns and
              row['posterior_split_attempts'] == nt and row['memo_states'] == fs+ns and
              row['work_steps'] == ft+nt)
        self.states += fs+ns
        self.work += ft+nt
        chosen = {}
        def retain(key):
            node = proof['states'][key]
            chosen[key] = {k: v for k, v in node.items() if k != 'attempts'}
            if node['reason'] != 'singleton_leaf':
                retain(node['negative_state'])
                retain(node['positive_state'])
        if proof['feasible']:
            for local, candidate in enumerate(candidates):
                key, path = proof['root_state'], []
                while proof['states'][key]['reason'] != 'singleton_leaf':
                    node = proof['states'][key]
                    index = node['chosen_split']['board_index']
                    check('original_candidate_prediction_path', index not in path and
                          bool(self.pred[candidate, index]) == bool(node['chosen_split']['positive_mask'] & (1 << local)))
                    path.append(index)
                    key = node['positive_state'] if self.pred[candidate, index] else node['negative_state']
                check('original_candidate_leaf_and_depth', proof['states'][key]['mask'] == (1 << local) and
                      len(path) == proof['leaf_depths'][str(candidate)] <= 4)
                self.paths += 1
            retain(proof['root_state'])
        proof['states'] = chosen
        self.cache[group] = proof
        return row, proof

    def finish(self):
        self.check('all_original_groups_reachable', self.used == set(range(len(self.rows))))
        return self.states, self.work
