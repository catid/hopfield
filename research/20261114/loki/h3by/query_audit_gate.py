"""Rehashed semantic tamper rejection for the new independent paid-query audit."""
import hashlib
import json
import numpy as np
from common import SRC, load, save, sha
from ledger import Ledger, canonical
from audit_queries import PaidQueries
from audit_costs import choose


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    pred = np.array([[bool(sig & (1<<c)) for sig in range(16)] for c in range(4)])
    boards = np.arange(16, dtype=np.uint8)[:, None]
    scores = ['1', '2', '3', '4']
    candidates, truth = [0, 1, 2, 3], 2
    selection = choose(pred, scores, candidates, [])
    request = dict(bits_path='predictions-bits.npy', scores_path='score-values8.json',
                   board_count=16, candidates=candidates, queried=[])
    index = selection['query_index']
    label = bool(pred[truth, index])
    remaining = [c for c in candidates if bool(pred[c, index]) == label]
    ledger = Ledger(folder/'synthetic-ledger.jsonl')
    prior = []
    for arm in ['deeper', 'baseline']:
        q = ledger.append('query', dict(case=0, arm=arm, phase='map', step=0,
                          request=request, selection=selection, board_zero_based=boards[index].tolist(),
                          previous_same_case_query_sequences=list(prior)))
        ledger.append('answer', dict(case=0, arm=arm, phase='map', step=0,
                      query_sha256=q['sha256'], label=label,
                      candidates_before=candidates, candidates_after=remaining))
        prior.append(q['sequence'])
    ledger.close()
    chain = load(SRC.parent/'h3bj/verify_public.py', 'query_audit_gate_chain').chain
    events = chain((folder/'synthetic-ledger.jsonl').read_bytes())
    def replay(rows):
        checks = 0
        def check(name, value):
            nonlocal checks
            assert value, name
            checks += 1
        audit = PaidQueries(rows, pred, boards, check)
        for arm in ['deeper', 'baseline']:
            active, _ = audit.ask(0, truth, arm, 'map', candidates, [], request, selection)
            assert active == remaining
        assert audit.at == len(rows) and audit.calls == dict(deeper=1, baseline=1)
        assert audit.history[(0, index)] == [0, 2]
        return checks
    checks = replay(events)
    def rechain(rows):
        for i, row in enumerate(rows):
            row.pop('sha256', None)
            row['sequence'] = i
            row['previous_sha256'] = rows[i-1]['sha256'] if i else '0'*64
            if row['kind'] == 'answer' and i and rows[i-1]['kind'] == 'query':
                row['data']['query_sha256'] = rows[i-1]['sha256']
            row['sha256'] = hashlib.sha256(canonical(row)).hexdigest()
        return chain(b''.join(canonical(row)+b'\n' for row in rows))
    mutations = {
        'wrong_case': lambda r: r[2]['data'].update(case=1),
        'wrong_arm': lambda r: r[2]['data'].update(arm='deeper'),
        'wrong_phase': lambda r: r[2]['data'].update(phase='tree'),
        'inherited_query': lambda r: r[2]['data']['request'].update(queried=[index]),
        'candidate_dropped_from_request': lambda r: r[2]['data']['request'].update(candidates=[0, 1, 2]),
        'forged_candidate_prediction': lambda r: r[2]['data']['selection']['candidate_predictions'].update({'0': not bool(pred[0, index])}),
        'different_board_columns': lambda r: r[2]['data'].update(board_zero_based=[255]),
        'wrong_oracle_label': lambda r: r[3]['data'].update(label=not label),
        'wrong_candidate_filter': lambda r: r[3]['data'].update(candidates_after=[]),
        'unpaid_cross_arm_reuse_annotation': lambda r: r[2]['data'].update(previous_same_case_query_sequences=[]),
    }
    rejected = {}
    for name, change in mutations.items():
        rows = json.loads(json.dumps(events))
        change(rows)
        rows = rechain(rows)
        try:
            replay(rows)
        except AssertionError:
            rejected[name] = True
        else:
            raise AssertionError('Semantic mutation escaped independent audit: '+name)
    result = dict(status='passed_paid_query_binding_and_semantic_tamper_gate', checks=checks,
                  independently_replayed_synthetic_queries=2, cross_arm_reacquisitions_paid=1,
                  all10_rehashed_semantic_mutations_rejected=rejected,
                  source_sha256={name: sha(SRC/name) for name in
                     ['audit_queries.py', 'query_audit_gate.py', 'audit_costs.py', 'ledger.py']},
                  fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    save(folder/'summary.json', result)
    return result
