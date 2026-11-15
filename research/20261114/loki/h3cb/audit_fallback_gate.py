"""Independently replay fixed fallback paths and charged scan totals."""
import hashlib
import json
import numpy as np
from common import SRC,load
from audit_queries import PaidQueries
from audit_fallback import verify as verify_case,selection_reference


def verify(folder,io,check):
    pred=np.array([[False]*4+[bool(c&(1<<j)) for j in range(5)]+[bool(c&1),c==0,True] for c in range(20)])
    boards=np.arange(12,dtype=np.uint8)[:,None]
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_retained_fallback_chain').chain
    ledger=PaidQueries(chain(io.data(folder/'query-ledger.jsonl')),pred,boards,check,arms=('B','E'))
    bases=io.read(folder/'original-checkpoints.json');rows=io.read(folder/'case-results.json')
    check('all_fixed_checkpoints_retained',len(bases)==len(rows)==24 and [r['case'] for r in bases]==list(range(24)))
    for base,row in zip(bases,rows):check('fixed_fallback_full_replay',verify_case(base,pred,ledger,check)==row)
    check('all_fixed_paid_labels',ledger.at==len(ledger.events) and ledger.calls==dict(B=0,E=80))
    nf=folder/'no-information';npred=np.zeros((20,8),bool)
    no=PaidQueries(chain(io.data(nf/'query-ledger.jsonl')),npred,np.arange(8,dtype=np.uint8)[:,None],check,arms=('B','E'))
    nrow=io.read(nf/'case-result.json');nresult=verify_case(nrow['original_policy_result'],npred,no,check)
    check('fixed_no_information_stop',nresult==nrow and no.at==len(no.events) and sum(no.calls.values())==0)
    cap=io.read(folder/'scan-cap/receipt.json')
    check('fixed_cap_no_scan_label_or_result',io.data(folder/'scan-cap/query-ledger.jsonl')==b'' and cap['whole_study_incomplete'] and cap['paid_queries']==0 and
        cap['budget']==dict(candidate_board_cells=0,selection_requests=0,maximum_candidate_board_cells=159,
            rejected_request=dict(requested_candidate_board_cells=160,completed_candidate_board_cells=0,maximum_candidate_board_cells=159)))
    cells=sum(r['fallback']['candidate_board_cells'] for r in rows)+nrow['fallback']['candidate_board_cells']
    requests=sum(r['fallback']['selection_requests'] for r in rows)+nrow['fallback']['selection_requests']
    small=np.array([[bool(mask&(1<<c)) for mask in range(1,16)] for c in range(4)])
    for subset in range(1,16):
        active=[c for c in range(4) if subset&(1<<c)]
        if len(active)<2:continue
        for queried in [[],[0],[1,3],[0,1,2,3,4,5,6]]:
            result=selection_reference(small,active,queried);cells+=result['candidate_board_cells'];requests+=1
    cells+=4*3+3*64*(8192-3);requests+=4
    receipt=io.read(folder/'summary.json')
    check('all_runtime_fixture_producer_scans_charged',receipt['charged_candidate_board_cells']==cells and receipt['charged_selection_requests']==requests and
          receipt['fresh_truth_draws']==receipt['fresh_source_draws']==receipt['actual_oracle_calls']==0)
    expected={'truth_directed_nonoptimal_board':'full_boardwise_fallback_argmin_and_first_tie',
              'later_equal_optimum_tie':'full_boardwise_fallback_argmin_and_first_tie',
              'previously_queried_board':'full_boardwise_fallback_argmin_and_first_tie',
              'incorrect_candidate_count':'full_boardwise_fallback_argmin_and_first_tie',
              'undercharged_full_board_scan':'full_boardwise_fallback_argmin_and_first_tie',
              'wrong_oracle_label':'immediate_answer_truth_and_filter_binding',
              'premature_true_singleton':'fallback_final_receipt_and_nonnegative_timing',
              'unpaid_label_reuse':'exact_event_order_query'}
    check('bound_rejection_receipts',receipt['semantic_mutations_rejected']==expected)
    for name,digest in receipt['source_sha256'].items():check('fixed_gate_source_identity',io.sha(SRC/name)==digest)
    return dict(charged_candidate_board_cells=cells,charged_selection_requests=requests,synthetic_paid_queries=80,
                every_fixed_label_and_stop_replayed=True,semantic_mutation_receipts=expected)
