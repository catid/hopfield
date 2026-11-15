"""Fixed nonrandom fallback fixtures, independent replay and semantic mutations."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
import numpy as np
from common import SRC,save,sha,load
from ledger import Ledger,canonical
from oracle import PaidOracle
from audit_queries import PaidQueries
from fallback import ScanBudget,select,continue_case,extend_result
from audit_fallback import selection_reference,verify


def checkpoint(case,truth,active,arm='E',maps=4,exact=False):
    steps=[dict(query_index=i,previous_same_case_query_sequences=[]) for i in range(maps)]
    return dict(arm=arm,case=case,candidate=truth,initial_candidates=active,map_steps=steps,tree_steps=[],map_calls=maps,
        tree_calls=0,handoff_MAP_calls=None,terminal_no_information=None,
        routing=[dict(reason='candidate_count_eligibility_limit')],remaining_candidates=active,
        exact_true_singleton=exact,membership_calls=maps,independently_paid_cross_arm_reacquisitions=0)


def run(folder,budget=None):
    folder.mkdir(parents=True,exist_ok=False);checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_fallback_chain').chain
    pred=np.array([[False]*4+[bool(c&(1<<j)) for j in range(5)]+[bool(c&1),c==0,True] for c in range(20)])
    boards=np.arange(pred.shape[1],dtype=np.uint8)[:,None]
    np.save(folder/'predictions-bits.npy',np.packbits(pred,axis=1,bitorder='little'))
    np.save(folder/'boards.npy',boards)
    check('selector_has_no_truth_argument',list(inspect.signature(select).parameters)==['predictions','candidates','queried','budget'])
    budget=ScanBudget() if budget is None else budget
    start_cells,start_requests=budget.cells,budget.requests
    records=[]
    ledger=Ledger(folder/'query-ledger.jsonl');oracle=PaidOracle(ledger,boards,np.packbits(pred,axis=1,bitorder='little'),arms=('B','E'))
    bases=[checkpoint(c,c,list(range(20))) for c in range(20)]
    bases += [checkpoint(20,4,[4],maps=0,exact=True),checkpoint(21,0,list(range(20)),arm='B'),
              checkpoint(22,0,list(range(16)),maps=4),checkpoint(23,0,list(range(20)),maps=3)]
    for original in bases:
        extension=continue_case(original,pred,ledger,oracle.ask,budget)
        records.append(extend_result(original,extension))
    ledger.close();events=chain((folder/'query-ledger.jsonl').read_bytes())
    save(folder/'original-checkpoints.json',bases);save(folder/'case-results.json',records)
    def replay(rows):
        audit=PaidQueries(rows,pred,boards,check,arms=('B','E'))
        for original,stored in zip(bases,records):
            result=verify(original,pred,audit,check)
            check('complete_fixed_fallback_reconstruction',result==stored)
        check('complete_fixture_ledger_and_paid_calls',audit.at==len(rows) and audit.calls==oracle.arm_calls)
    replay(events)
    rescued=sum(r['exact_true_singleton'] for r in records[:20])
    unresolved=20-rescued
    check('both_rescued_and_four_call_unresolved_paths',rescued>0 and unresolved>0 and
        all(r['fallback']['additional_calls']<=4 for r in records) and
        all(r['fallback']['stop']=='four_additional_queries_exhausted' for r in records[:20] if not r['exact_true_singleton']))
    check('all_noneligible_and_comparator_stops_zero_queries',all(r['fallback']['additional_calls']==r['fallback']['candidate_board_cells']==0 for r in records[20:]))
    check('more_than16_cannot_have_universal_four_query_recovery',rescued<20)
    # Additional independent no-information stop with a retained twenty-model set.
    noinfo=np.zeros((20,8),bool);nf=folder/'no-information';nf.mkdir()
    nb=budget;nl=Ledger(nf/'query-ledger.jsonl');no=PaidOracle(nl,np.arange(8,dtype=np.uint8)[:,None],np.packbits(noinfo,axis=1,bitorder='little'),arms=('B','E'))
    original=checkpoint(0,0,list(range(20)));extension=continue_case(original,noinfo,nl,no.ask,nb);nl.close()
    replay_n=PaidQueries(chain((nf/'query-ledger.jsonl').read_bytes()),noinfo,np.arange(8,dtype=np.uint8)[:,None],check,arms=('B','E'))
    result=verify(original,noinfo,replay_n,check)
    check('no_information_has_one_scan_zero_labels',result==extend_result(original,extension) and extension['stop']=='no_informative_query' and
        no.calls==0 and extension['selection_requests']==1 and extension['candidate_board_cells']==80 and replay_n.at==len(replay_n.events))
    save(nf/'case-result.json',result)
    # Requested work is rejected before any count scan, paid label, or accepted receipt.
    cf=folder/'scan-cap';cf.mkdir();cb=ScanBudget(159);cl=Ledger(cf/'query-ledger.jsonl');co=PaidOracle(cl,boards,np.packbits(pred,axis=1,bitorder='little'),arms=('B','E'))
    try:continue_case(checkpoint(0,0,list(range(20))),pred,cl,co.ask,cb)
    except RuntimeError as error:cap_error=str(error)
    else:raise AssertionError('Fallback cap escaped')
    cl.close()
    check('scan_cap_before_label_or_fallback_receipt',co.calls==cb.cells==cb.requests==0 and (cf/'query-ledger.jsonl').stat().st_size==0 and
        cb.rejected==dict(requested_candidate_board_cells=160,completed_candidate_board_cells=0,maximum_candidate_board_cells=159))
    save(cf/'receipt.json',dict(error=cap_error,budget=cb.snapshot(),paid_queries=co.calls,whole_study_incomplete=True))
    # Exhaustive small truth-independent selectors include unique optima, ties and exclusions.
    small=np.array([[bool(mask&(1<<c)) for mask in range(1,16)] for c in range(4)])
    comparisons=0;unique=tied=0
    for subset in range(1,16):
        active=[c for c in range(4) if subset&(1<<c)]
        if len(active)<2:continue
        for queried in [[],[0],[1,3],[0,1,2,3,4,5,6]]:
            expected=selection_reference(small,active,queried);actual=select(small,active,queried,budget)
            check('exhaustive_small_argmin_tie_and_exclusion',actual==expected)
            comparisons+=1;unique+=expected['optimum_tied_board_count']==1;tied+=expected['optimum_tied_board_count']>1
    one=np.array([[0,0,1],[0,0,1],[1,0,1],[1,0,1]],bool)
    unique_result=select(one,[0,1,2,3],[],budget)
    check('unique_optimum_fixture',unique_result==selection_reference(one,[0,1,2,3],[]) and unique_result['optimum_tied_board_count']==1)
    # All mutation ledgers are rehashed, so failures cannot rely on stale chain hashes.
    def rechain(rows):
        for i,row in enumerate(rows):
            row.pop('sha256',None);row['sequence']=i;row['previous_sha256']=rows[i-1]['sha256'] if i else '0'*64
            if row['kind']=='answer' and i and rows[i-1]['kind']=='query':row['data']['query_sha256']=rows[i-1]['sha256']
            row['sha256']=hashlib.sha256(canonical(row)).hexdigest()
        return chain(b''.join(canonical(r)+b'\n' for r in rows))
    first=next(i for i,e in enumerate(events) if e['kind']=='fallback_selection_bound')
    complete=next(i for i,e in enumerate(events) if e['kind']=='fallback_case_complete' and e['data']['case']==0)
    def chosen(rows,index):
        rows[first]['data']['selection'].update(query_index=index,candidate_predictions={str(c):bool(pred[c,index]) for c in range(20)})
    mutations={
        'truth_directed_nonoptimal_board':lambda r:chosen(r,10),
        'later_equal_optimum_tie':lambda r:chosen(r,9),
        'previously_queried_board':lambda r:chosen(r,0),
        'incorrect_candidate_count':lambda r:r[first]['data']['selection'].update(worst_branch_count=9),
        'undercharged_full_board_scan':lambda r:r[first]['data']['selection'].update(candidate_board_cells=159),
        'wrong_oracle_label':lambda r:r[first+2]['data'].update(label=not r[first+2]['data']['label']),
        'premature_true_singleton':lambda r:r[complete]['data'].update(remaining_candidates=[0],exact_true_singleton=True),
        'unpaid_label_reuse':lambda r:r.__delitem__(slice(first+1,first+3)),
    }
    rejected={}
    for name,change in mutations.items():
        rows=copy.deepcopy(events);change(rows);rows=rechain(rows)
        try:replay(rows)
        except AssertionError as error:rejected[name]=str(error)
        else:raise AssertionError('Semantic mutation escaped: '+name)
    check('distinct_named_semantic_rejections',len(rejected)==8 and all(rejected.values()))
    # Public arithmetic fixture; no source bank, truth schedule or RNG call.
    synthetic=((np.arange(64)[:,None]*17+np.arange(8192)[None,:]*13)%31)<15
    fast=[];scalar=[]
    for _ in range(3):
        start=time.perf_counter();v=select(synthetic,list(range(64)),[0,7,19],budget);fast.append(time.perf_counter()-start)
        start=time.perf_counter();w=selection_reference(synthetic,list(range(64)),[0,7,19]);scalar.append(time.perf_counter()-start)
        check('fixed_scan_performance_exact_full_output',v==w)
    timing=dict(shape=[64,8192],repetitions=3,vector_seconds=fast,scalar_seconds=scalar,
        vector_median=statistics.median(fast),scalar_median=statistics.median(scalar),
        median_ratio=statistics.median(scalar)/statistics.median(fast),candidate_board_cells_per_repeat=64*(8192-3),
        scope='Synthetic selector-only timing; not scientific end-to-end speedup or a fresh-source pilot')
    result=dict(status='passed_fixed_balanced_fallback_and_independent_semantic_audit',checks=checks,
        fixed_truth_paths=20,rescued=rescued,unresolved=unresolved,synthetic_paid_queries=oracle.calls,
        paid_queries_are_synthetic=True,actual_oracle_calls=0,fresh_truth_draws=0,fresh_source_draws=0,
        exhaustive_selector_comparisons=comparisons,tied_comparisons=tied,explicit_unique_optimum=True,
        charged_candidate_board_cells=budget.cells-start_cells,charged_selection_requests=budget.requests-start_requests,
        scan_budget=budget.snapshot(),scan_cap=cb.snapshot(),semantic_mutations_rejected=rejected,benchmark=timing,
        source_sha256={n:sha(SRC/n) for n in ['fallback.py','audit_fallback.py','fallback_gate.py','oracle.py','audit_queries.py','ledger.py']})
    save(folder/'summary.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);run(parser.parse_args().folder)
