"""Deterministic balanced continuation; selection has no truth argument."""
import hashlib
import resource
import time
import numpy as np


class ScanBudget:
    def __init__(self,maximum=500000000):
        self.maximum,self.cells,self.requests,self.rejected=maximum,0,0,None

    def charge(self,amount):
        assert type(amount) is int and amount>=0
        if self.cells+amount>self.maximum:
            self.rejected=dict(requested_candidate_board_cells=amount,completed_candidate_board_cells=self.cells,
                               maximum_candidate_board_cells=self.maximum)
            raise RuntimeError('Fallback candidate-board examination cap; whole study incomplete')
        self.cells+=amount;self.requests+=1

    def snapshot(self):
        return dict(candidate_board_cells=self.cells,selection_requests=self.requests,
                    maximum_candidate_board_cells=self.maximum,rejected_request=self.rejected)


def select(predictions,candidates,queried,budget):
    """Count every unqueried board, minimizing the largest branch; ID breaks ties."""
    assert candidates==sorted(set(candidates)) and len(candidates)>1
    assert len(queried)==len(set(queried)) and all(0<=q<predictions.shape[1] for q in queried)
    known=set(queried);boards=[q for q in range(predictions.shape[1]) if q not in known]
    budget.charge(len(candidates)*len(boards))
    positive=predictions[np.ix_(candidates,boards)].sum(axis=0,dtype=np.int64)
    negative=len(candidates)-positive
    informative=(positive>0)&(negative>0)
    counts_sha=hashlib.sha256(positive.astype('<u4').tobytes()).hexdigest()
    if informative.any():
        worst=np.maximum(positive,negative)
        optimum=int(worst[informative].min())
        tied=np.flatnonzero(informative & (worst==optimum))
        position=int(tied[0]);index=boards[position]
        predictions_at_choice={str(c):bool(predictions[c,index]) for c in candidates}
    else:
        optimum=index=None;tied=[];predictions_at_choice={}
    return dict(query_index=index,candidate_predictions=predictions_at_choice,
                worst_branch_count=optimum,optimum_tied_board_count=len(tied),
                considered_board_count=len(boards),candidate_board_cells=len(candidates)*len(boards),
                positive_counts_little_endian_uint32_sha256=counts_sha)


def eligible(original):
    return (not original['exact_true_singleton'] and original['map_calls']==4 and
            original['tree_calls']==0 and original['handoff_MAP_calls'] is None and
            len(original['remaining_candidates'])>16 and original['terminal_no_information'] is None and
            original['routing'][-1].get('reason')=='candidate_count_eligibility_limit')


def continue_case(original,predictions,ledger,ask,budget):
    """The truth passes only to the paid oracle; it is never passed to select()."""
    arm,case,truth=original['arm'],original['case'],original['candidate']
    assert arm in ['B','E']
    active=list(original['remaining_candidates'])
    queried=[s['query_index'] for s in original['map_steps']+original['tree_steps']]
    start_cells,start_requests=budget.cells,budget.requests
    started_wall=time.perf_counter();started_cpu=time.process_time()
    selected=[];steps=[]
    qualifies=eligible(original)
    entered=arm=='E' and qualifies
    if not entered:
        stop='comparator_disabled' if arm=='B' else 'original_exact' if original['exact_true_singleton'] else 'original_stop_ineligible'
    else:
        stop=None
        while len(steps)<4 and len(active)>1:
            request=dict(policy='minimize_maximum_candidate_branch_then_first_board',
                         candidates=active,queried=queried,board_count=predictions.shape[1])
            selection=select(predictions,active,queried,budget)
            event=ledger.append('fallback_selection_bound',dict(case=case,arm=arm,additional_queries=len(steps),
                                request=request,selection=selection))
            selected.append(dict(request=request,selection=selection,sequence=event['sequence'],sha256=event['sha256']))
            if selection['query_index'] is None:
                stop='no_informative_query';break
            active,step=ask(case,truth,arm,'fallback',len(queried),active,queried,request,selection)
            steps.append(step);queried=queried+[step['query_index']]
        if stop is None:stop='true_singleton' if len(active)==1 else 'four_additional_queries_exhausted'
    assert truth in active and (len(active)!=1 or active==[truth]) and len(queried)<=8
    extension=dict(eligible_original_stop=qualifies,entered=entered,selections=selected,steps=steps,
                   stop=stop,remaining_candidates=active,exact_true_singleton=active==[truth],
                   additional_calls=len(steps),candidate_board_cells=budget.cells-start_cells,
                   selection_requests=budget.requests-start_requests,
                   CPU_seconds=time.process_time()-started_cpu,wall_seconds=time.perf_counter()-started_wall)
    ledger.append('fallback_case_complete',dict(case=case,arm=arm,**extension))
    return extension


def extend_result(original,extension):
    return dict(**original,original_policy_result=dict(original)) if extension is None else {
        **original,'original_policy_result':dict(original),'fallback':extension,
        'remaining_candidates':extension['remaining_candidates'],'exact_true_singleton':extension['exact_true_singleton'],
        'membership_calls':original['membership_calls']+extension['additional_calls'],
        'independently_paid_cross_arm_reacquisitions':original['independently_paid_cross_arm_reacquisitions']+
            sum(bool(s['previous_same_case_query_sequences']) for s in extension['steps'])}
