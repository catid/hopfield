"""Independent scalar full-board argmin/tie and paid fallback reconstruction."""
import hashlib
import struct


def selection_reference(pred,candidates,queried):
    records=[];positive_counts=[]
    for board in range(pred.shape[1]):
        if board in queried:continue
        yes=sum(bool(pred[c,board]) for c in candidates)
        positive_counts.append(yes)
        if 0<yes<len(candidates):records.append((max(yes,len(candidates)-yes),board))
    if records:
        worst,index=min(records)
        ties=sum(v==worst for v,_ in records)
        values={str(c):bool(pred[c,index]) for c in candidates}
    else:worst=index=None;ties=0;values={}
    raw=b''.join(struct.pack('<I',v) for v in positive_counts)
    return dict(query_index=index,candidate_predictions=values,worst_branch_count=worst,
        optimum_tied_board_count=ties,considered_board_count=len(positive_counts),
        candidate_board_cells=len(candidates)*len(positive_counts),
        positive_counts_little_endian_uint32_sha256=hashlib.sha256(raw).hexdigest())


def verify(original,pred,ledger,check):
    arm,case,truth=original['arm'],original['case'],original['candidate']
    active=list(original['remaining_candidates'])
    queried=[r['query_index'] for r in original['map_steps']+original['tree_steps']]
    qualifies=(active!=[truth] and original['map_calls']==4 and not original['tree_calls'] and
        original['handoff_MAP_calls'] is None and len(active)>16 and original['terminal_no_information'] is None and
        original['routing'][-1].get('reason')=='candidate_count_eligibility_limit')
    entered=arm=='E' and qualifies
    selected=[];steps=[];cells=requests=0
    if not entered:stop='comparator_disabled' if arm=='B' else 'original_exact' if active==[truth] else 'original_stop_ineligible'
    else:
        stop=None
        while len(steps)<4 and len(active)>1:
            request=dict(policy='minimize_maximum_candidate_branch_then_first_board',candidates=active,queried=queried,board_count=pred.shape[1])
            selection=selection_reference(pred,active,queried)
            event=ledger.take('fallback_selection_bound')
            check('full_boardwise_fallback_argmin_and_first_tie',event['data']==dict(case=case,arm=arm,additional_queries=len(steps),request=request,selection=selection))
            selected.append(dict(request=request,selection=selection,sequence=event['sequence'],sha256=event['sha256']))
            cells+=selection['candidate_board_cells'];requests+=1
            if selection['query_index'] is None:stop='no_informative_query';break
            active,step=ledger.ask(case,truth,arm,'fallback',active,queried,request,selection)
            steps.append(step);queried=queried+[step['query_index']]
        if stop is None:stop='true_singleton' if len(active)==1 else 'four_additional_queries_exhausted'
    check('fallback_truthful_stop_and_eight_call_limit',truth in active and (len(active)!=1 or active==[truth]) and len(queried)<=8)
    event=ledger.take('fallback_case_complete');data=event['data']
    extension=dict(eligible_original_stop=qualifies,entered=entered,selections=selected,steps=steps,stop=stop,
        remaining_candidates=active,exact_true_singleton=active==[truth],additional_calls=len(steps),
        candidate_board_cells=cells,selection_requests=requests,CPU_seconds=data['CPU_seconds'],wall_seconds=data['wall_seconds'])
    check('fallback_final_receipt_and_nonnegative_timing',data==dict(case=case,arm=arm,**extension) and
          data['CPU_seconds']>=0 and data['wall_seconds']>=0)
    if original['exact_true_singleton']:check('zero_additional_queries_for_original_exact',not entered and not steps and cells==0)
    result=dict(original,original_policy_result=dict(original),fallback=extension,remaining_candidates=active,
        exact_true_singleton=active==[truth],membership_calls=original['membership_calls']+len(steps),
        independently_paid_cross_arm_reacquisitions=original['independently_paid_cross_arm_reacquisitions']+
            sum(bool(s['previous_same_case_query_sequences']) for s in steps))
    return result
