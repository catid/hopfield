"""Every requested membership read gets a durable query and separate answer."""


class PaidOracle:
    def __init__(self, ledger, boards, bits, total_limit=16384, arm_limit=8192):
        self.ledger, self.boards, self.bits = ledger, boards, bits
        self.total_limit, self.arm_limit = total_limit, arm_limit
        self.calls, self.arm_calls, self.history = 0, dict(H=0,T=0), {}

    def ask(self, case, truth, arm, phase, step, candidates, queried, request, selection):
        index = selection['query_index']
        assert arm in self.arm_calls and index is not None and index not in queried and step < 8
        assert self.calls < self.total_limit and self.arm_calls[arm] < self.arm_limit
        previous = list(self.history.get((case,index),[]))
        query = self.ledger.append('query',dict(case=case,arm=arm,phase=phase,step=step,
            request=request,selection=selection,board_zero_based=self.boards[index].tolist(),
            previous_same_case_query_sequences=previous))
        label = bool((self.bits[truth,index//8] >> (index%8)) & 1)
        self.calls += 1; self.arm_calls[arm] += 1
        remaining = [c for c in candidates if selection['candidate_predictions'][str(c)] == label]
        answer = self.ledger.append('answer',dict(case=case,arm=arm,phase=phase,step=step,
            query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining))
        assert truth in remaining
        self.history.setdefault((case,index),[]).append(query['sequence'])
        return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,
            candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],
            answer_sequence=answer['sequence'],answer_sha256=answer['sha256'],
            previous_same_case_query_sequences=previous)
