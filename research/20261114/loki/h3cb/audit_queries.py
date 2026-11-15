"""Independent exact binding of every paid query and answer in both arms."""


class PaidQueries:
    def __init__(self, events, predictions, boards, check, arms=('deeper','baseline')):
        self.events, self.pred, self.boards, self.check = events, predictions, boards, check
        self.at = 0
        self.history = {}
        self.calls = {arm:0 for arm in arms}

    def take(self, kind):
        self.check('event_available_'+kind, self.at < len(self.events))
        event = self.events[self.at]
        self.at += 1
        self.check('exact_event_order_'+kind, event['kind'] == kind)
        return event

    def ask(self, case, truth, arm, phase, candidates, queried, request, selection):
        index = selection['query_index']
        step = len(queried)
        self.check('selected_board_and_full_candidate_binding', index is not None and
                   index not in queried and step < 8 and candidates == sorted(set(candidates)) and
                   truth in candidates and request['candidates'] == candidates and
                   request['queried'] == queried and selection['candidate_predictions'] ==
                   {str(c): bool(self.pred[c, index]) for c in candidates})
        prior = list(self.history.get((case, index), []))
        query = self.take('query')
        self.check('complete_query_record_binding', query['data'] == dict(
                   case=case, arm=arm, phase=phase, step=step, request=request, selection=selection,
                   board_zero_based=self.boards[index].tolist(), previous_same_case_query_sequences=prior))
        label = bool(self.pred[truth, index])
        remaining = [c for c in candidates if bool(self.pred[c, index]) == label]
        answer = self.take('answer')
        self.check('immediate_answer_truth_and_filter_binding', answer['data'] == dict(
                   case=case, arm=arm, phase=phase, step=step, query_sha256=query['sha256'],
                   label=label, candidates_before=candidates, candidates_after=remaining) and
                   query['monotonic_ns'] < answer['monotonic_ns'])
        self.history.setdefault((case, index), []).append(query['sequence'])
        self.calls[arm] += 1
        return remaining, dict(step=step, query_index=index, label=label,
                              candidates_before=candidates, candidates_after=remaining,
                              query_sequence=query['sequence'], query_sha256=query['sha256'],
                              answer_sequence=answer['sequence'], answer_sha256=answer['sha256'],
                              previous_same_case_query_sequences=prior)
