"""Post-audit diagnosis from sealed cases, paid labels, and supplied predictions."""
import gzip
import json
import numpy as np
from common import ROOT,SRC,read,save,sha
P=ROOT/'results/research/loki/h3bz';A=ROOT/'artifacts/research/20261114/loki'


def main():
    audit=read(A/'h3bz-full-raw-audit.json')
    assert audit['status']=='passed_complete_fresh_two_score_two_arm_audit' and audit['joint_decision']=='falsified'
    g=read(P/'capacity-gate.json');assert g['status']=='complete_verified'
    pairs,sources=read(P/'case-results.json'),read(P/'source-banks.json')
    results={a:read(P/a/'arm-results.json') for a in ['H','U']}
    bits=np.load(ROOT/'results/research/loki/h3bc/predictions-bits.npy')
    events=[json.loads(x) for x in gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes()).splitlines()]
    query={e['sequence']:e['data'] for e in events if e['kind']=='query'}
    manifest={r['logical_path']:r for r in read(P/'proof-export.json')['proofs']}
    rows=[]
    for pair in pairs:
        if pair['H_exact']==pair['U_exact']:continue
        case,truth=pair['case'],pair['candidate'];failed='U' if pair['H_exact'] else 'H';successful='H' if failed=='U' else 'U'
        fail,success=results[failed][case],results[successful][case]
        remaining=fail['remaining_candidates'];eq={}
        for c in remaining:eq.setdefault(bytes(bits[c]),[]).append(c)
        clusters=list(eq.values());true_cluster=next(v for v in clusters if truth in v)
        last=fail['routing'][-1];proof=None
        if fail['terminal_no_information'] is not None:stop='no_informative_target_query'
        elif fail['map_calls']==4 and last.get('feasible') is False:
            stop='terminal_depth4_full_class_tree_infeasible_after_four_MAP'
            proof=manifest[f'{failed}/group{last["group"]}/proof.json']
            assert proof['raw_sha256']==last['proof_sha256']
        elif fail['map_calls']==4 and last.get('reason')=='candidate_count_eligibility_limit':stop='terminal_candidate_count_above16_after_four_MAP'
        else:raise AssertionError(('Unclassified stop',case,failed))
        assert truth in remaining and len(remaining)>1 and fail['tree_calls']==0
        replay=list(remaining);used=[]
        known=set(s['query_index'] for s in fail['map_steps']+fail['tree_steps'])
        for step in success['map_steps']+success['tree_steps']:
            if replay==[truth]:break
            i=step['query_index']
            if i in known:continue
            replay=[c for c in replay if bool((bits[c,i//8]>>(i%8))&1)==step['label']]
            used.append(dict(query_index=i,label=step['label'],already_paid_successful_arm_query_sequence=step['query_sequence'],remaining_candidates=replay))
        assert replay==[truth]
        map_routes={a:[dict(query_index=step['query_index'],label=step['label'],guide_candidate=query[step['query_sequence']]['selection']['guide_candidate'],
                     candidates_before=step['candidates_before'],candidates_after=step['candidates_after']) for step in results[a][case]['map_steps']] for a in ['H','U']}
        rows.append(dict(case=case,truth_candidate=truth,successful_arm=successful,failed_arm=failed,
            initial_candidates=sources[case]['initial_candidates'],map_routes=map_routes,
            failed_remaining_candidates=remaining,failed_true_prediction_equivalence_class=true_cluster,
            failed_remaining_target_equivalence_classes=clusters,failed_stop=stop,failed_MAP_calls=fail['map_calls'],
            failed_tree_calls=fail['tree_calls'],failed_actual_calls=fail['membership_calls'],
            successful_actual_calls=success['membership_calls'],successful_handoff_MAP_calls=success['handoff_MAP_calls'],
            successful_handoff_depth=success['handoff_depth'],successful_remaining_candidates=success['remaining_candidates'],
            true_model_never_removed=all(truth in s['candidates_after'] for a in ['H','U'] for s in results[a][case]['map_steps']+results[a][case]['tree_steps']),
            wrong_terminal_model_selected=False,whole_study_resource_exhausted=False,terminal_infeasibility_proof=proof,
            posthoc_replay_of_already_paid_successful_labels=used,
            posthoc_replay_scope='Descriptive reuse of existing sealed labels on the failed final candidate set; not an executed rescue policy or fresh oracle call'))
    assert len(rows)==5 and sum(r['failed_arm']=='U' for r in rows)==1
    result=dict(status='postaudit_sealed_transcript_diagnosis',scope='Posthoc descriptive diagnosis; frozen joint endpoint remains falsified',
                cases=rows,fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0,
                full_audit_sha256=sha(A/'h3bz-full-raw-audit.json'),protocol_sha256=sha(SRC/'protocol.json'),
                diagnose_source_sha256=sha(__file__),input_sha256={str(x.relative_to(ROOT)):sha(x) for x in
                  [P/'case-results.json',P/'source-banks.json',P/'H/arm-results.json',P/'U/arm-results.json',P/'query-ledger.jsonl.gz',
                   ROOT/'results/research/loki/h3bc/predictions-bits.npy',P/'proof-export.json']})
    save(P/'discordant-case-diagnosis.json',result)
    print([dict(case=r['case'],truth=r['truth_candidate'],failed=r['failed_arm'],stop=r['failed_stop'],
                remaining=r['failed_remaining_candidates'],equivalence=r['failed_remaining_target_equivalence_classes'],
                paid_successful_labels_to_resolve=len(r['posthoc_replay_of_already_paid_successful_labels'])) for r in rows])


if __name__=='__main__':main()
