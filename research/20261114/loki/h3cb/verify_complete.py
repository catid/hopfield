"""Full independent fresh-source, two-arm paid-query and proof replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from common import load
from audit_io import AuditIO
from audit_queries import PaidQueries
from audit_sources import verify as verify_sources
from audit_runtime import verify as verify_runtime
from audit_trees import TreeAudit
from audit_arm import verify as verify_arm
from audit_namespace_gate import verify as verify_namespace
from audit_recovery import pair as verify_pair,endpoint as independent_endpoint
from audit_recovery_namespace import verify as verify_recovery_gate
from audit_fallback_gate import verify as verify_fallback_gate
from audit_costs import GraphAudit, choose, exact_guard

from audit_cap_gate import verify as verify_cap
from role_gate import run as verify_roles


def verify(root, archive_dir):
    started = time.perf_counter()
    R, P = Path(root).resolve(), Path(archive_dir).resolve()
    S = R/'research/20261114/loki/h3cb'
    checks = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    io = AuditIO(P, S, check)
    read, sha = io.read, io.sha
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    p, g, raw, prep = read(S/'protocol.json'), read(P/'capacity-gate.json'), read(P/'study-result.json'), read(P/'preparation.json')
    check('complete_inside_unchanged_aggregate_caps', g['status'] == raw['status'] == 'complete_verified' and
          g['primary_decision_emitted'] and not g['internally_complete_but_not_accepted'] and
          g['study_result_sha256']==sha(P/'study-result.json') and
          not g['monitor_expired'] and g['aggregate_cpu_seconds'] <= 600 and g['wall_seconds'] <= 900 and
          g['global_memo_states'] <= 2000000 and g['global_split_evaluations'] <= 50000000 and
          g['fallback_candidate_board_cells']<=500000000 and
          g['full_monitor_process_lifetime_CPU_charged'] and g['initial_source_hashing_inside_wall_and_CPU_budget'] and
          g['subprocess_inclusive_CPU_accounting'] and all(g[k] == v for k, v in raw.items() if k != 'utc'))
    check('declared_cases_and_budgets', p['budget']['planned_cases'] == 1522 and
          p['budget']['membership_calls'] == 24352 and p['budget']['membership_calls_per_arm'] == 12176 and
          p['seed_bases'] == dict(permutation=2027700000,source=2027800000))
    for category in ['source_sha256', 'source_data_sha256']:
        for relative, digest in p[category].items():
            check('frozen_identity '+relative, sha(R/relative) == digest)
    for name in ['case-results','case-statuses','source-banks','truth-schedule','preparation']+[a+'/'+n for a in ['B','E'] for n in ['arm-results','groups','counterfactual-groups']]:
        check('complete_artifact_seal '+name, g[name.replace('-', '_').replace('/','_')+'_sha256'] == sha(P/(name+'.json')))
    check('protocol_and_scientific_source_seals', g['protocol_sha256'] == prep['protocol_sha256'] == sha(S/'protocol.json') and
          g['source_sha256'] == sha(S/'study.py') and prep['preflight_sha256'] == sha(S/'preflight.json'))
    for name in ['population-parity', 'ledger-gate', 'synthetic-gate', 'restriction-gate',
                 'pruning-gate', 'cache-gate', 'cost-gate', 'baseline-gate', 'wrapper-gate', 'namespace-gate', 'cap-gate','fallback-gate','recovery-gate']:
        check('runtime_gate_seal '+name, prep[name+'_sha256'] == sha(P/(name+'.json')))
    check('source_gate_seals', prep['source_prior_gate_sha256'] == sha(P/'source-gates/prior-gate.json') and
          prep['fitted_weight_gate_sha256'] == sha(P/'source-gates/fitted-weight-gate.json') and
          prep['two_score_gate_sha256'] == sha(P/'source-gates/two-score-gate.json') and
          prep['source_gate'] == read(P/'source-gates/seed-audit.json') and prep['all_runtime_gates_before_prospective_draws'])
    bits = np.load(R/'results/research/loki/h3bc/predictions-bits.npy')
    boards = np.load(R/'results/research/loki/h3bc/ordinary13.npy')
    pred = np.unpackbits(bits, axis=1, count=len(boards), bitorder='little').astype(bool)
    scores = read(P/'score-values8.json')
    check('complete_universe_and_scores_pinned', pred.shape == (1541, 73712) and
          scores == read(R/'results/research/loki/h3bm/score-values8.json'))
    ledger_bytes = io.ledger()
    chain = load(S.parent/'h3bj/verify_public.py', 'h3cb_independent_chain').chain
    events = chain(ledger_bytes)
    check('exact_raw_append_only_ledger_identity', hashlib.sha256(ledger_bytes).hexdigest() == g['ledger_sha256'] and
          len(events) == g['ledger_records'] and events[-1]['sha256'] == g['ledger_last_record_sha256'] and
          p['frozen_utc'] < prep['utc'] < events[0]['utc'] < raw['utc'])
    gate = read(P/'ledger-gate.json')
    synthetic_bytes = io.data(P/'synthetic-ledger.jsonl')
    synthetic_events = chain(synthetic_bytes)
    check('synthetic_durability_gate_identity', sha(P/'synthetic-ledger.jsonl') == gate['synthetic_ledger_sha256'] and
          len(synthetic_events) == 16 and gate['oracle_callbacks_observed_durable_query_before_label'] == 8 and
          all(gate['tamper_cases_rejected'].values()) and gate['source_sha256'] == sha(S/'ledger_gate.py') and
          gate['ledger_source_sha256'] == sha(S/'ledger.py'))
    for i, (query_path, answer_path) in enumerate(p['ledger_gate']['archival_query_answer_paths']):
        oldq, olda = read(R/query_path), read(R/answer_path)
        query, answer = synthetic_events[2*i:2*i+2]
        receipt = gate['paired_format_parity'][i]
        check('synthetic_archived_selection_and_answer', query['data'] == dict(index=i,
              request=oldq['request'], selection=oldq['selection'], board_zero_based=oldq['board_zero_based']) and
              answer['data'] == dict(index=i, label=olda['label'], candidates_before=olda['candidates_before'],
              candidates_after=olda['candidates_after'], query_sha256=query['sha256']) and
              receipt['archival_query_sha256'] == sha(R/query_path) and receipt['archival_answer_sha256'] == sha(R/answer_path) and
              receipt['new_query_record_sha256'] == query['sha256'] and receipt['new_answer_record_sha256'] == answer['sha256'] and
              query['monotonic_ns'] < receipt['oracle_callback_monotonic_ns'] < answer['monotonic_ns'])
    namespace=read(P/'namespace-gate.json')
    check('namespace_gate_identical_receipts',namespace==read(P/'namespace-gate/summary.json'))
    ns,nt=namespace['synthetic_memo_states'],namespace['synthetic_work_steps']
    capped=verify_cap(P/'cap-gate',io,check)
    recovered=verify_recovery_gate(P/'recovery-gate',io,check)
    fallback=verify_fallback_gate(P/'fallback-gate',io,check)
    check('runtime_recovery_and_fallback_receipt_seals',read(P/'recovery-gate.json')==read(P/'recovery-gate/summary.json') and
          read(P/'fallback-gate.json')==read(P/'fallback-gate/summary.json'))
    baseprep=dict(prep,runtime_gate_memo_states=prep['runtime_gate_memo_states']-ns-capped['synthetic_memo_states']-recovered['synthetic_memo_states'],
                  runtime_gate_work_steps=prep['runtime_gate_work_steps']-nt-capped['synthetic_work_steps']-recovered['synthetic_work_steps'])
    gates=verify_runtime(P,S,baseprep,check,io,canonical)
    namespace_audit=verify_namespace(P/'namespace-gate',io,check,ns,nt)
    check('all_namespace_fixture_receipts',all(namespace[k]==v for k,v in namespace_audit.items()) and
          namespace['actual_oracle_calls']==namespace['fresh_truth_draws']==namespace['fresh_source_draws']==0)
    roles=verify_roles(P/'namespace-gate',ns,nt,archive_root=P)
    check('independent_score_role_mutation_receipts',roles==namespace['score_role_mutations'])
    gate_states,gate_work=gates['memo_states']+ns+capped['synthetic_memo_states']+recovered['synthetic_memo_states'],gates['work_steps']+nt+capped['synthetic_work_steps']+recovered['synthetic_work_steps']
    gate_cells=fallback['charged_candidate_board_cells']+recovered['charged_candidate_board_cells']
    gate_requests=fallback['charged_selection_requests']+recovered['charged_selection_requests']
    check('all_runtime_producer_scan_work_charged',prep['runtime_gate_fallback_cells']==gate_cells and prep['runtime_gate_fallback_requests']==gate_requests)
    sources=read(P/'source-banks.json')
    paired=read(P/'case-results.json')
    rows={a:read(P/a/'arm-results.json') for a in ['B','E']}
    check('all1024_pairs_and_complete_statuses',len(paired)==len(sources)==1522 and
          read(P/'case-statuses.json')==[dict(case=i,source='complete',B='complete',E='complete') for i in range(1522)])
    ledger=PaidQueries(events,pred,boards,check,arms=('B','E'))
    check('study_start_seal',ledger.take('study_start')['data']==dict(protocol_sha256=sha(S/'protocol.json'),
          preparation_sha256=sha(P/'preparation.json')))
    source_audit=verify_sources(R,P,p,sources,ledger,read,sha,check)
    score_gate=read(P/'source-gates/two-score-gate.json')
    arm_audits={}
    resources={}
    resource_names=['tree_memo_states','compiler_work_steps','huffman_merge_operations',
                    'canonical_computation_reuse_hits','counterfactual_memo_states',
                    'counterfactual_MAP_partitions','distinct_compiled_inputs']
    for arm in ['B','E']:
        arm_scores=read(P/arm/'tree-scores.json')
        selector_scores=read(P/arm/'selector-scores.json')
        check('distinct_arm_score_and_prediction_identity',arm_scores==score_gate['T_scores'] and selector_scores==score_gate['H_scores'] and
              sha(P/arm/'predictions-bits.npy')==sha(R/'results/research/loki/h3bc/predictions-bits.npy'))
        arm_audits[arm]=verify_arm(arm,P/arm,S,sources,rows[arm],pred,boards,selector_scores,arm_scores,ledger,io,check)
        resources[arm]={k:arm_audits[arm][k] for k in resource_names}
    for h,u,pair in zip(rows['B'],rows['E'],paired):
        expected=verify_pair(h,u,check)
        check('all_paired_costs_and_invariants_retained',pair==expected==ledger.take('paired_case_result')['data'])
    endpoint=independent_endpoint(paired)
    check('every_paid_call_in_joint_denominator',endpoint['B_calls']==ledger.calls['B']<=12176 and
          endpoint['E_calls']==ledger.calls['E']<=12176 and endpoint['actual_membership_calls']<=24352)
    outcome=dict(**endpoint,cases=1522,MAP_calls={a:sum(r['map_calls'] for r in rows[a]) for a in ['B','E']},
                 tree_calls={a:sum(r['tree_calls'] for r in rows[a]) for a in ['B','E']},
                 truth_permutation_draws=1,individual_truth_draws=0,new_source_draws=1522,source_positive_union_appearances=12176,positive_arm_appearances=24352,
                 distinct_N12_boards=source_audit['distinct_N12_boards'],
                 source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),
                 source_cases_matching_archived_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),
                 independently_paid_cross_arm_reacquisitions=sum(r['independently_paid_cross_arm_reacquisitions'] for r in paired),
                 truth_frequencies=[dict(candidate=r['candidate'],eligible=r['eligible'],sampling_integer_weight=r['sampling_integer_weight'],
                                        cases=source_audit['truth_counts'].get(r['candidate'],0)) for r in p['truths']],
                 handoff_MAP_call_histogram={a:{str(i):sum(r['handoff_MAP_calls']==i for r in rows[a]) for i in range(5)} for a in ['B','E']},
                 cases_without_tree_handoff={a:sum(r['handoff_MAP_calls'] is None for r in rows[a]) for a in ['B','E']},
                 resources_by_arm=resources,runtime_gate_memo_states=gate_states,runtime_gate_work_steps=gate_work,
                 global_memo_states=gate_states+sum(r['tree_memo_states']+r['counterfactual_memo_states'] for r in resources.values()),
                 global_split_evaluations=gate_work+sum(r['compiler_work_steps']+r['counterfactual_MAP_partitions'] for r in resources.values()),
                 fallback_calls={a:sum(r['fallback']['additional_calls'] for r in rows[a]) for a in ['B','E']},
                 runtime_gate_fallback_cells=gate_cells,runtime_gate_fallback_requests=gate_requests,
                 fallback_candidate_board_cells=gate_cells+sum(r['fallback']['candidate_board_cells'] for arm in rows.values() for r in arm),
                 fallback_selection_requests=gate_requests+sum(r['fallback']['selection_requests'] for arm in rows.values() for r in arm),
                 fallback_resources_by_arm={a:dict(candidate_board_cells=sum(r['fallback']['candidate_board_cells'] for r in rr),
                     selection_requests=sum(r['fallback']['selection_requests'] for r in rr),CPU_seconds=sum(r['fallback']['CPU_seconds'] for r in rr),
                     wall_seconds=sum(r['fallback']['wall_seconds'] for r in rr),maximum_active_set=max([len(r['original_policy_result']['remaining_candidates']) for r in rr if r['fallback']['entered']]+[0])) for a,rr in rows.items()})
    check('joint_primary_and_all_resources_reconstructed',ledger.take('study_complete')['data']==outcome and
          all(g[k]==v for k,v in outcome.items()) and ledger.at==len(events))
    storage=io.finish()
    return dict(status='passed_complete_fresh_balanced_recovery_two_arm_audit',checks=checks,cases=1522,
                decision=endpoint['decision'],primary=endpoint['primary'],exact_transitions=endpoint['exact_transitions'],
                B_paid_queries=ledger.calls['B'],E_paid_queries=ledger.calls['E'],actual_paid_queries=endpoint['actual_membership_calls'],
                all_fresh_prior_tickets_and_source_indices_verified=True,all_paired_exact_flags_verified=True,
                every_failure_and_duplicate_retained=True,separate_fitted_scores_and_paid_arms_verified=True,
                all_deeper_and_counterfactual_proofs_verified=True,arm_audits=arm_audits,runtime_gate_audit=gates,
                namespace_gate_audit=namespace_audit,cap_gate_audit=capped,score_role_gate=roles,recovery_gate_audit=recovered,fallback_gate_audit=fallback,
                charged_fallback_candidate_board_cells=outcome['fallback_candidate_board_cells'],charged_memo_states=outcome['global_memo_states'],
                charged_work_steps=outcome['global_split_evaluations'],storage=storage,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--archive-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.archive_dir)))
