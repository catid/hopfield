"""Portable sources, per-arm scores, paid labels and cost-endpoint replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from common import load
from audit_queries import PaidQueries
from audit_sources import verify as verify_sources
from audit_costs import choose
from audit_namespace_gate import independent_endpoint
from verify_complete import verify as verify_complete


def verify(root):
    R=Path(root).resolve();S=R/'research/20261114/loki/h3ca';P=R/'results/research/loki/h3ca'
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    s,p,g=read(P/'summary.json'),read(S/'protocol.json'),read(P/'capacity-gate.json');checks=0
    def check(name,value):
        nonlocal checks
        assert value,name
        checks+=1
    check('validated_complete_summary',s['status']=='validated' and s['cases']==1024 and s['protocol_sha256']==sha(S/'protocol.json') and
          s['verifier_source_sha256']==sha(S/'verify_public.py') and s['analyze_source_sha256']==sha(S/'analyze.py'))
    for key in ['source_sha256','source_data_sha256']:
        for rel,digest in p[key].items():check('frozen_identity '+rel,sha(R/rel)==digest)
    check('authoritative_final_capacity_receipt',g['status']=='complete_verified' and g['primary_decision_emitted'] and
          not g['internally_complete_but_not_accepted'] and not g['monitor_expired'] and not g['final_budget_exceeded'] and
          g['study_result_sha256']==sha(P/'study-result.json') and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and
          g['global_memo_states']<=2000000 and g['global_split_evaluations']<=50000000 and
          all(g[k]==v for k,v in read(P/'study-result.json').items() if k!='utc'))
    for rel,digest in s['public_sha256'].items():check('public_metadata '+rel,sha(P/rel)==digest)
    for rel,digest in s['audit_dependency_sha256'].items():check('audit_source '+rel,sha(R/rel)==digest)
    boundary,manifest=read(P/'archive-boundary.json'),read(P/'proof-export.json')
    check('external_proof_boundary',not boundary['full_proof_payloads_in_git'] and boundary['full_verification_requires_external_archive'] and
          boundary['proof_manifest_sha256']==sha(P/'proof-export.json') and not any(P.rglob('proof.json')) and not any(P.rglob('proof.json.gz')) and
          len(manifest['proofs'])==manifest['proof_files']==boundary['omitted_compressed_proof_files'] and
          sum(r['raw_bytes'] for r in manifest['proofs'])==boundary['omitted_raw_proof_bytes'] and
          sum(r['compressed_bytes'] for r in manifest['proofs'])==boundary['omitted_compressed_proof_bytes'])
    raw=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes())
    chain=load(S.parent/'h3bj/verify_public.py','h3ca_portable_chain').chain;events=chain(raw)
    receipt=read(P/'ledger-export.json')
    check('exact_paid_ledger',hashlib.sha256(raw).hexdigest()==receipt['raw_sha256']==s['ledger_sha256'] and len(raw)==receipt['raw_bytes'] and
          receipt['gzip_sha256']==sha(P/'query-ledger.jsonl.gz') and len(events)==s['ledger_records'])
    bits=np.load(R/'results/research/loki/h3bc/predictions-bits.npy');boards=np.load(R/'results/research/loki/h3bc/ordinary13.npy')
    pred=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool)
    sources=read(P/'source-banks.json');replay=PaidQueries(events,pred,boards,check,arms=('H','T'))
    check('study_start',replay.take('study_start')['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json')))
    source_audit=verify_sources(R,P,p,sources,replay,read,sha,check)
    score_gate=read(P/'source-gates/two-score-gate.json')
    scores={a:read(P/a/'tree-scores.json') for a in ['H','T']}
    selectors={a:read(P/a/'selector-scores.json') for a in ['H','T']}
    results={a:read(P/a/'arm-results.json') for a in ['H','T']}
    for arm in ['H','T']:
        check('per_arm_exact_score_and_prediction_inputs',scores[arm]==score_gate[arm+'_scores'] and selectors[arm]==score_gate['H_scores']==score_gate['selector_scores'][arm] and
              np.array_equal(np.load(P/arm/'predictions-bits.npy'),bits))
        check('arm_namespace_bound',replay.take('arm_namespace_bound')['data']==dict(arm=arm,relative_folder=arm,
              selector_scores_path='selector-scores.json',tree_scores_path='tree-scores.json',
              selector_scores_sha256=sha(P/arm/'selector-scores.json'),tree_scores_sha256=sha(P/arm/'tree-scores.json'),
              initial_tree_cache_entries=0,initial_counterfactual_cache_entries=0,selector_process_fresh=True))
        check('full_arm_cases_in_order',len(results[arm])==1024 and [r['case'] for r in results[arm]]==list(range(1024)))
        for case,stored in enumerate(results[arm]):
            source=sources[case];active,queried,maps,trees=list(source['initial_candidates']),[],[],[]
            check('source_only_arm_initial_state',replay.take('arm_source_bound')['data']==dict(case=case,arm=arm,candidate=source['candidate'],
                  source_record_sha256=source['source_record_sha256'],initial_candidates=active,inherited_queries=0))
            while replay.events[replay.at]['kind']!='arm_case_result':
                event=replay.events[replay.at];kind,data=event['kind'],event['data']
                if kind=='query':
                    request,selection,phase=data['request'],data['selection'],data['phase']
                    if phase=='map':
                        check('per_arm_MAP_selection',not trees and len(maps)<4 and selection==choose(pred,selectors[arm],active,queried) and
                              request==dict(bits_path='predictions-bits.npy',scores_path='selector-scores.json',board_count=len(boards),candidates=active,queried=queried))
                    else:
                        check('tree_label_and_bound_scope',phase=='tree' and len(trees)<8-len(maps) and selection==dict(query_index=selection['query_index'],
                              candidate_predictions={str(c):bool(pred[c,selection['query_index']]) for c in active}))
                    active,step=replay.ask(case,source['candidate'],arm,phase,active,queried,request,selection)
                    (maps if phase=='map' else trees).append(step);queried=queried+[step['query_index']]
                else:
                    check('routing_namespace_and_declared_kind',kind in ['handoff_ineligible_candidate_count','online_compiled_tree_bound',
                          'posterior_cost_guard_bound','no_informative_query'] and data['arm']==arm and data['case']==case)
                    replay.take(kind)
            check('complete_paid_arm_record',replay.take('arm_case_result')['data']==stored and stored['arm']==arm and stored['case']==case and
                  stored['map_steps']==maps and stored['tree_steps']==trees and stored['remaining_candidates']==active and stored['membership_calls']==len(queried)<=8 and
                  stored['map_calls']==len(maps) and stored['tree_calls']==len(trees) and stored['exact_true_singleton']==(active==[source['candidate']]) and
                  stored['independently_paid_cross_arm_reacquisitions']==sum(bool(x['previous_same_case_query_sequences']) for x in maps+trees))
        check('complete_arm_before_next_namespace',replay.take('all_arm_executions_complete')['data']==dict(arm=arm,cases=1024,paid_calls=replay.calls[arm],
              arm_results_sha256=sha(P/arm/'arm-results.json'),groups_sha256=sha(P/arm/'groups.json'),counterfactual_registry_sha256=sha(P/arm/'counterfactual-groups.json')))
    pairs=read(P/'case-results.json')
    check('all_pairs_retained',len(pairs)==1024)
    for h,u,pair in zip(results['H'],results['T'],pairs):
        check('all_exact_transitions_and_costs',pair==dict(case=h['case'],candidate=h['candidate'],H_calls=h['membership_calls'],T_calls=u['membership_calls'],
              H_exact=h['exact_true_singleton'],T_exact=u['exact_true_singleton'],all_actual_calls=h['membership_calls']+u['membership_calls'],
              independently_paid_cross_arm_reacquisitions=u['independently_paid_cross_arm_reacquisitions'],
              common_prefix_handoff_and_exact_verified=True,matched_expected_tree_costs=pair['matched_expected_tree_costs'])==replay.take('paired_case_result')['data'])
        check('common_MAP_prefix_and_handoff', all(h[k]==u[k] for k in ['initial_candidates','handoff_MAP_calls','handoff_candidates','handoff_depth','map_calls','exact_true_singleton']) and
              [[v[k] for k in ['step','query_index','label','candidates_before','candidates_after']] for v in h['map_steps']]==
              [[v[k] for k in ['step','query_index','label','candidates_before','candidates_after']] for v in u['map_steps']] and
              (h['terminal_no_information'] is None)==(u['terminal_no_information'] is None))
        expected=pair['matched_expected_tree_costs']
        if h['handoff_MAP_calls'] is None:
            check('common_no_handoff',expected is None and h['remaining_candidates']==u['remaining_candidates'])
        else:
            from fractions import Fraction
            check('sealed_matched_expectation_receipt',expected['candidates']==h['handoff_candidates'] and expected['nonincrease'] and
                  Fraction(expected['T_expected'])<=Fraction(expected['H_expected']) and
                  Fraction(expected['H_expected'])==Fraction(int(expected['H_weighted_cost']),int(expected['matched_weight_sum'])) and
                  Fraction(expected['T_expected'])==Fraction(int(expected['T_weighted_cost']),int(expected['matched_weight_sum'])))
    endpoint=independent_endpoint(pairs)
    outcome=replay.take('study_complete')['data']
    check('primary_from_all1024_pairs',all(s[k]==v==outcome[k] for k,v in endpoint.items()) and replay.at==len(events) and
          endpoint['H_calls']==replay.calls['H'] and endpoint['T_calls']==replay.calls['T'])
    check('full_complete_statuses',read(P/'case-statuses.json')==[dict(case=i,source='complete',H='complete',T='complete') for i in range(1024)])
    check('zero_frequency_models_and_query_composition',s['truth_frequencies']==[dict(candidate=r['candidate'],eligible=r['eligible'],
              sampling_integer_weight=r['sampling_integer_weight'],cases=source_audit['truth_counts'].get(r['candidate'],0)) for r in p['truths']] and
          s['MAP_calls']=={a:sum(r['map_calls'] for r in results[a]) for a in ['H','T']} and
          s['tree_calls']=={a:sum(r['tree_calls'] for r in results[a]) for a in ['H','T']})
    strata=[]
    for k in [0,1,2]:
        subset=[r for r in pairs if len(p['truths'][r['candidate']]['orbits'])==k]
        strata.append(dict(true_orbit_count=k,cases=len(subset),H_calls=sum(r['H_calls'] for r in subset),T_calls=sum(r['T_calls'] for r in subset),
            H_exact=sum(r['H_exact'] for r in subset),T_exact=sum(r['T_exact'] for r in subset),
            H_only_exact=sum(r['H_exact'] and not r['T_exact'] for r in subset),T_only_exact=sum(r['T_exact'] and not r['H_exact'] for r in subset)))
    check('descriptive_strata_only',s['descriptive_by_true_orbit_count']==strata)
    from describe import describe
    check('descriptive_sealed_transcript_reconstruction',s['descriptive']==describe(pairs,results,p))
    full=read(P/'full-archive-audit.json')
    check('full_audit_receipt',full['scientific_audit']==s['audit'] and full['scientific_audit']['status']=='passed_complete_fresh_two_score_two_arm_audit' and
          full['scientific_audit']['actual_paid_queries']==endpoint['actual_membership_calls'] and full['protocol_sha256']==sha(S/'protocol.json') and
          full['proof_manifest_sha256']==sha(P/'proof-export.json'))
    for rel,digest in full['audit_source_sha256'].items():check('full_audit_source_identity',sha(R/rel)==digest)
    plot=read(P/'plot-provenance.json')
    for rel,digest in plot['input_sha256'].items():check('plot_input',sha(R/rel)==digest)
    for rel,digest in plot['output_sha256'].items():check('plot_output',sha(P/rel)==digest)
    return dict(status='passed_portable_two_score_sources_paid_queries_and_cost_endpoint',checks=checks,cases=1024,
        decision=endpoint['decision'],H_paid_queries=replay.calls['H'],T_paid_queries=replay.calls['T'],exact_transitions=endpoint['exact_transitions'],
        full_optimality_certificates_replayed=False,full_counterfactual_MAP_and_tree_proofs_replayed=False,
        expected_cost_receipts_recomputed_from_leaf_depths=False,limitation='Full tree/counterfactual optimality and expectation-from-leaf-depth replay require the retained external archive; portable replay checks hash-bound expectation receipts, actual MAP decisions, paid labels and common handoffs')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);parser.add_argument('--archive-dir',type=Path)
    args=parser.parse_args();result=dict(portable=verify(args.root))
    if args.archive_dir is not None:result['external_full_audit']=verify_complete(args.root,args.archive_dir)
    print(json.dumps(result))
