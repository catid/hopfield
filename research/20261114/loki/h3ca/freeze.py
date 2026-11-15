"""Reviewable dual-score protocol; actual freeze follows coordinator review."""
import argparse
from common import ROOT,SRC,OUT,read,save,sha,now


def main(draft=False):
    assert not (SRC/'protocol.json').exists() and not (OUT/'preparation.json').exists()
    previous=read(SRC.parent/'h3bz/protocol.json')
    pre=read(SRC/'preflight.json');seed=read(SRC/'seed-audit.json')
    audit=read(SRC/'component-audit.json');monitor=read(SRC/'monitor-gate.json')
    sg=read(SRC/'prior-gate.json');source_audit=read(SRC/'source-audit-gate.json');preparation=read(SRC/'prepare-gate.json');archive=read(SRC/'archive-gate.json')
    assert archive['status']=='passed_fixed_full_runtime_archive_replay'
    assert preparation['status']=='passed_full_preparation_with_RNG_forbidden' and preparation['RNG_API_attempts']==0
    assert pre['status']=='passed_prefreeze_synthetic_gates'
    assert pre['independent_inline']['status']=='passed_independent_synthetic_artifact_audit'
    assert pre['independent_baseline']['status']=='passed_independent_baseline_artifact_audit'
    assert pre['wrapper_gate']['status']=='passed_independently_verified_baseline_wrapper_gate'
    assert pre['namespace_gate']['status']=='passed_common_selector_tree_weight_namespace_and_payment_gate'
    assert pre['cap_gate']['status']=='passed_weight_sensitive_cap_and_no_fallback_gate'
    assert len(pre['namespace_gate']['score_role_mutations']['mutations'])==10
    assert audit['cap_gate_audit']['status']=='passed_independent_weight_sensitive_cap_artifact_audit'
    assert pre['new_truth_draws']==pre['new_source_draws']==pre['actual_oracle_calls']==0
    assert source_audit['status']=='passed_independent_two_score_sampling_support_gate'
    assert source_audit['RNG_API_attempts']==0 and len(source_audit['semantic_mutations_rejected'])==11
    assert all(source_audit['semantic_mutations_rejected'].values())
    assert audit['status']=='passed_independent_complete_verifier_components'
    assert all(audit['paid_query_binding_gate']['all10_rehashed_semantic_mutations_rejected'].values())
    assert not seed['explicit_collisions'] and not seed['expanded_collisions']
    assert monitor['status']=='passed_monitor_accounting_probes'
    assert {r['mode'] for r in monitor['probes']}=={'initial_hash','descendants','cpu_cap','wall_cap','late_cpu_cap','late_wall_cap'}
    for row in monitor['probes']:
        if row['mode'].startswith('late_'):
            result=row['receipt'];assert result['status']=='failed_or_incomplete' and result['internally_complete_but_not_accepted']
            assert not result['primary_decision_emitted'] and 'decision' not in result and 'primary' not in result
    assert monitor['gate_source_sha256']==sha(SRC/'gate.py') and monitor['probe_source_sha256']==sha(SRC/'monitor_gate.py')
    for receipt,key in [(pre,'scientific_source_sha256'),(source_audit,'source_sha256'),(source_audit,'gate_sha256'),
                        (audit,'source_sha256'),(audit['paid_query_binding_gate'],'source_sha256'),
                        (pre['namespace_gate'],'source_sha256'),(preparation,'source_sha256'),(archive,'source_sha256')]:
        for name,digest in receipt[key].items():assert sha(SRC/name)==digest,name
    core=['compiler.py','selector.py','weights.py','ledger.py',
          'baseline.py','baseline_compiler.py','baseline_firstlex.py','memoized_compiler.py','prior.py']
    for name in core:assert sha(SRC/name)==sha(SRC.parent/'h3bz'/name),name
    inputs=set(previous['source_data_sha256'])|set(seed['historical_protocol_sha256'])
    inputs.update(['research/20261114/loki/h3bz/protocol.json','results/research/loki/h3bz/summary.json',
                   'results/research/loki/h3bz/source-banks.json'])
    for name in ['DESIGN.md','proposal.json','proposed-seed-audit.json','prefreeze-attempts.json',
                 'SAME_MAP_LEMMA.md','BOUND_LEMMA.md','DEPTH_LEMMA.md','GUARD_LEMMA.md','LEMMA.md','preflight.json',
                 'prior-gate.json','fitted-weight-gate.json','two-score-gate.json','seed-audit.json',
                 'source-audit-gate.json','component-audit.json','monitor-gate.json','prepare-gate.json','archive-gate.json','pair-cache-benchmark.json']:
        inputs.add(str((SRC/name).relative_to(ROOT)))
    sources=sorted(SRC.glob('*.py'))+[SRC.parent/'h3at/source.py']
    p=dict(id='H3ca-common-hierarchical-MAP-matched-tree-weights',frozen_utc=None if draft else now(),
        hypothesis='Across all1024 fresh uniform-truth pairs, matched-tree T uses at least2% fewer actual membership calls than hierarchical-tree H: H>0 and100*T<=98*H. Common MAP routing, handoff, exact flags and matched expected-cost inequalities are mandatory correctness invariants.',
        primary=dict(multiplier=100,comparator_multiplier=98,threshold_percent=2,
                     exact_condition='Identical paired exact flags are a prospective correctness invariant, not a cost tradeoff',
                     unequal_exact_flags='Whole-study correctness failure; no primary decision',
                     complete_failure='A complete valid cohort failing100*T<=98*H falsifies the cost hypothesis',
                     zero_comparator='inconclusive_zero_comparator',missing_or_capped='whole_study_incomplete_no_primary_decision'),
        candidate_prior=previous['candidate_prior'],normalizers=previous['normalizers'],
        truths=sg['prior_rows'],fitted_prior_rows=previous['fitted_prior_rows'],
        arms=dict(H=dict(selector_score='C(55,k)*C(M,8)',tree_score='C(55,k)*C(M,8)',posterior='Hierarchical tree objective'),
                  T=dict(selector_score='C(55,k)*C(M,8)',tree_score='C(M,8)',posterior='Matched uniform-prior tree objective')),
        source_support='All1541 candidates are filtered only by compatibility with8 distinct source positives. Both score tables useNone exactly for19models withM<8; those models cannot contain8 distinct positives. Never restrict inference by true identity, true orbit count or observed cohort frequency.',
        seed_bases=dict(truth=2027500000,source=2027600000),historical_protocol_sha256=seed['historical_protocol_sha256'],
        truth_sampling='For i0..1023,default_rng(2027500000+i).integers(1522) and exact cumulative uniform eligible tickets in candidate-ID order. No pilot, redraw, tuning or sample extension.',
        source_sampling='For i0..1023,independent default_rng(2027600000+i).choice(M,8,replace=False). Durably log all1024 pairs before any actual membership query. One common source bank per pair;8192 new positive appearances,16384 arm appearances. Retain every duplicate, historical match, failure and zero-frequency row.',
        routing=previous['routing'],compilation=previous['compilation'],counterfactual=previous['counterfactual'],
        weighting='Both actual and counterfactual MAP paths use hierarchical selector-scores.json in both arms. Compilers and their exact guard use per-arm tree-scores.json. Both role hashes/paths are explicit in tree and counterfactual input/cache identities. Reciprocal positive integer weights retain100000-bit cap.',
        cost_guard='Same hierarchical MAP continuation in both arms; terminal depth4 compiler uses that arm tree weights. All candidates must identify, then exact new tree cost must not exceed continuation. Admissible implies acceptance; MAP4 must tie. Any uncertified or capped computation is whole-study incomplete, never infeasible or a fallback.',
        expectation_scope='Under completed exact plans, same MAP scores imply common prefixes/handoffs/exact flags. At each common handoff the matched posterior expected tree cost T<=H. Weights can change search/pruning/cap exposure. No realized per-case dominance, realized2% guarantee or population2% claim.',
        execution_order=['H','T'],execution='All runtime gates before draws, all1024sources before queries, allHcases beforeT. Separate selector subprocesses, role-specific score files, compiler/counterfactual caches and proof namespaces. Every actual label separately paid; no cross-arm proof/computation/label reuse. Reacquisition annotations occur after selection.',
        original_baseline='Hypothetical continuation inside each arm guard only. No separately executed original-baseline arm and no measured baseline-call claim.',
        correctness='Independently validate all complete proofs, actual/counterfactual selector-score paths, role identities, same MAP prefixes and common handoff sets/depths, identical exact flags and matched expected tree cost T<=H. Any violation invalidates the whole-study primary decision while retaining all evidence.',
        ledger=previous['ledger'],ledger_gate=previous['ledger_gate'],
        gates='Retain all inherited exact compiler/pruning/bounds/cache/original-wrapper/ledger gates. Add8 common-selector fixtures and10 role/cache mutations, full independent paired replay, fixed cap asymmetry where23work finishes uniform but interrupts skewed after an incumbent, forced state/integer caps and actual ArmRunner no-fallback/no-paid-label checks. Source-only1522tickets/1541scores and11 semantic mutations. CPU/wall monitor probes include complete-then-late-cap failures with no final primary. All runtime gate work charged.',
        seed_audit_scope=seed['scope'],
        budget=dict(aggregate_CPU_seconds=600,wall_seconds=900,GPUs=0,max_memo_states_total=2000000,
                    max_split_evaluations_total=50000000,max_integer_bits=100000,membership_calls=16384,
                    membership_calls_per_arm=8192,planned_cases=1024,new_positive_appearances=8192,positive_arm_appearances=16384,
                    enforcement='Aggregate both actual arms,all counterfactuals and runtime gates,selector descendants,preparation,imports/sourcehashing,serialization and full monitor lifetime. Charge each memo state,Huffman merge,partition and counterfactual MAP step.8actualqueries percase/arm. Any cap/error leaves whole study incomplete; preserve all statuses/proofs/paid labels. No continuation,retry,cap extension or partial endpoint. Postrun independent audit/export separately timed.'),
        interpretation='Finite1024-pair cost test with prospective exact equality invariants. Report all exact outcomes and every failure/duplicate. Descriptive orbit strata are not additional endpoints. BP AdamW unmeasured and first in method tables. Supplied finite class,predictions and noiseless membership oracle do not establish arbitrary-rule or positive-only learning,optimizer superiority or uniform population gains.',
        publication='Complete cases,querychain,source/protocol identities and full audit receipts public. Large proof payloads outsideGit with every raw/compressed identity; portable replay explicitly omits full optimality certificates.',
        source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},
        source_data_sha256={rel:sha(ROOT/rel) for rel in sorted(inputs)},target_data_sha256={})
    target=SRC/('proposal-protocol.json' if draft else 'protocol.json')
    save(target,p);print('DRAFT' if draft else 'FROZEN',sha(target),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');main(parser.parse_args().draft)
