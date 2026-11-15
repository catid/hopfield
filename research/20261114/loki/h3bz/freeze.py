"""Reviewable dual-score protocol; actual freeze follows coordinator review."""
import argparse
from common import ROOT,SRC,OUT,read,save,sha,now


def main(draft=False):
    assert not (SRC/'protocol.json').exists() and not (OUT/'preparation.json').exists()
    previous=read(SRC.parent/'h3by/protocol.json')
    pre=read(SRC/'preflight.json');seed=read(SRC/'seed-audit.json')
    audit=read(SRC/'component-audit.json');monitor=read(SRC/'monitor-gate.json')
    sg=read(SRC/'prior-gate.json');source_audit=read(SRC/'source-audit-gate.json')
    assert pre['status']=='passed_prefreeze_synthetic_gates'
    assert pre['independent_inline']['status']=='passed_independent_synthetic_artifact_audit'
    assert pre['independent_baseline']['status']=='passed_independent_baseline_artifact_audit'
    assert pre['wrapper_gate']['status']=='passed_independently_verified_baseline_wrapper_gate'
    assert pre['namespace_gate']['status']=='passed_two_arm_namespace_disagreement_and_payment_gate'
    assert pre['new_truth_draws']==pre['new_source_draws']==pre['actual_oracle_calls']==0
    assert source_audit['status']=='passed_independent_two_score_sampling_support_gate'
    assert source_audit['RNG_API_attempts']==0 and len(source_audit['semantic_mutations_rejected'])==10
    assert all(source_audit['semantic_mutations_rejected'].values())
    assert audit['status']=='passed_independent_complete_verifier_components'
    assert all(audit['paid_query_binding_gate']['all10_rehashed_semantic_mutations_rejected'].values())
    assert not seed['explicit_collisions'] and not seed['expanded_collisions']
    assert monitor['status']=='passed_monitor_accounting_probes'
    assert monitor['gate_source_sha256']==sha(SRC/'gate.py') and monitor['probe_source_sha256']==sha(SRC/'monitor_gate.py')
    for receipt,key in [(pre,'scientific_source_sha256'),(source_audit,'source_sha256'),(source_audit,'gate_sha256'),
                        (audit,'source_sha256'),(audit['paid_query_binding_gate'],'source_sha256'),
                        (pre['namespace_gate'],'source_sha256')]:
        for name,digest in receipt[key].items():assert sha(SRC/name)==digest,name
    core=['compiler.py','policy.py','counterfactual.py','selector.py','weights.py','ledger.py',
          'baseline.py','baseline_compiler.py','baseline_firstlex.py','memoized_compiler.py','gate.py','monitor_gate.py','prior.py']
    for name in core:assert sha(SRC/name)==sha(SRC.parent/'h3by'/name),name
    inputs=set(previous['source_data_sha256'])|set(seed['historical_protocol_sha256'])
    inputs.update(['research/20261114/loki/h3by/protocol.json','results/research/loki/h3by/summary.json',
                   'results/research/loki/h3by/source-banks.json'])
    for name in ['DESIGN.md','proposal.json','proposed-seed-audit.json','prefreeze-attempts.json',
                 'BOUND_LEMMA.md','DEPTH_LEMMA.md','GUARD_LEMMA.md','LEMMA.md','preflight.json',
                 'prior-gate.json','fitted-weight-gate.json','two-score-gate.json','seed-audit.json',
                 'source-audit-gate.json','component-audit.json','monitor-gate.json']:
        inputs.add(str((SRC/name).relative_to(ROOT)))
    sources=sorted(SRC.glob('*.py'))+[SRC.parent/'h3at/source.py']
    p=dict(id='H3bz-uniform-fitted-vs-hierarchical-fitted-deeper-policies',frozen_utc=None if draft else now(),
        hypothesis='On all1024 fresh uniform-truth pairs, U uses at least2% fewer actual membership calls than H and loses no paired exact identification that H achieved. Sole joint finite-cohort endpoint:100*U<=98*H and every H_exact implies U_exact.',
        primary=dict(multiplier=100,comparator_multiplier=98,threshold_percent=2,
                     exact_condition='For every case H_exact implies U_exact; U-only exact is allowed',
                     unequal_exact_flags='Retained scientific outcomes, not a correctness error',
                     complete_failure='Either joint condition false falsifies the joint hypothesis',
                     zero_comparator='inconclusive_zero_comparator',missing_or_capped='whole_study_incomplete_no_primary_decision'),
        candidate_prior=previous['candidate_prior'],normalizers=previous['normalizers'],
        truths=sg['prior_rows'],fitted_prior_rows=previous['fitted_prior_rows'],
        arms=dict(H=dict(score='C(55,k)*C(M,8)',posterior='Exact reciprocal score; original hierarchical fitted prior'),
                  U=dict(score='C(M,8)',posterior='Exact reciprocal score; matched uniform-model fitted prior')),
        source_support='All1541 candidates are filtered only by compatibility with8 distinct source positives. Both score tables useNone exactly for19models withM<8; those models cannot contain8 distinct positives. Never restrict inference by true identity, true orbit count or observed cohort frequency.',
        seed_bases=dict(truth=2027300000,source=2027400000),historical_protocol_sha256=seed['historical_protocol_sha256'],
        truth_sampling='For i0..1023,default_rng(2027300000+i).integers(1522) and exact cumulative uniform eligible tickets in candidate-ID order. No pilot, redraw, tuning or sample extension.',
        source_sampling='For i0..1023,independent default_rng(2027400000+i).choice(M,8,replace=False). Durably log all1024 pairs before any actual membership query. One common source bank per pair;8192 new positive appearances,16384 arm appearances. Retain every duplicate, historical match, failure and zero-frequency row.',
        routing=previous['routing'],compilation=previous['compilation'],counterfactual=previous['counterfactual'],
        weighting='Both arms use exact reciprocal-score integer weights with100000-bit cap; only the specified H/U score formula changes. Same supplied73712-board N13 universe and lexicographic ties.',
        cost_guard='Each arm independently evaluates every candidate under its own fitted-score continuation of remaining4-m MAP calls then depth4 tree; all branches must identify and exact weighted new-tree cost must not exceed that continuation. Equality accepted; any false exact branch vetoes. An infeasible terminal tree makes zero hypothetical additional calls and exact=False.',
        expectation_scope='U guard expectations use its matched uniform posterior; H guard expectations use the hierarchical fitted posterior. No theorem orders these routed policies against each other, forces cross-arm exact equality or establishes a population2% gain.',
        execution_order=['H','U'],execution='Allruntime gates before draws,all1024sources before queries,allHcases beforeU. Both actual arms are deeper policies with separate selector subprocesses, score files, tree and counterfactual cache objects and full proof namespaces. Every case starts source-only with empty query history. No cross-arm computation or label reuse. Same-case query annotations occur after selection and never supply a free label.',
        original_baseline='Hypothetical continuation inside each arm guard only. No separately executed original-baseline arm and no measured baseline-call claim.',
        correctness='Validate each route, full source compatibility, fitted score, all-candidate guard, exact flag, certificate and paid call. Cross-arm exact disagreement is part of the joint scientific endpoint, not an assertion failure.',
        ledger=previous['ledger'],ledger_gate=previous['ledger_gate'],
        gates='Retain allfixed compiler/pruning/bounds/cache/original-baseline-wrapper/ledger gates. Add source-only all1522tickets/all1541score/support verification and10 exact-check semantic mutations. Fixed20-model one-hot namespace fixture producesH-only andU-only successes plus shared-board paid reacquisitions and same-arm cache repeats. Independently replay every synthetic paid label and full proof; all runtime fixture work charged.',
        seed_audit_scope=seed['scope'],
        budget=dict(aggregate_CPU_seconds=600,wall_seconds=900,GPUs=0,max_memo_states_total=2000000,
                    max_split_evaluations_total=50000000,max_integer_bits=100000,membership_calls=16384,
                    membership_calls_per_arm=8192,planned_cases=1024,new_positive_appearances=8192,positive_arm_appearances=16384,
                    enforcement='Aggregate both actual arms,all counterfactuals and runtime gates,selector descendants,preparation,imports/sourcehashing,serialization and full monitor lifetime. Charge each memo state,Huffman merge,partition and counterfactual MAP step.8actualqueries percase/arm. Any cap/error leaves whole study incomplete; preserve all statuses/proofs/paid labels. No continuation,retry,cap extension or partial endpoint. Postrun independent audit/export separately timed.'),
        interpretation='Finite1024-pair joint test. Report allfour exact transitions and every failure/duplicate. Descriptive orbit strata are not additional endpoints. BP AdamW unmeasured and first in method tables. Supplied finite class,predictions and noiseless membership oracle do not establish arbitrary-rule or positive-only learning,optimizer superiority or uniform population gains.',
        publication='Complete cases,querychain,source/protocol identities and full audit receipts public. Large proof payloads outsideGit with every raw/compressed identity; portable replay explicitly omits full optimality certificates.',
        source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},
        source_data_sha256={rel:sha(ROOT/rel) for rel in sorted(inputs)},target_data_sha256={})
    target=SRC/('proposal-protocol.json' if draft else 'protocol.json')
    save(target,p);print('DRAFT' if draft else 'FROZEN',sha(target),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');main(parser.parse_args().draft)
