"""Guarded selected-checkpoint posterior log-score and rank diagnostics."""
from decimal import Decimal,localcontext
from fractions import Fraction
import math
import resource
import signal
import time
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from metrics import distribution,diagnose,aggregate,decision,rational
from synthetic_gate import run as synthetic_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def decode(x):return Fraction(int(x['numerator']),int(x['denominator']))


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();diagnostics=[]
    def expired(*_):raise TimeoutError('H3bo30CPU60wall diagnostic incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        synthetic_gate();M=ROOT/'results/research/loki/h3bm';N=ROOT/'results/research/loki/h3bn';cases=read(M/'case-results.json');costs=read(N/'case-results.json');oldsummary=read(M/'summary.json');prior=read(N/'summary.json');selection=read(ROOT/p['normalizers']['selection_path']);scores=[]
        assert oldsummary['audit']['all1541_source_target_graph_filters_and_both_likelihood_tables_verified'] and prior['audit']['all439_weighted_optimality_certificates_and_prior_witnesses_verified'] and prior['audit']['all5577_inherited_MAP_decisions_replayed']
        for row in selection['candidates']:
            M12=row['complete_source_cardinality'];k=len(row['orbit_indices']);scores.append(str(math.comb(55,k)*math.comb(M12,8)) if M12>=8 else None)
        assert scores==read(M/'score-values8.json')==read(N/'score-values8.json');distributions={};contexts=[]
        for case,cost in zip(cases,costs):
            cid=case['case'];truth=case['candidate'];arm=case['arms']['8'];assert cost['case']==cid and cost['candidate']==truth and cost['old_tree_continuation_calls']==arm['tree_calls'] and cost['old_exact']==cost['new_exact']==arm['exact_true_singleton'];eligible=arm['compiler_feasible'] is True;group=arm['tree_group'];context=dict(case=cid,candidate=truth,bank=case['bank'],primary_diagnostic_eligible=eligible,context_reason='feasible_checkpoint' if eligible else 'known_depth4_infeasible' if group is not None else 'already_identified',remaining_candidates=arm['checkpoint_candidates'],parent_group=group,old_continuation_calls=arm['tree_calls'],weighted_continuation_calls=cost['new_tree_continuation_calls'],diagnostic_index=None)
            if eligible:
                candidates=arm['checkpoint_candidates']
                if group not in distributions:
                    record=distribution(candidates,scores);weights=read(N/f'group{group}/weights.json');assert record['weights']==weights['integer_weights'];distributions[group]=dict(parent_group=group,**record)
                record=distributions[group];diagnostic=dict(case=cid,candidate=truth,bank=case['bank'],parent_group=group,candidate_count=len(candidates),MAP_ties=record['MAP_ties'],MAP_lex_candidate=record['MAP_lex_candidate'],MAP_confidence=record['MAP_confidence'],entropy_bits=record['entropy_bits'],old_continuation_calls=arm['tree_calls'],weighted_continuation_calls=cost['new_tree_continuation_calls'],actual_cost_change=cost['new_tree_continuation_calls']-arm['tree_calls'],**diagnose(record,truth));context['diagnostic_index']=len(diagnostics);diagnostics.append(diagnostic)
            contexts.append(context)
        assert len(contexts)==1408 and len(diagnostics)==1067 and len(distributions)==439 and sum(r['context_reason']=='known_depth4_infeasible' for r in contexts)==54 and sum(r['context_reason']=='already_identified' for r in contexts)==287;save(OUT/'case-contexts.json',contexts);save(OUT/'case-diagnostics.json',diagnostics);save(OUT/'group-posteriors.json',[distributions[i] for i in sorted(distributions)]);primary=aggregate(diagnostics);bins=[];ECE=Fraction(0)
        for index in range(10):
            rows=[r for r in diagnostics if min(9,(10*int(r['MAP_confidence']['numerator']))//int(r['MAP_confidence']['denominator']))==index];count=len(rows);confidence=sum((decode(r['MAP_confidence']) for r in rows),Fraction(0))/count if count else None;accuracy=Fraction(sum(r['MAP_lex_correct'] for r in rows),count) if count else None;gap=confidence-accuracy if count else None;contribution=Fraction(count,1067)*abs(gap) if count else Fraction(0);ECE+=contribution;bins.append(dict(bin=index,lower=rational(Fraction(index,10)),upper=rational(Fraction(index+1,10)),upper_inclusive=index==9,cases=count,mean_confidence=rational(confidence) if count else None,MAP_lex_accuracy=rational(accuracy) if count else None,overconfidence_gap=rational(gap) if count else None,ECE_contribution=rational(contribution),case_ids=[r['case'] for r in rows]))
        cost_strata=[dict(actual_cost_change=delta,**aggregate([r for r in diagnostics if r['actual_cost_change']==delta]),case_ids=[r['case'] for r in diagnostics if r['actual_cost_change']==delta]) for delta in range(-3,4)];truth_results=[dict(candidate=truth,**aggregate([r for r in diagnostics if r['candidate']==truth])) for truth in p['truth_ids']];mean_confidence=sum((decode(r['MAP_confidence']) for r in diagnostics),Fraction(0))/1067;accuracy=Fraction(sum(r['MAP_lex_correct'] for r in diagnostics),1067);save(OUT/'calibration-bins.json',bins);save(OUT/'cost-change-strata.json',cost_strata);save(OUT/'truth-diagnostics.json',truth_results);outcome=dict(status='complete_verified',utc=now(),decision=decision(primary['mean_excess_logloss_bits']),primary=primary,threshold_bits='0.25',numerical_guard_bits='1e-12',decimal_precision=50,diagnostic_cases=1067,distinct_checkpoint_sets=439,context_cases=1408,infeasible_context_cases=54,already_identified_context_cases=287,descriptive_MAP_mean_confidence=rational(mean_confidence),descriptive_MAP_lex_accuracy=rational(accuracy),descriptive_MAP_overconfidence=rational(mean_confidence-accuracy),descriptive_fixed_bin_ECE=rational(ECE),descriptive_actual_cost_change=sum(r['actual_cost_change'] for r in diagnostics),new_source_draws=0,new_oracle_labels=0,case_contexts_sha256=sha(OUT/'case-contexts.json'),case_diagnostics_sha256=sha(OUT/'case-diagnostics.json'),group_posteriors_sha256=sha(OUT/'group-posteriors.json'),calibration_bins_sha256=sha(OUT/'calibration-bins.json'),cost_change_strata_sha256=sha(OUT/'cost-change-strata.json'),truth_diagnostics_sha256=sha(OUT/'truth-diagnostics.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__));save(OUT/'study-result.json',outcome);print('DIAGNOSTIC',outcome['decision'],'mean_excess_bits',primary['mean_excess_logloss_bits'],'cases1067',flush=True)
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_diagnostics=diagnostics,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
