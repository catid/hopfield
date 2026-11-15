"""Independent score-based natural-log formula, exact ranks and calibration bins."""
import sys
sys.set_int_max_str_digits(0)
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time


def rational(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
def decode(x):return Fraction(int(x['numerator']),int(x['denominator']))


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bo';P=R/'results/research/loki/h3bo';M=R/'results/research/loki/h3bm';N=R/'results/research/loki/h3bn';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];tolerance=Decimal('1e-45')
    def check(name,value):assert value,name;checks.append(name)
    def near(name,a,b):check(name,abs(Decimal(a)-Decimal(b))<=tolerance)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');cases=read(M/'case-results.json');costs=read(N/'case-results.json');rows=read(P/'case-diagnostics.json');contexts=read(P/'case-contexts.json');stored_groups=read(P/'group-posteriors.json');toy=read(P/'synthetic-gate.json');selection=read(R/p['normalizers']['selection_path']);prior=read(N/'summary.json');scores=[]
    check('wholebudgetandlineage',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=30 and g['wall_seconds']<=60 and p['budget']['GPUs']==0 and g['new_source_draws']==g['new_oracle_labels']==0 and all(g[k]==raw[k] for k in raw if k!='utc') and p['frozen_utc']<toy['utc']<raw['utc'] and g['protocol_sha256']==sha(S/'protocol.json') and g['source_sha256']==sha(S/'study.py') and prior['audit']['all439_weighted_optimality_certificates_and_prior_witnesses_verified'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    for name in ['case-contexts','case-diagnostics','group-posteriors','calibration-bins','cost-change-strata','truth-diagnostics','synthetic-gate']:check('stage '+name,sha(P/f'{name}.json')==g[name.replace('-','_')+'_sha256'])
    check('synthetictiesemptyguard',toy['all_analytic_tests_passed'] and toy['guard_values']==['0.250000000001','0.25','0.249999999999','0.249999999998999999'] and toy['guard_expected']==['supported','inconclusive_numerical_guard','inconclusive_numerical_guard','falsified'] and toy['empty_aggregate']['cases']==0 and toy['empty_aggregate']['mean_excess_logloss_bits'] is None and toy['uniform_fixture']['diagnostic']['true_probability']==rational(Fraction(1,2)) and toy['uniform_fixture']['diagnostic']['true_rank_best']==1 and toy['uniform_fixture']['diagnostic']['true_rank_worst']==2 and not toy['uniform_fixture']['diagnostic']['MAP_lex_correct'] and toy['uniform_fixture']['diagnostic']['true_in_MAP_tie']);near('toy_uniform_excess',toy['uniform_fixture']['diagnostic']['excess_logloss_bits'],'0');near('toy_unequal_logloss',toy['unequal_fixture']['diagnostic']['posterior_logloss_bits'],'2');near('toy_unequal_excess',toy['unequal_fixture']['diagnostic']['excess_logloss_bits'],'1')
    for record in selection['candidates']:
        M12=record['complete_source_cardinality'];k=len(record['orbit_indices']);scores.append(math.comb(55,k)*math.comb(M12,8) if M12>=8 else None)
    score_strings=[str(x) if x is not None else None for x in scores];check('all1541scoreformulas',len(scores)==1541 and score_strings==read(M/'score-values8.json')==read(N/'score-values8.json'))
    groups={};calculated=[];expected_contexts=[]
    with localcontext() as context:
        context.prec=50;ln2=Decimal(2).ln()
        for case,cost in zip(cases,costs):
            cid=case['case'];truth=case['candidate'];arm=case['arms']['8'];eligible=arm['compiler_feasible'] is True;group=arm['tree_group'];check(f'case{cid}costlineage',cost['case']==cid and cost['candidate']==truth and cost['old_tree_continuation_calls']==arm['tree_calls'] and cost['old_exact']==cost['new_exact']==arm['exact_true_singleton']);entry=dict(case=cid,candidate=truth,bank=case['bank'],primary_diagnostic_eligible=eligible,context_reason='feasible_checkpoint' if eligible else 'known_depth4_infeasible' if group is not None else 'already_identified',remaining_candidates=arm['checkpoint_candidates'],parent_group=group,old_continuation_calls=arm['tree_calls'],weighted_continuation_calls=cost['new_tree_continuation_calls'],diagnostic_index=None)
            if eligible:
                candidates=arm['checkpoint_candidates'];K=len(candidates)
                if group not in groups:
                    reciprocals={c:Fraction(1,scores[c]) for c in candidates};Z=sum(reciprocals.values(),Fraction(0));probs={c:v/Z for c,v in reciprocals.items()};maxprob=max(probs.values());ties=[c for c in candidates if probs[c]==maxprob];L=math.lcm(*(scores[c] for c in candidates));weights=[L//scores[c] for c in candidates];entropy=sum((Decimal(v.numerator)/Decimal(v.denominator))*(Decimal(v.denominator).ln()-Decimal(v.numerator).ln())/ln2 for v in probs.values());record=dict(parent_group=group,candidates=candidates,weights=list(map(str,weights)),weight_sum=str(sum(weights)),MAP_ties=ties,MAP_lex_candidate=min(ties),MAP_confidence=rational(maxprob),entropy_bits=str(entropy));groups[group]=dict(record=record,probabilities=probs,Z=Z);check(f'group{group}parentweights',record['weights']==read(N/f'group{group}/weights.json')['integer_weights'])
                info=groups[group];record=info['record'];probs=info['probabilities'];Z=info['Z'];pt=probs[truth];better=sum(v>pt for v in probs.values());atleast=sum(v>=pt for v in probs.values());mid=Fraction(1+better+atleast,2)
                # Independent score-form formula: ln(d_true) + ln(sum_i 1/d_i).
                # The study instead subtracts logs of exact integer weights.
                posterior=(Decimal(scores[truth]).ln()+Decimal(Z.numerator).ln()-Decimal(Z.denominator).ln())/ln2;uniform=Decimal(K).ln()/ln2;excess=(Decimal(scores[truth]).ln()+Decimal(Z.numerator).ln()-Decimal(Z.denominator).ln()-Decimal(K).ln())/ln2;logrank=(Decimal(mid.numerator).ln()-Decimal(mid.denominator).ln())/ln2;row=dict(case=cid,candidate=truth,bank=case['bank'],parent_group=group,candidate_count=K,MAP_ties=record['MAP_ties'],MAP_lex_candidate=record['MAP_lex_candidate'],MAP_confidence=record['MAP_confidence'],entropy_bits=record['entropy_bits'],old_continuation_calls=arm['tree_calls'],weighted_continuation_calls=cost['new_tree_continuation_calls'],actual_cost_change=cost['new_tree_continuation_calls']-arm['tree_calls'],true_probability=rational(pt),true_rank_best=1+better,true_rank_worst=atleast,true_rank_midpoint=rational(mid),log2_midrank=str(logrank),posterior_logloss_bits=str(posterior),uniform_logloss_bits=str(uniform),excess_logloss_bits=str(excess),MAP_lex_correct=truth==record['MAP_lex_candidate'],true_in_MAP_tie=truth in record['MAP_ties']);saved=rows[len(calculated)]
                approximate=['entropy_bits','log2_midrank','posterior_logloss_bits','uniform_logloss_bits','excess_logloss_bits'];check(f'case{cid}exactposteriorranksandcosts',all(saved[k]==v for k,v in row.items() if k not in approximate))
                for key in approximate:near(f'case{cid}independent{key}',saved[key],row[key])
                entry['diagnostic_index']=len(calculated);calculated.append(row)
            expected_contexts.append(entry)
        check('all1067and341context',contexts==expected_contexts and len(contexts)==1408 and len(rows)==len(calculated)==1067 and len(groups)==439 and sum(r['context_reason']=='known_depth4_infeasible' for r in contexts)==54 and sum(r['context_reason']=='already_identified' for r in contexts)==287)
        check('allgroupidentities',len(stored_groups)==439 and [r['parent_group'] for r in stored_groups]==sorted(groups)==p['parent_group_ids'])
        for saved in stored_groups:
            expected=groups[saved['parent_group']]['record'];check(f'group{saved["parent_group"]}record',all(saved[k]==v for k,v in expected.items() if k!='entropy_bits'));near(f'group{saved["parent_group"]}entropy',saved['entropy_bits'],expected['entropy_bits'])
        def aggregate(selected):
            if not selected:return dict(cases=0,mean_posterior_logloss_bits=None,mean_uniform_logloss_bits=None,mean_excess_logloss_bits=None,mean_true_probability=None,mean_midrank=None,MAP_lex_accuracy=None)
            n=len(selected);return dict(cases=n,mean_posterior_logloss_bits=str(sum(Decimal(r['posterior_logloss_bits']) for r in selected)/n),mean_uniform_logloss_bits=str(sum(Decimal(r['uniform_logloss_bits']) for r in selected)/n),mean_excess_logloss_bits=str(sum(Decimal(r['excess_logloss_bits']) for r in selected)/n),mean_true_probability=rational(sum((decode(r['true_probability']) for r in selected),Fraction(0))/n),mean_midrank=rational(sum((decode(r['true_rank_midpoint']) for r in selected),Fraction(0))/n),MAP_lex_accuracy=rational(Fraction(sum(r['MAP_lex_correct'] for r in selected),n)))
        def audit_aggregate(prefix,saved,expected):
            for key,value in expected.items():
                if key.startswith('mean_') and key.endswith('_bits') and value is not None:near(prefix+key,saved[key],value)
                else:check(prefix+key,saved[key]==value)
        primary=aggregate(calculated);audit_aggregate('primary',g['primary'],primary);excess=Decimal(primary['mean_excess_logloss_bits']);decision='supported' if excess>=Decimal('0.25')+Decimal('1e-12') else 'falsified' if excess<Decimal('0.25')-Decimal('1e-12') else 'inconclusive_numerical_guard';bins=[];ECE=Fraction(0)
        for i in range(10):
            selected=[r for r in calculated if (Fraction(i,10)<=decode(r['MAP_confidence'])<Fraction(i+1,10)) or (i==9 and decode(r['MAP_confidence'])==1)];n=len(selected);confidence=sum((decode(r['MAP_confidence']) for r in selected),Fraction(0))/n if n else None;accuracy=Fraction(sum(r['MAP_lex_correct'] for r in selected),n) if n else None;gap=confidence-accuracy if n else None;contribution=Fraction(n,1067)*abs(gap) if n else Fraction(0);ECE+=contribution;bins.append(dict(bin=i,lower=rational(Fraction(i,10)),upper=rational(Fraction(i+1,10)),upper_inclusive=i==9,cases=n,mean_confidence=rational(confidence) if n else None,MAP_lex_accuracy=rational(accuracy) if n else None,overconfidence_gap=rational(gap) if n else None,ECE_contribution=rational(contribution),case_ids=[r['case'] for r in selected]))
        check('fixedexactbinsandemptybins',bins==read(P/'calibration-bins.json') and sum(r['cases'] for r in bins)==1067)
        for saved,delta in zip(read(P/'cost-change-strata.json'),range(-3,4)):
            selected=[r for r in calculated if r['actual_cost_change']==delta];audit_aggregate(f'costdelta{delta}',saved,aggregate(selected));check(f'costdelta{delta}identities',saved['actual_cost_change']==delta and saved['case_ids']==[r['case'] for r in selected])
        check('allsevenstrata',len(read(P/'cost-change-strata.json'))==7)
        truthrows=read(P/'truth-diagnostics.json');check('all22truths',len(truthrows)==len(p['truth_ids'])==22)
        for saved,truth in zip(truthrows,p['truth_ids']):check(f'truth{truth}identity',saved['candidate']==truth);audit_aggregate(f'truth{truth}',saved,aggregate([r for r in calculated if r['candidate']==truth]))
        mean_conf=sum((decode(r['MAP_confidence']) for r in calculated),Fraction(0))/1067;accuracy=Fraction(sum(r['MAP_lex_correct'] for r in calculated),1067);outcome=dict(decision=decision,threshold_bits='0.25',numerical_guard_bits='1e-12',decimal_precision=50,diagnostic_cases=1067,distinct_checkpoint_sets=439,context_cases=1408,infeasible_context_cases=54,already_identified_context_cases=287,descriptive_MAP_mean_confidence=rational(mean_conf),descriptive_MAP_lex_accuracy=rational(accuracy),descriptive_MAP_overconfidence=rational(mean_conf-accuracy),descriptive_fixed_bin_ECE=rational(ECE),descriptive_actual_cost_change=sum(r['actual_cost_change'] for r in calculated),new_source_draws=0,new_oracle_labels=0);check('guardeddecisionanddescriptiveidentities',all(g[k]==v for k,v in outcome.items()) and outcome['descriptive_actual_cost_change']==188)
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==v for k,v in outcome.items()) and s['primary']==g['primary'] and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_exact_eight_positive_score_formulas_verified=True,all1067_posteriors_and_tied_ranks_verified=True,all439_weight_vectors_and_entropy_values_verified=True,independent_score_based_natural_log_formula_precision=50,maximum_allowed_log_formula_difference='1e-45',guarded_quarter_bit_threshold_verified=True,all341_context_cases_retained_without_primary_pooling=True,all10_exact_confidence_bins_and7_cost_strata_including_empty_verified=True,all22truth_diagnostic_rows_retained=True,calibration_is_selected_examined_sample_only=True,no_new_draws_or_labels=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
