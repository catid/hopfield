"""Exact finite posterior ranks and fifty-digit natural-log diagnostics."""
from decimal import Decimal,localcontext
from fractions import Fraction
import math


def rational(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def distribution(candidates,scores):
    L=math.lcm(*(int(scores[c]) for c in candidates));weights=[L//int(scores[c]) for c in candidates];W=sum(weights);maximum=max(weights);ties=[c for c,w in zip(candidates,weights) if w==maximum]
    with localcontext() as context:
        context.prec=50;ln2=Decimal(2).ln();entropy=sum((Decimal(w)/Decimal(W))*(Decimal(W).ln()-Decimal(w).ln())/ln2 for w in weights)
    return dict(candidates=candidates,weights=list(map(str,weights)),weight_sum=str(W),MAP_ties=ties,MAP_lex_candidate=min(ties),MAP_confidence=rational(Fraction(maximum,W)),entropy_bits=str(entropy))


def diagnose(row,truth):
    candidates=row['candidates'];weights=list(map(int,row['weights']));W=int(row['weight_sum']);K=len(candidates);weight=weights[candidates.index(truth)];best=1+sum(w>weight for w in weights);worst=sum(w>=weight for w in weights);mid=Fraction(best+worst,2)
    with localcontext() as context:
        context.prec=50;ln2=Decimal(2).ln();posterior=(Decimal(W).ln()-Decimal(weight).ln())/ln2;uniform=Decimal(K).ln()/ln2;regret=(Decimal(W).ln()-Decimal(weight).ln()-Decimal(K).ln())/ln2;logrank=(Decimal(mid.numerator).ln()-Decimal(mid.denominator).ln())/ln2
    return dict(true_probability=rational(Fraction(weight,W)),true_rank_best=best,true_rank_worst=worst,true_rank_midpoint=rational(mid),log2_midrank=str(logrank),posterior_logloss_bits=str(posterior),uniform_logloss_bits=str(uniform),excess_logloss_bits=str(regret),MAP_lex_correct=truth==row['MAP_lex_candidate'],true_in_MAP_tie=truth in row['MAP_ties'])


def decision(excess):
    x=Decimal(excess);threshold=Decimal('0.25');guard=Decimal('1e-12')
    if x>=threshold+guard:return 'supported'
    if x<threshold-guard:return 'falsified'
    return 'inconclusive_numerical_guard'


def aggregate(rows):
    if not rows:return dict(cases=0,mean_posterior_logloss_bits=None,mean_uniform_logloss_bits=None,mean_excess_logloss_bits=None,mean_true_probability=None,mean_midrank=None,MAP_lex_accuracy=None)
    with localcontext() as context:
        context.prec=50;n=Decimal(len(rows));mean=lambda name:sum(Decimal(r[name]) for r in rows)/n;prob=sum(Fraction(int(r['true_probability']['numerator']),int(r['true_probability']['denominator'])) for r in rows)/len(rows);rank=sum(Fraction(int(r['true_rank_midpoint']['numerator']),int(r['true_rank_midpoint']['denominator'])) for r in rows)/len(rows);accuracy=Fraction(sum(r['MAP_lex_correct'] for r in rows),len(rows))
        return dict(cases=len(rows),mean_posterior_logloss_bits=str(mean('posterior_logloss_bits')),mean_uniform_logloss_bits=str(mean('uniform_logloss_bits')),mean_excess_logloss_bits=str(mean('excess_logloss_bits')),mean_true_probability=rational(prob),mean_midrank=rational(rank),MAP_lex_accuracy=rational(accuracy))
