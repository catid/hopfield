"""Exact-rational bounds certify a scalar wrong-rounding lambda4 example."""
from fractions import Fraction as Q
import json
from pathlib import Path
import hashlib
import time


def log_integer_interval(n,terms=128):
    # log(n)=2 atanh((n-1)/(n+1)); positive series with geometric tail bound.
    y=Q(n-1,n+1);power=y;total=Q(0)
    for k in range(terms):
        total+=2*power/(2*k+1);power*=y*y
    remainder=2*power/((2*terms+1)*(1-y*y))
    return total,total+remainder


def main():
    started=time.perf_counter();l39,u39=log_integer_interval(39);l49,u49=log_integer_interval(49)
    assert u39<Q(37,10) and l49>Q(187,50)
    term=Q(1);exp_lower=term
    for k in range(1,9):term*=Q(3,k);exp_lower+=term
    assert exp_lower>19
    curvature_lower=Q(1600,39)-8;assert curvature_lower>0
    # Foru in[3,log49], sigmoid(u)>19/20; derivativeT'=1/2+4p(1-p)<=69/100.
    contraction_upper=Q(1,2)+4*Q(19,20)*Q(1,20);assert contraction_upper==Q(69,100)<1
    def interval(a,b):return {'lower_exact':str(a),'upper_exact':str(b),'lower_float':float(a),'upper_float':float(b)}
    result={'status':'proved_by_exact_rational_bounds','energy':'E(p)=p logp+(1-p)log(1-p)+(1/10)p+4p(1-p)','map':'T(u)=(1/2)u+(1/2)[-1/10+4(2sigmoid(u)-1)]','initial_u':3,'binary_energy_gap_E1_minus_E0_exact':'1/10','wrong_rounding_flip_E0_minus_E1_exact':'-1/10','log39':interval(l39,u39),'log49':interval(l49,u49),'derivative_at_p39_over40':interval(l39-Q(37,10),u39-Q(37,10)),'derivative_at_p49_over50':interval(l49-Q(187,50),u49-Q(187,50)),'equilibrium_probability_interval_exact':['39/40','49/50'],'curvature_lower_on_interval_exact':str(curvature_lower),'exp3_strict_lower_exact':str(exp_lower),'map_contraction_upper_on_u3_to_log49_exact':str(contraction_upper),'proof':'Exactlogbounds giveEprime(39/40)<0<Eprime(49/50), andEsecond>0 throughout. T is increasing; exp3>19 impliesT(3)>3, whilelog49>187/50 impliesT(log49)<log49. ThusT maps[3,log49] intoitself andhas Lipschitzconstant<=69/100 there. Banach yields convergencefromnativeu3 toitsuniquefixedpoint, whosep isbetween39/40and49/50. It rounds to1, whoseoriginalbinaryflip lowersenergyby1/10.','scope':'Analytic scalar counterexample, independentofmazeoutcomes. It proves binary-energy preservation and removalofnearhalfminima do not guarantee binarycertification, evenfromu3/eta.5. It is not a proof of the fullmaze failure mechanism or newmodel-inputevaluation.','seconds':time.perf_counter()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
