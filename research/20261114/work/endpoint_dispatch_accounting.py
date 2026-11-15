"""H2br complete answer accounting and fixed six-block dual-control costs."""
from fractions import Fraction
import math
from bp_first_geometry_accounting import batch_cost as inherited


def batch_cost(receipt):
    if receipt['mode']!='endpoint_dispatch':return inherited(receipt)
    components=receipt['components'];n=receipt['n'];bp=receipt['actual_BP_inputs'];low=receipt['direct_LP_inputs'];rescue=receipt['rescue_LP_inputs']
    if not {'endpoint_extraction_and_dispatch','BP_subset_gather'}<=components.keys():raise ValueError('Uncharged dispatcher operation')
    if not all(isinstance(v,int) and v>=0 for v in [n,bp,low,rescue]) or bp+low!=n or rescue>bp or receipt['actual_subset_inputs']!=low+rescue:raise ValueError('Inconsistent actual routing cardinalities')
    if bp==0 and any(components[k]!=0 for k in ['BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return']):raise ValueError('Empty BP subset claims execution')
    return inherited({**receipt,'mode':'BP_geometry_LP96'})


def six_block_endpoint(rows,control):
    if control not in ['LP96','BP_geometry_LP96']:raise ValueError('Unknown complete-cost control')
    if len(rows)!=6 or sorted(r['block'] for r in rows)!=list(range(6)):raise ValueError('All six fixed blocks required')
    pairs=[];fractions=[]
    for row in sorted(rows,key=lambda r:r['block']):
        a,b=row[control],row['endpoint_dispatch']
        if not all(math.isfinite(x) and x>0 for x in [a,b]):raise ValueError('Positive finite complete costs required')
        value=Fraction(1)-Fraction(b)/Fraction(a);fractions.append(value);pairs.append({'block':row['block'],'control_seconds':a,'dispatcher_seconds':b,'saving':float(value)})
    ordered=sorted(fractions);median=(ordered[2]+ordered[3])/2
    return {'control':control,'blocks':pairs,'median_saving':float(median),'exact_median_fraction':str(median),'minimum_saving':.05,'decision':'supported' if median>=Fraction(1,20) else 'falsified'}
