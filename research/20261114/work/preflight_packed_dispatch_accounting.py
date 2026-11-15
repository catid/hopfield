"""H2bs exact endpoint boundaries and mandatory host/zero-slot cost checks."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import packed_dispatch_accounting as A


def main():
    started=time.perf_counter();cases=rejected=0
    def reject(call):
        nonlocal rejected
        try:call()
        except (ValueError,KeyError):rejected+=1
        else:raise AssertionError('Malformed cost record accepted')
    for candidate,decision in [(90.,'supported'),(math.nextafter(90.,math.inf),'falsified'),(math.nextafter(90.,0.),'supported'),(110.,'falsified')]:
        rows=[{'pair':i,'original_order':100.,'packed':candidate} for i in range(6)]
        result=A.endpoint(rows);assert result['decision']==decision and Fraction(result['exact_median_fraction'])==1-Fraction(candidate)/100;cases+=1
    rows=[{'pair':i,'original_order':100.,'packed':v} for i,v in enumerate([91.,89.,88.,85.,110.,80.])];expected=sorted(1-Fraction(r['packed'])/100 for r in rows)
    assert Fraction(A.endpoint(rows)['exact_median_fraction'])==(expected[2]+expected[3])/2;cases+=1
    reject(lambda:A.endpoint(rows[:5]));reject(lambda:A.endpoint(rows[:-1]+[rows[0]]));reject(lambda:A.endpoint([{**r,'packed':float('nan')} for r in rows]))
    keys=['input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance','BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance','endpoint_extraction_and_dispatch','BP_subset_gather']
    components={k:1. for k in keys};duration=sum(components.values())+.5
    def receipt(n):return {'mode':'endpoint_dispatch','components':components.copy(),'n':n,'actual_BP_inputs':n,'direct_LP_inputs':0,'rescue_LP_inputs':n,'actual_subset_inputs':n,'answer_seconds':duration,'unattributed_in_segment_seconds':.5}
    models=[{'seed':seed,'indices':list(range(33)),'actual_inputs':33,'slots_per_model':3,'preparation':A.segment(1.,3.),'finalization':A.segment(5.,8.)} for seed in range(3)]
    ledger=[]
    for seed in range(3):
        for slot,indices in enumerate([list(range(32)),[32],[]]):
            ledger.append({'seed':seed,'slot':slot,'zero_work':not indices,'indices':indices,'answer_timing':receipt(len(indices)) if indices else None,'materialization':A.segment(10.,11.) if indices else None,'scatter':A.segment(20.,23.) if indices else None})
    assert A.complete(10.,models,ledger)==10+3*(5+2*(duration+4));cases+=1
    bad=copy.deepcopy(ledger);bad[0]['materialization']=None;reject(lambda:A.complete(10.,models,bad))
    bad=copy.deepcopy(ledger);bad[2]['answer_timing']=receipt(1);reject(lambda:A.complete(10.,models,bad))
    bad=copy.deepcopy(ledger);bad[1]['answer_timing']['n']=32;reject(lambda:A.complete(10.,models,bad))
    bad=copy.deepcopy(ledger);bad[1]['indices']=[0];reject(lambda:A.complete(10.,models,bad))
    bad=copy.deepcopy(ledger);bad[0]['materialization']['seconds']=0;reject(lambda:A.complete(10.,models,bad))
    reject(lambda:A.complete(10.,models,ledger[:-1]));reject(lambda:A.complete(10.,models,ledger+[ledger[0]]));reject(lambda:A.complete(10.,models[:2],ledger));reject(lambda:A.segment(1.,0.));reject(lambda:A.segment(0.,float('nan')))
    dest=Path(__file__).resolve().parents[3]/'artifacts/research/20261114/work/bp_first_packed_dispatch'
    result={'status':'passed','exact_cost_cases':cases,'invalid_receipts_rejected':rejected,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str(p.relative_to(dest.parents[4])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(A.__file__)]}}
    (dest/'accounting-synthetic-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
