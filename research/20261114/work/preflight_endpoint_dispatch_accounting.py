"""H2br six-block exact median and complete dispatch-cost contract checks."""
from fractions import Fraction
import json
import math
import study as S
import endpoint_dispatch_accounting as A


def fails(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid cost receipt accepted')


def main():
    cases=0
    for candidate,decision in [(95.,'supported'),(math.nextafter(95.,math.inf),'falsified'),(math.nextafter(95.,0.),'supported'),(110.,'falsified')]:
        rows=[{'block':i,'LP96':100.,'BP_geometry_LP96':100.,'endpoint_dispatch':candidate} for i in range(6)]
        for control in ['LP96','BP_geometry_LP96']:
            result=A.six_block_endpoint(rows,control);assert result['decision']==decision and Fraction(result['exact_median_fraction'])==1-Fraction(candidate)/100;cases+=1
    rows=[{'block':i,'LP96':100.,'BP_geometry_LP96':110.,'endpoint_dispatch':value} for i,value in enumerate([91.,99.,96.,93.,101.,90.])]
    for control in ['LP96','BP_geometry_LP96']:
        expected=sorted(1-Fraction(r['endpoint_dispatch'])/Fraction(r[control]) for r in rows);assert Fraction(A.six_block_endpoint(rows,control)['exact_median_fraction'])==(expected[2]+expected[3])/2;cases+=1
    fails(lambda:A.six_block_endpoint(rows[:5],'LP96'));fails(lambda:A.six_block_endpoint(rows[:-1]+[rows[0]],'LP96'));fails(lambda:A.six_block_endpoint(rows,'BP_AdamW'))
    keys=['input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance','BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance','endpoint_extraction_and_dispatch','BP_subset_gather']
    for bp,low,rescue in [(0,32,0),(32,0,0),(32,0,32),(18,14,3)]:
        components={k:1. for k in keys}
        if bp==0:
            for k in ['BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return']:components[k]=0.
        if low+rescue==0:
            for k in ['LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return']:components[k]=0.
        receipt={'mode':'endpoint_dispatch','components':components,'n':32,'actual_BP_inputs':bp,'direct_LP_inputs':low,'rescue_LP_inputs':rescue,'actual_subset_inputs':low+rescue,'answer_seconds':sum(components.values())+.5,'unattributed_in_segment_seconds':.5}
        assert A.batch_cost(receipt)==receipt['answer_seconds'];cases+=1
        fails(lambda:A.batch_cost({**receipt,'actual_subset_inputs':33}))
        fails(lambda:A.batch_cost({**receipt,'components':{k:v for k,v in components.items() if k!='endpoint_extraction_and_dispatch'}}))
    out={'status':'passed','exact_cost_cases':cases,'invalid_receipts_rejected':11,'fresh_inputs_drawn':0,'source_sha256':{name:S.sha(S.HERE/name) for name in ['endpoint_dispatch_accounting.py','preflight_endpoint_dispatch_accounting.py','bp_first_geometry_accounting.py']}}
    S.save(S.OUT/'bp_first_endpoint_dispatch/accounting-synthetic-gate.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
