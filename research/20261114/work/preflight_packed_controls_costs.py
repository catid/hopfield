"""H2bt exact dual endpoints, six permutations and original source proof parity."""
from fractions import Fraction
import itertools
import json
import math
import time
import study as S
from endpoint_dispatch_accounting import six_block_endpoint


def main():
    started=time.perf_counter();p=json.loads((S.HERE/'bp-first-packed-controls-proposal.json').read_text());schedule=p['timing_schedule'];assert {tuple(r['order']) for r in schedule}==set(itertools.permutations(['LP96','BP_geometry_LP96','packed_dispatch']))
    for gpu in [0,1]:
        rows=[r for r in schedule if r['gpu']==gpu]
        for ordinal in range(3):assert {r['order'][ordinal] for r in rows}=={'LP96','BP_geometry_LP96','packed_dispatch'}
    cases=rejected=0
    for candidate,decision in [(95.,'supported'),(math.nextafter(95.,math.inf),'falsified'),(math.nextafter(95.,0.),'supported'),(110.,'falsified')]:
        rows=[{'block':i,'LP96':100.,'BP_geometry_LP96':100.,'endpoint_dispatch':candidate} for i in range(6)]
        for control in ['LP96','BP_geometry_LP96']:
            r=six_block_endpoint(rows,control);assert r['decision']==decision and Fraction(r['exact_median_fraction'])==1-Fraction(candidate)/100;cases+=1
    rows=[{'block':i,'LP96':100.,'BP_geometry_LP96':90.,'endpoint_dispatch':v} for i,v in enumerate([91.,94.,95.,96.,92.,90.])]
    assert six_block_endpoint(rows,'LP96')['decision']=='supported' and six_block_endpoint(rows,'BP_geometry_LP96')['decision']=='falsified';cases+=2
    for invalid in [rows[:5],rows[:-1]+[rows[0]],[{**r,'endpoint_dispatch':float('nan')} for r in rows]]:
        try:six_block_endpoint(invalid,'LP96')
        except ValueError:rejected+=1
        else:raise AssertionError('Incomplete/invalid six-block cost accepted')
    names=['preflight_packed_controls_costs.py','endpoint_dispatch_accounting.py','audit_packed_controls.py'];r={'status':'passed','exact_cost_cases':cases,'invalid_receipts_rejected':rejected,'balanced_permutations_checked':6,'separate_endpoint_disagreement_retained':True,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names}}
    S.save(S.OUT/'bp_first_packed_controls/cost-synthetic-gate.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
