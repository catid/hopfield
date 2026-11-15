"""Independent H2br actual source-cost/barrier and exact decision preflight."""
import json
import math
import time
import study as S
from audit_endpoint_dispatch_costs import verify_costs,cost_decisions
from endpoint_dispatch_accounting import six_block_endpoint
DEST=S.OUT/'bp_first_endpoint_dispatch'


def main():
    target=DEST/'source-cost-preflight.json';assert not target.exists();started=time.perf_counter();packet=DEST/'source-compatibility-packet.json';p=json.loads(packet.read_text());ph=S.sha(packet);cohorts=[]
    for gpu in [0,1]:
        for source in p['sources']:
            for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
                path=DEST/'source-compatibility-attempt1'/'source-gates'/f'gpu{gpu}-{source["name"]}-{mode}';r=json.loads((path/'result.json').read_text());r.update(kind='gate',directory=str(path.relative_to(S.ROOT)));cohorts.append(r)
    checked=verify_costs(S.ROOT,DEST,ph,cohorts,source_only=True)
    cases=0
    for candidate in [95.,math.nextafter(95.,math.inf),math.nextafter(95.,0.),110.]:
        rows=[{'block':i,'LP96':100.,'BP_geometry_LP96':110.,'endpoint_dispatch':candidate} for i in range(6)]
        fake=[{'kind':'timing','pair':r['block'],'mode':mode,'complete_answer_seconds':r[mode]} for r in rows for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']]
        independent=cost_decisions(fake)
        for mode in ['LP96','BP_geometry_LP96']:
            actual=six_block_endpoint(rows,mode);assert independent[mode]==actual;cases+=1
    result={'status':'passed','source_cohorts_checked':checked['source_cohorts_checked'],'actual_source_answer_intervals_checked':sum(len(c['rows']) for c in cohorts),'all_source_barrier_events_checked':sum(len(c['host_barrier']['events']) for c in cohorts),'independent_exact_cost_decisions_checked':cases,'fresh_inputs_drawn':0,'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in ['preflight_endpoint_dispatch_cost_archives.py','audit_endpoint_dispatch_costs.py','endpoint_dispatch_accounting.py']}}
    S.save(target,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
