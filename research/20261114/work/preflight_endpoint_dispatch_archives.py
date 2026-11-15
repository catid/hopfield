"""Independently audit the already sealed H2br source-compatibility archives."""
import json
import time
import study as S
from audit_endpoint_dispatch_sources import source_check
from audit_endpoint_dispatch_costs import verify_costs
DEST=S.OUT/'bp_first_endpoint_dispatch'


def main():
    target=DEST/'source-archive-preflight.json';assert not target.exists()
    started=time.perf_counter();packet=DEST/'source-compatibility-packet.json';p=json.loads(packet.read_text());ph=S.sha(packet);p['compatibility_packet_sha256']=ph;cohorts=[]
    for gpu in [0,1]:
        for source in p['sources']:
            for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
                path=DEST/'source-compatibility-attempt1'/'source-gates'/f'gpu{gpu}-{source["name"]}-{mode}'
                r=json.loads((path/'result.json').read_text());assert r['status']=='complete' and r['protocol_sha256']==ph and r['model_inputs']==192
                r.update(kind='gate',source_name=source['name'],directory=str(path.relative_to(S.ROOT)));cohorts.append(r)
    results=[source_check((S.ROOT,DEST,p,name,cohorts),gate_only=True) for name in ['corner','uniform']]
    result={'status':'passed','source_model_pipeline_copies_checked':sum(r['source_gate_and_warmup_model_copies_checked'] for r in results),'sources':results,'packet_sha256':ph,'fresh_inputs_drawn':0,'seconds':time.perf_counter()-started,'source_sha256':{name:S.sha(S.HERE/name) for name in ['preflight_endpoint_dispatch_archives.py','audit_endpoint_dispatch_sources.py','audit_endpoint_dispatch_replays.py','verify_endpoint_dispatch_trace.py','audit_endpoint_dispatch_states.py']}}
    S.save(target,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
