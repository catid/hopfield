"""H2br seal every fixed block, warmup and source gate before fresh outcomes."""
import json
import study as S
DEST=S.OUT/'bp_first_endpoint_dispatch'
MODES=['LP96','BP_geometry_LP96','endpoint_dispatch']


def main():
    target=DEST/'sealed-replays.json';assert not target.exists()
    ph=S.sha(DEST/'protocol.json');p=json.loads((DEST/'protocol.json').read_text());packet_hash=S.sha(DEST/'source-compatibility-packet.json');assert p['compatibility_packet_sha256']==packet_hash
    files={}
    def bind(path):files[str(path.relative_to(DEST))]=S.sha(path)
    bind(DEST/'source-compatibility-packet.json')
    for gpu in [0,1]:
        for kind,base,limit,expected in [('gate',DEST/'source-compatibility-attempt1',240,packet_hash),('timing',DEST,3000,ph)]:
            path=base/f'{kind}-worker{gpu}.json';r=json.loads(path.read_text());assert r['status']=='complete' and r['protocol_sha256']==expected and r['worker_wall_seconds']<=limit;bind(path)
        for name in [f'timing-gpu{gpu}.json',f'blocks-gpu{gpu}.json']:bind(DEST/name)
        bind(DEST/'source-compatibility-attempt1'/f'gate-gpu{gpu}.json')
        for source in p['sources']:
            for mode in MODES:
                for kind,base,n,expected in [('gate',DEST/'source-compatibility-attempt1'/'source-gates',192,packet_hash),('warmup',DEST/'warmup',96,ph)]:
                    path=base/f'gpu{gpu}-{source["name"]}-{mode}'/'result.json';r=json.loads(path.read_text());assert r['status']=='complete' and r['protocol_sha256']==expected and r['model_inputs']==n and r['source_gate'];bind(path)
    for block in range(6):
        for mode in MODES:
            path=DEST/'runs'/f'block{block}-{mode}'/'result.json';r=json.loads(path.read_text());assert r['status']=='complete' and r['protocol_sha256']==ph and r['model_inputs']==12288 and not r['source_gate'];bind(path)
    assert len(files)==53
    inputs={}
    for name in ['datasets.json','generation-worker.json','independent-input-audit.json','input-audit-worker.json']:
        path=DEST/name;r=json.loads(path.read_text());assert r['status']=='complete' if name.endswith('worker.json') else r['protocol_sha256']==ph;inputs[name]=S.sha(path)
    result={'status':'complete','protocol_sha256':ph,'sealed_utc':S.utc(),'cohort_and_worker_receipt_sha256':files,'fresh_input_receipt_sha256':inputs,'timing_blocks':6,'actual_modes':MODES,'model_inputs_per_timed_arm':12288,'labels_or_shortestness_outcomes_joined':False}
    S.save(target,result);print(json.dumps({'status':'complete','sealed_sha256':S.sha(target),'receipts':len(files)},indent=2))


if __name__=='__main__':main()
