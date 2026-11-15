"""H2bs all-archive receipt seal before performance endpoint joins."""
import json
import study as S
DEST=S.OUT/'bp_first_packed_dispatch'


def main():
    target=DEST/'sealed-replays.json';assert not target.exists();ph=S.sha(DEST/'protocol.json');packet=S.sha(DEST/'source-packet.json');files={}
    def bind(path):files[str(path.relative_to(DEST))]=S.sha(path)
    bind(DEST/'source-packet.json')
    for gpu in [0,1]:
        for command,limit,expected in [('gate',240,packet),('timing',2400,ph)]:
            file=DEST/f'{command}-worker{gpu}.json';worker=json.loads(file.read_text());assert worker['status']=='complete' and worker['worker_wall_seconds']<=limit and worker['protocol_sha256']==expected;bind(file)
            file=DEST/f'{command}-gpu{gpu}.json';assert json.loads(file.read_text())['status']=='passed';bind(file)
        bind(DEST/f'pairs-gpu{gpu}.json')
        for mode in ['original_order','packed']:
            for kind,expected,n in [('source-gates',packet,384),('warmup',ph,96)]:
                file=DEST/kind/f'gpu{gpu}-{mode}'/'result.json';r=json.loads(file.read_text());assert r['status']=='complete' and r['protocol_sha256']==expected and r['model_inputs']==n;bind(file)
                if kind=='warmup':bind(file.parent/'source-check.json')
    for pair in range(6):
        for mode in ['original_order','packed']:
            file=DEST/'runs'/f'pair{pair}-{mode}'/'result.json';r=json.loads(file.read_text());assert r['status']=='complete' and r['protocol_sha256']==ph and r['model_inputs']==12288;bind(file)
    assert len(files)==35
    S.save(target,{'status':'complete','protocol_sha256':ph,'sealed_utc':S.utc(),'receipt_sha256':files,'endpoint_aggregates_joined':False,'fresh_scientific_inputs':0,'timed_model_pipeline_copies':147456})
    print(json.dumps({'status':'complete','receipts':len(files),'seal_sha256':S.sha(target)},indent=2))


if __name__=='__main__':main()
