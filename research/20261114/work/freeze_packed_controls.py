"""H2bt final protocol after both unchanged reviewed source gates pass."""
import json
import shutil
import study as S
DEST=S.OUT/'bp_first_packed_controls'


def main():
    target=S.HERE/'bp-first-packed-controls-protocol.json';assert not target.exists() and not (DEST/'protocol.json').exists()
    packet=DEST/'source-packet.json';p=json.loads(packet.read_text());assert p['status']=='frozen_before_changed_source_gates'
    for key in ['source_sha256','source_evidence_sha256']:
        for path,digest in p[key].items():assert S.sha(S.ROOT/path)==digest,path
    gate_hashes={};extra=[packet]
    for gpu in [0,1]:
        file=DEST/f'gate-gpu{gpu}.json';r=json.loads(file.read_text());assert r['status']=='passed' and r['protocol_sha256']==S.sha(packet) and r['source_model_inputs_per_arm']==384
        assert len(r['checks'])==3 and all(c['status']=='passed' and c['model_pipeline_copies']==384 for c in r['checks']);gate_hashes[str(gpu)]=S.sha(file);extra.append(file)
        file=DEST/f'gate-worker{gpu}.json';r=json.loads(file.read_text());assert r['status']=='complete' and r['worker_wall_seconds']<=240 and r['peak_cpu_rss_kib']<=8*1024**2;extra.append(file)
        for mode in ['LP96','BP_geometry_LP96','packed_dispatch']:extra.append(DEST/'source-gates'/f'gpu{gpu}-{mode}'/'result.json')
    extra += [DEST/'live-gate.json',DEST/'launcher-gate.json']
    p.update(status='frozen_before_packed_timing',frozen_utc=S.utc(),source_packet_sha256=S.sha(packet),gate_receipts_sha256=gate_hashes)
    p['source_evidence_sha256'].update({str(path.relative_to(S.ROOT)):S.sha(path) for path in extra})
    S.save(target,p);shutil.copyfile(target,DEST/'protocol.json');print(json.dumps({'protocol_sha256':S.sha(target),'source_bindings':len(p['source_sha256']),'evidence_bindings':len(p['source_evidence_sha256'])},indent=2))


if __name__=='__main__':main()
