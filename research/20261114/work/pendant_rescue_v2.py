"""H2ap gate-coverage successor, with the H2ao intervention unchanged."""
import argparse
import json
import pendant_rescue as OLD
import study as S
PP=S.HERE/'pendant-rescue-v2-protocol.json'
DEST=S.OUT/'pendant_rescue_v2'
PARENT=OLD.PARENT


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=OLD.protocol();old_path=OLD.PP;certificate=OLD.DEST/'old-chain-fixed-point-control.json';failed=OLD.DEST/'failed-gate.json';r=json.loads(certificate.read_text());assert r['status']=='passed' and r['protocol_sha256']==S.sha(old_path);assert json.loads(failed.read_text())['new_searches']==0
    p.update(id='seen-cohort-articulation-pendant-rescue-gate-coverage-successor',hypothesis_id='H2ap',frozen_utc=S.utc(),predecessor_protocol={'path':str(old_path.relative_to(S.ROOT)),'sha256':S.sha(old_path)},predecessor_gate_failure={'path':str(failed.relative_to(S.ROOT)),'sha256':S.sha(failed)},reused_fixedpoint_certificate={'path':str(certificate.relative_to(S.ROOT)),'sha256':S.sha(certificate),'protocol_sha256':S.sha(old_path)},gate_revision='Score all admitted candidates from128fixedrandom5x5masks and12explicit9x9motifs withhand-enumeratedexactcellsets/witnesses:backbone+fork/internaljunction,attachedcycle,multiplependants;fourrotations;alloriginal120boundarycases unchanged. Exactnegative-scoretie gate usesknownsyntheticunaryenergy andsameimmutable solve;not scientifictrajectory. Brutecheckallcounts/dedup/witnesses. No changes to intervention,models,readouts,selection,target,budget.',duplicate_geometry_note='A nonempty component behind one cut has singleton boundary at that cut in its original connectedcomponent; identical cellset cannot have two different cut witnesses. Dedup remains defensive and exact zero-duplicate counts are checked.')
    for n in ['pendant_rescue_v2.py','gate_pendant_rescue_v2.py','pendant_gate_fixtures.py']:p['source_sha256'][str((S.HERE/n).relative_to(S.ROOT))]=S.sha(S.HERE/n)
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.emit('pendant_rescue_v2_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for r in group.values():assert S.sha(S.ROOT/r['path'])==r['sha256']
    assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json')
    for name in ['predecessor_protocol','predecessor_gate_failure','reused_fixedpoint_certificate']:
        r=p[name];assert S.sha(S.ROOT/r['path'])==r['sha256']
    c=p['reused_fixedpoint_certificate'];r=json.loads((S.ROOT/c['path']).read_text());assert r['status']=='passed' and r['protocol_sha256']==c['protocol_sha256'];return p


def evaluate(gpu):
    p=protocol();c=p['reused_fixedpoint_certificate'];S.save(DEST/'old-chain-fixed-point-control.json',{'status':'passed','protocol_sha256':S.sha(PP),'source_certificate':c,'new_searches':0,'new_certificate_runs':0,'scope':'Exact unchanged candidate/native dependencies and models;hashbound reuse of all12288oldterminalstates.'});OLD.PP=PP;OLD.DEST=DEST;OLD.protocol=protocol;OLD.evaluate(gpu)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();freeze() if v.command=='freeze' else evaluate(v.gpu)
