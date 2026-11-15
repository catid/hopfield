"""H2ar independent-gate metadata successor; H2aq scientific policy unchanged."""
import argparse
import json
import cycle_cut_rescue as OLD
import study as S
PP=S.HERE/'cycle-cut-rescue-v2-protocol.json'
DEST=S.OUT/'cycle_cut_rescue_v2'
PARENT=OLD.PARENT


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=OLD.protocol();failure=OLD.DEST/'failed-gate.json';assert json.loads(failure.read_text())['new_searches']==0
    p.update(id='seen-cohort-positive-cost-cycle-cut-gate-schema-successor',hypothesis_id='H2ar',frozen_utc=S.utc(),immediate_predecessor={'path':str(OLD.PP.relative_to(S.ROOT)),'sha256':S.sha(OLD.PP)},immediate_gate_failure={'path':str(failure.relative_to(S.ROOT)),'sha256':S.sha(failure)},gate_schema_revision='Removeindependentreferenceextra candidate_count metadatafieldbeforecomparingexactchaingeometrydict. Lists,ranks,policy,energychecks,positiveacceptancefixture,endpoints,budgets unchanged.')
    for name in ['cycle_cut_rescue_v2.py','gate_cycle_cut_rescue_v2.py','verify_cycle_cut_geometry_v2.py']:p['source_sha256'][str((S.HERE/name).relative_to(S.ROOT))]=S.sha(S.HERE/name)
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.emit('cycle_cut_rescue_v2_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for r in group.values():assert S.sha(S.ROOT/r['path'])==r['sha256']
    for name in ['predecessor_protocol','predecessor_gate_failure','reused_fixedpoint_certificate','predecessor_intervention','predecessor_audit','immediate_predecessor','immediate_gate_failure']:
        r=p[name];assert S.sha(S.ROOT/r['path'])==r['sha256']
    assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json');return p


def evaluate(gpu):
    OLD.PP=PP;OLD.DEST=DEST;OLD.protocol=protocol;OLD.evaluate(gpu)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();freeze() if v.command=='freeze' else evaluate(v.gpu)
