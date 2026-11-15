"""H2as science freeze after complete checked source/synthetic preflight."""
import argparse
import json
import cycle_cut_rescue_v2 as PREVIOUS
import cycle_cut_rescue as DRIVER
import study as S
PP=S.HERE/'cycle-cut-rescue-checked-protocol.json'
DEST=S.OUT/'cycle_cut_rescue_checked'
PARENT=DRIVER.PARENT


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=PREVIOUS.protocol();preflight=S.OUT/'cycle_cut_preflight/numerical-gate.json';r=json.loads(preflight.read_text());assert r['status']=='passed' and r['kind']=='preflight_not_scientific_study' and r['policy_reference_protocol_sha256']==S.sha(PREVIOUS.PP);assert r['source_sha256']==S.sha(S.HERE/'preflight_cycle_cut_gate.py') and r['geometry_verifier_sha256']==S.sha(S.HERE/'verify_cycle_cut_geometry_checked.py');assert len(r['synthetic_positive_acceptance']['rows'])==4 and all(x['positive_moves']==16 and x['block_cap'] for x in r['synthetic_positive_acceptance']['rows'])
    p.update(id='seen-cohort-positive-cost-cycle-cut-checked-preflight',hypothesis_id='H2as',frozen_utc=S.utc(),superseded_unexecuted_protocol={'path':str(PREVIOUS.PP.relative_to(S.ROOT)),'sha256':S.sha(PREVIOUS.PP)},checked_source_preflight={'path':str(preflight.relative_to(S.ROOT)),'sha256':S.sha(preflight),'scope':'Completecorrectedsource/syntheticgateexecutedbeforethissciencefreeze;nostudycohortinterventions. Rawcandidatecountcheckedagainstrawlistbeforesemanticcomparison. Exactunchangedscientifichelper,models,fixturesandbudget hash-bound.'})
    for name in ['cycle_cut_rescue_checked.py','preflight_cycle_cut_gate.py','verify_cycle_cut_geometry_checked.py']:p['source_sha256'][str((S.HERE/name).relative_to(S.ROOT))]=S.sha(S.HERE/name)
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.emit('cycle_cut_rescue_checked_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for r in group.values():assert S.sha(S.ROOT/r['path'])==r['sha256']
    for name in ['predecessor_protocol','predecessor_gate_failure','reused_fixedpoint_certificate','predecessor_intervention','predecessor_audit','immediate_predecessor','immediate_gate_failure','superseded_unexecuted_protocol','checked_source_preflight']:
        r=p[name];assert S.sha(S.ROOT/r['path'])==r['sha256']
    assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json');return p


def evaluate(gpu):
    p=protocol();ref=p['checked_source_preflight'];r=json.loads((S.ROOT/ref['path']).read_text());assert r['status']=='passed';r.update(protocol_sha256=S.sha(PP),reused_checked_preflight=ref,new_gate_runs=0);S.save(DEST/'numerical-gate.json',r);DRIVER.PP=PP;DRIVER.DEST=DEST;DRIVER.protocol=protocol;DRIVER.evaluate(gpu)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();freeze() if v.command=='freeze' else evaluate(v.gpu)
