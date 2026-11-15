"""H2at frozen lexical selection control of the H2as energy-ranked cycle cuts."""
import argparse
import json
import study as S
import cycle_cut_rescue_checked as ENERGY
import pendant_rescue as DRIVER
import lexical_cycle_cut_moves as B
PP=S.HERE/'lexical-cycle-cut-protocol.json'
DEST=S.OUT/'lexical_cycle_cut'
PARENT=DRIVER.PARENT


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=ENERGY.protocol();audit=ENERGY.DEST/'audit.json';a=json.loads(audit.read_text());assert a['status']=='passed' and a['targets']['joint_rescues']==21;preflight=S.OUT/'lexical_cycle_cut_preflight/numerical-gate.json';r=json.loads(preflight.read_text());assert r['status']=='passed' and r['selector_only_source_change_verified'] and r['source_sha256']==S.sha(S.HERE/'preflight_lexical_cycle_cut.py') and r['lexical_helper_sha256']==S.sha(S.HERE/'lexical_cycle_cut_moves.py')
    p.update(id='seen-cohort-lexical-control-of-learned-cycle-cut-ranking',hypothesis_id='H2at',frozen_utc=S.utc(),energy_ranking_protocol={'path':str(ENERGY.PP.relative_to(S.ROOT)),'sha256':S.sha(ENERGY.PP)},energy_ranking_audit={'path':str(audit.relative_to(S.ROOT)),'sha256':S.sha(audit)},lexical_selector_preflight={'path':str(preflight.relative_to(S.ROOT)),'sha256':S.sha(preflight)},primary={'minimum_energy_ranked_only_joint_gains':1,'maximum_lexical_only_joint_gains':0,'decision':'Finiteposthocsourcehypothesis:energyrankingmustsolveatleastoneadditionaljointinputandlosezeroinputsversuslexical. Separateendpoints;nogeneralization/confidenceclaim.'},policy='Allarchivedvetooutputs included; exactinitialsimplepathidentity. Identicaloldchain/cycle-rank/connectivity candidates,positiveacceptance,originalnative descent/caps asH2as;onlyselectorchanges tominimumlexicographicsortedcelltuple, independentofenergy. Fulloriginalcostsstillcomputedandarchivedforidenticalaudit/charging;notneededforlexicalselection. No label/shortestdistance selection.',determinism='Lexicographic minimumsortedcandidatecelltuple. Noenergyordering. Allgeometry/nativeoperations unchanged.',charging='Archivedfullvetoprefixplusselector/cache/search. Lexicalcontrolretainscostcomputationsforfullauditandmatchedtiming;notclaimingoptimizedlexicalruntime. Sharedhidden separate.')
    for name in ['lexical_cycle_cut.py','lexical_cycle_cut_moves.py','preflight_lexical_cycle_cut.py']:p['source_sha256'][str((S.HERE/name).relative_to(S.ROOT))]=S.sha(S.HERE/name)
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.emit('lexical_cycle_cut_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for r in group.values():assert S.sha(S.ROOT/r['path'])==r['sha256']
    for name in ['reused_fixedpoint_certificate','checked_source_preflight','energy_ranking_protocol','energy_ranking_audit','lexical_selector_preflight']:
        r=p[name];assert S.sha(S.ROOT/r['path'])==r['sha256']
    assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json');return p


def evaluate(gpu):
    p=protocol();ref=p['lexical_selector_preflight'];r=json.loads((S.ROOT/ref['path']).read_text());r.update(protocol_sha256=S.sha(PP),reused_preflight=ref,new_gate_runs=0);S.save(DEST/'numerical-gate.json',r);S.save(DEST/'old-chain-fixed-point-control.json',{'status':'passed','protocol_sha256':S.sha(PP),'source_certificate':p['reused_fixedpoint_certificate'],'new_searches':0});DRIVER.PP=PP;DRIVER.DEST=DEST;DRIVER.protocol=protocol;DRIVER.B=B;DRIVER.evaluate(gpu)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();freeze() if v.command=='freeze' else evaluate(v.gpu)
