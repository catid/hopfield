"""H2aq source-cohort positive-cost cycle-cut acceptance experiment."""
import argparse
import hashlib
import json
import pendant_rescue as OLD
import pendant_rescue_v2 as P
import cycle_cut_moves as B
import study as S
PP=S.HERE/'cycle-cut-rescue-protocol.json'
DEST=S.OUT/'cycle_cut_rescue'
PARENT=OLD.PARENT


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=P.protocol();certificate=p['reused_fixedpoint_certificate'];prior=P.DEST/'audit.json';assert json.loads(prior.read_text())['targets']['joint_rescues']==0
    original=(S.HERE/'graph_block_moves.py').read_text().split('def scores(',1)[1];new=(S.HERE/'cycle_cut_moves.py').read_text().split('def scores(',1)[1];assert new==original.replace('if values[j]<THRESHOLD:chosen.append(j)\n            else:done[i]=True','chosen.append(j)')
    p.update(id='seen-cohort-positive-cost-cycle-cut-acceptance',hypothesis_id='H2aq',frozen_utc=S.utc(),predecessor_intervention={'path':str(P.PP.relative_to(S.ROOT)),'sha256':S.sha(P.PP)},predecessor_audit={'path':str(prior.relative_to(S.ROOT)),'sha256':S.sha(prior)},candidate_rule='Sameoldmaximalselectedfreeglobaldegree1or2chaincomponents,size2..256,endpointconnectivityveto; additionallyrequirestrictdecrease ofselected4neighborcycle rank E-V+component_count immediatelyafterdeletion. Allcomponentscounted;no shortestdistance/labels.',policy='Initialsimpleendpointpaths return byte-identicalanswers/flags. Allothereligible archivedvetooutputs usecyclecutsearch. Choose minimumfinitefulloriginalenergy delta amongeligiblecandidates, exacttieslexicographiccelltuple; accept evenzero/positivecost. Unchangedfourcolorbinarydescent aftereachacceptedmove,16rounds/512sweepscaps retainedasfailures. No repeatedstateearlyexit. Nochanges to models/readouts/objective. Explicitgraphconstraint andenergy-increasingintervention,notoriginalenergydescent.',determinism='Originalchainenumeration, sortedcelltuples, minimumfulloriginaldelta then lexicographic tie. Cycle rank computed for allselectedcomponents includingisolatedones.',gate_revision='Fixed12hand-enumeratedprior9x9motifs,128seededrandom5x5 masks andfixedfirst4seen120boardswith16corneropenmask; independentchaingeometry/reachability/cyclerank andliteraloriginalenergies. SyntheticknownunaryE=-sumfreep onattachedcycle:positivecost+3deletion mustexecute; unchangednative restoresall3vertices andcycleeachround,leading16roundcap. Fourrotations, no scientificsourceoutput trajectories inpositivegate.',unchanged_score_native_suffix_except_acceptance_sha256=hashlib.sha256(new.encode()).hexdigest())
    for n in ['cycle_cut_rescue.py','cycle_cut_moves.py','gate_cycle_cut_rescue.py','verify_cycle_cut_geometry.py']:p['source_sha256'][str((S.HERE/n).relative_to(S.ROOT))]=S.sha(S.HERE/n)
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.emit('cycle_cut_rescue_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for r in group.values():assert S.sha(S.ROOT/r['path'])==r['sha256']
    for name in ['predecessor_protocol','predecessor_gate_failure','reused_fixedpoint_certificate','predecessor_intervention','predecessor_audit']:
        r=p[name];assert S.sha(S.ROOT/r['path'])==r['sha256']
    assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json')
    return p


def evaluate(gpu):
    p=protocol();c=p['reused_fixedpoint_certificate'];S.save(DEST/'old-chain-fixed-point-control.json',{'status':'passed','protocol_sha256':S.sha(PP),'source_certificate':c,'new_searches':0,'new_certificate_runs':0,'scope':'Hashboundall12288oldstates;cyclefilteronlyrestrictsoldcandidatefamily, oldacceptancethresholdhasnoallowedfirstmove.'});OLD.PP=PP;OLD.DEST=DEST;OLD.protocol=protocol;OLD.B=B;OLD.evaluate(gpu)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();freeze() if v.command=='freeze' else evaluate(v.gpu)
