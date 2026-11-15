"""Publish source cycle-acceptance and lexical controls; no fresh totals added."""
import json
import shutil
import study as S


def main():
    public=S.ROOT/'results/research/work';audits={}
    for campaign,artifact,hypothesis,pp in [('cycle_cut_rescue','cycle_cut_rescue_checked','H2as','cycle-cut-rescue-checked-protocol.json'),('lexical_cycle_cut','lexical_cycle_cut','H2at','lexical-cycle-cut-protocol.json')]:
        source=S.OUT/artifact;out=public/campaign.replace('_','-');out.mkdir(exist_ok=True);a=json.loads((source/'audit.json').read_text());assert a['status']=='passed';audits[campaign]=a;S.save(public/(campaign.replace('_','-')+'-summary.json'),a)
        for name in ['protocol.json','numerical-gate.json','old-chain-fixed-point-control.json','audit.json','per-input.csv','worker0.json','worker1.json','independent-state-verification.json','state-import-isolation.json']:shutil.copyfile(source/name,out/name)
        if campaign=='lexical_cycle_cut':shutil.copyfile(source/'paired-ranking.csv',out/'paired-ranking.csv')
        fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
        for path in (source/'state-fixtures').iterdir():
            if path.is_file():shutil.copyfile(path,fixtures/path.name)
    for name,artifact,files in [('cycle-cut-failed-gate','cycle_cut_rescue',['protocol.json','failed-gate.json']),('cycle-cut-unexecuted','cycle_cut_rescue_v2',['protocol.json','superseded.json']),('cycle-cut-preflight','cycle_cut_preflight',['numerical-gate.json']),('lexical-cycle-preflight','lexical_cycle_cut_preflight',['numerical-gate.json','preflight-serialization-failure.json'])]:
        out=public/name;out.mkdir(exist_ok=True)
        for file in files:shutil.copyfile(S.OUT/artifact/file,out/file)
    summary=json.loads((public/'summary.json').read_text())
    for campaign,artifact,hypothesis,pp in [('cycle_cut_rescue','cycle_cut_rescue_checked','H2as','cycle-cut-rescue-checked-protocol.json'),('lexical_cycle_cut','lexical_cycle_cut','H2at','lexical-cycle-cut-protocol.json')]:
        assert campaign not in summary;a=audits[campaign];summary[campaign]={'hypothesis':hypothesis,'fresh_inputs':0,'included_model_inputs':12288,'new_search_inputs':63,'reused_veto_prefixes':12288,'targets':a['targets'],'joint_success':a['joint_success'],'accepted_moves':a['accepted_moves'],'positive_cost_moves':a['positive_cost_moves'],'new_optimizer_updates':0,'scope':'Finiteposthocsourcecontrol. Explicitenergy-increasingcyclegraphprior; nofreshgeneralizationclaim. LearnedrankingadvantagefalsifiedbyH2at. Allprefixes/controlsreused, fullcostcharged.'};summary['hypotheses'].append({'campaign':campaign,'hypothesis':hypothesis,'outcome':a['targets']});summary['audit_sha256'][str((S.OUT/artifact/'audit.json').relative_to(S.ROOT))]=S.sha(S.OUT/artifact/'audit.json')
    for pp,hypothesis,status in [('cycle-cut-rescue-protocol.json','H2aq','failed_verifier_metadata_gate_before_interventions'),('cycle-cut-rescue-v2-protocol.json','H2ar','superseded_unexecuted'),('cycle-cut-rescue-checked-protocol.json','H2as','completed'),('lexical-cycle-cut-protocol.json','H2at','completed')]:
        path=S.HERE/pp;p=json.loads(path.read_text());summary['protocols'][path.stem]={'sha256':S.sha(path),'frozen_utc':p['frozen_utc'],'hypothesis':hypothesis,'status':status}
    summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
