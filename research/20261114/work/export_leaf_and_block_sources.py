"""Publish independent H2ag/H2ah/H2ai source evidence without fresh-test counts."""
import json
import shutil
import study as S


def main():
    out=S.ROOT/'results/research/work';summary=json.loads((out/'summary.json').read_text())
    campaigns=[('leaf_block_feasibility','leaf-block-feasibility','H2ag','leaf-block-feasibility-protocol.json'),('leaf_decoder_gate','leaf-decoder-gate','H2ah','leaf-decoder-gate-protocol.json'),('graph_block_source','graph-block-source','H2ai','graph-block-source-protocol.json')]
    for name,public,hypothesis,protocol in campaigns:
        assert name not in summary;source=S.OUT/name;target=out/public;target.mkdir(exist_ok=True);shutil.copytree(source/'public-preview',target,dirs_exist_ok=True)
        for file in ['independent-verification.json','import-isolation.json']:shutil.copyfile(source/file,target/file)
        proof=json.loads((source/'independent-verification.json').read_text());assert proof['status']=='passed';isolation=json.loads((source/'import-isolation.json').read_text());assert isolation['isolation_passed']
        record={'status':'validated','hypothesis':hypothesis,'decision':proof['decision'],'fresh_test_inputs':0,'independent_verification':proof,'protocol_sha256':S.sha(S.HERE/protocol),'exporter_sha256':S.sha(__file__)}
        if hypothesis=='H2ag':record.update(numerical_LP_solves=3,constraint_rows=1701078,retained_leaf_rows=1216,duplicate_component_rows=6,new_source_trajectories=0,solver=json.loads((source/'solver-summary.json').read_text()),scope='All H2af source cases retained with leaf and already-favored component removal rows. Feasible witnesses, not minimum-norm optimality or decoder success.')
        elif hypothesis=='H2ah':record.update(source_model_input_trajectories=3000,baseline_source_outputs_reused=3000,baseline_wrong=580,new_wrong=9,paired_gains=571,paired_losses=0,scope='Additional leaf-margin fit, fixed original norm/cast, unchanged one-bit decoder. All original training inputs; descriptive source gate only.')
        else:record.update(source_model_input_interventions=3000,baseline_source_prefixes_reused=3000,baseline_wrong=580,new_wrong=6,paired_gains=574,paired_losses=0,scope='Original H2ae coefficients unchanged. Label-free selected-chain/cycle block deletion followed by unchanged one-bit descent; every source case retained. Descriptive source gate only.')
        S.save(out/f'{public}-summary.json',record);summary[name]={k:v for k,v in record.items() if k not in ['independent_verification','solver','exporter_sha256']};summary['hypotheses'].append({'campaign':name,'hypothesis':hypothesis,'outcome':{'decision':proof['decision'],'fresh_test_inputs':0}});p=json.loads((S.HERE/protocol).read_text());summary['protocols'][protocol.removesuffix('.json')]={'sha256':S.sha(S.HERE/protocol),'frozen_utc':p['frozen_utc']}
    summary['updated_utc']=S.utc();S.save(out/'summary.json',summary);S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


if __name__=='__main__':main()
