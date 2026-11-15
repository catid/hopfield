"""Publish the preserved H2af failed gate and independently verified source regression."""
import json
import shutil
import study as S
from mine_leaf_negatives import PP,DEST


def main():
    out=S.ROOT/'results/research/work';target=out/'leaf-removability';target.mkdir(exist_ok=True)
    for path in (DEST/'public-preview').rglob('*'):
        if path.is_file():q=target/path.relative_to(DEST/'public-preview');q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,q)
    for name in ['independent-source-verification.json','import-isolation.json']:shutil.copyfile(DEST/name,target/name)
    verification=json.loads((DEST/'independent-source-verification.json').read_text());assert verification['status']=='passed';solver=json.loads((DEST/'solver-summary.json').read_text());assert all(r['status']=='uninformative_empty_or_no_leaf' for r in solver['records'])
    record={'status':'validated','hypothesis':'H2af','decision':'uninformative_no_leaf_gate','wrong_source_outputs_by_model':[233,114,233],'eligible_leaf_rows_by_model':[496,224,496],'leafless_source_indices':[578,683],'source_model_input_trajectories':3000,'source_replay_trajectories_separate':6000,'fresh_test_inputs':0,'numerical_LP_solves':0,'independent_verification':verification,'solver':solver,'protocol_sha256':S.sha(PP),'exporter_sha256':S.sha(__file__),'scope':'Failed no-leaf screen preserved; all leaves retained, no selected subset or LP. Real source regression confirmed by exact scaling/readout/source identities and complete cached/reference replays. Source-only labels select leaves and components; not an inference procedure.'};S.save(out/'leaf-removability-summary.json',record)
    summary=json.loads((out/'summary.json').read_text());assert 'leaf_removability' not in summary
    summary['leaf_removability']={k:record[k] for k in ['decision','wrong_source_outputs_by_model','eligible_leaf_rows_by_model','leafless_source_indices','source_model_input_trajectories','source_replay_trajectories_separate','fresh_test_inputs','numerical_LP_solves','scope']};summary['hypotheses'].append({'campaign':'leaf_removability','hypothesis':'H2af every source wrong board has a false-positive free leaf and all such leaves can be made removable while preserving prior rows','outcome':{'decision':'uninformative_no_leaf_gate','numerical_LP_solves':0}});p=json.loads(PP.read_text());summary['protocols']['leaf-removability-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['updated_utc']=S.utc();S.save(out/'summary.json',summary)
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(path.relative_to(out)):S.sha(path) for path in sorted(out.rglob('*')) if path.is_file() and path!=out/'manifest.json'}})


if __name__=='__main__':main()
