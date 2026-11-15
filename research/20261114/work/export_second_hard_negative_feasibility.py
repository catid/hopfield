"""Export H2ad evidence; source trajectories remain separate from fresh tests."""
import json
import shutil
import study as S
from mine_second_hard_negatives import PP,DEST


def main():
    out=S.ROOT/'results/research/work';target=out/'second-hard-negative-feasibility';target.mkdir(exist_ok=True)
    source=DEST/'public-preview'
    for path in source.rglob('*'):
        if path.is_file():q=target/path.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,q)
    for name in ['independent-verification.json','import-isolation.json']:shutil.copyfile(DEST/name,target/name)
    proof=json.loads((DEST/'independent-verification.json').read_text());isolation=json.loads((DEST/'import-isolation.json').read_text());solver=json.loads((DEST/'solver-summary.json').read_text());mining=json.loads((DEST/'mining-audit.json').read_text());assert proof['status']=='passed' and isolation['isolation_passed'] and solver['status']=='complete'
    record={'status':'validated','hypothesis':'H2ad','decision':proof['decision'],'source_model_input_trajectories':3000,'source_unique_training_inputs':1000,'fresh_test_inputs':0,'retained_negative_count':869,'constraint_count':1699856,'new_mined_negative_count':14,'genuinely_new_board_count':9,'genuinely_new_constraint_count':9,'previous_unit_margin_violations':6,'previous_energy_order_violations':0,'numerical_LP_solves':3,'solver':solver,'mining':mining,'independent_verification':proof,'import_isolation':isolation,'protocol_sha256':S.sha(PP),'exporter_sha256':S.sha(__file__),'scope':'All original training inputs with three fixed readouts. Feasible witnesses for stated old and mined-negative inequalities. No minimum-norm optimality, global-optimality, training decoder success after refit, or generalization claim. All new source negatives already above labels under previous witness; only unit-margin deficits addressed. Source mining and LP solves excluded from fresh inference totals.'}
    S.save(out/'second-hard-negative-feasibility-summary.json',record)
    summary=json.loads((out/'summary.json').read_text());assert 'second_hard_negative_feasibility' not in summary
    summary['second_hard_negative_feasibility']={k:record[k] for k in ['source_model_input_trajectories','source_unique_training_inputs','fresh_test_inputs','retained_negative_count','constraint_count','numerical_LP_solves','new_mined_negative_count','genuinely_new_board_count','genuinely_new_constraint_count','previous_unit_margin_violations','previous_energy_order_violations','scope']}
    summary['hypotheses'].append({'campaign':'second_hard_negative_feasibility','hypothesis':'H2ad all three original-training mined-negative systems admit feasible real23-parameter witnesses','outcome':{'decision':proof['decision'],'fresh_test_inputs':0}})
    p=json.loads(PP.read_text());summary['protocols']['second-hard-negative-feasibility-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['updated_utc']=S.utc();S.save(out/'summary.json',summary)
    import export_binary_coordinate
    export_binary_coordinate.manifest(out)


if __name__=='__main__':main()
