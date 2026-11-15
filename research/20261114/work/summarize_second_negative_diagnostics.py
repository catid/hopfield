"""Join H2ae saved-state topology, objective comparisons and counter checks."""
import csv
import json
import numpy as np
import study as S
from second_negative_transfer import PP,DEST


def main():
    with (DEST/'per-model-input.csv').open() as f:outcomes={(int(r['input_index']),int(r['model_seed'])):r for r in csv.DictReader(f) if r['arm']=='second_negative_binary'}
    with (DEST/'structures.csv').open() as f:structures={(int(r['input_index']),int(r['model_seed'])):r for r in csv.DictReader(f)}
    rows=[];receipts=[]
    for g in [0,1]:
        receipt=json.loads((DEST/'landscape'/f'gpu{g}.json').read_text());path=DEST/'landscape'/f'per-state-gpu{g}.csv';assert receipt['status']=='complete' and S.sha(path)==receipt['per_state_csv_sha256'];receipts.append(receipt)
        with path.open() as f:
            for row in csv.DictReader(f):
                key=(int(row['input_index']),int(row['model_seed']));row['success']=int(outcomes[key]['success']);row['contains_canonical_label']=int(structures[key]['contains_canonical_label']);rows.append(row)
    assert len(rows)==len({(r['input_index'],r['model_seed']) for r in rows})==12288
    groups={}
    for name,part in [('all',rows),('wrong',[r for r in rows if not r['success']]),('solved',[r for r in rows if r['success']])]:
        gap=np.array([float(r['label_minus_output64']) for r in part]);margin=np.array([float(r['label_minimum_flip64']) for r in part])
        groups[name]={'n':len(part),'canonical_label_lower_energy':int((gap<-1e-6).sum()),'canonical_label_higher_energy':int((gap>1e-6).sum()),'energy_ties_within_1e_minus6':int((abs(gap)<=1e-6).sum()),'canonical_label_flip_stable':int((margin>=-1e-6).sum()),'label_minus_output_energy':{'minimum':float(gap.min()),'median':float(np.median(gap)),'maximum':float(gap.max())},'label_minimum_flip':{'minimum':float(margin.min()),'median':float(np.median(margin)),'maximum':float(margin.max())}}
    path=DEST/'landscape.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:(int(r['input_index']),int(r['model_seed']))))
    verified=json.loads((DEST/'independent-state-verification.json').read_text());assert verified['status']=='passed' and verified['canonical_label_states_checked']==48
    counts=json.loads((S.OUT/'binary_coordinate/source-counter-verification.json').read_text());assert counts['status']=='passed'
    S.save(DEST/'diagnosis.json',{'status':'complete','post_outcome_descriptive':True,'source_states':12288,'new_answer_trajectories':0,'objective_groups':groups,'structure_summary':json.loads((DEST/'structures.json').read_text()),'unchanged_decoder_H2aa_source_counter_verification':counts,'independent_state_verification':verified,'landscape_gpu_receipts':receipts,'protocol_sha256':S.sha(PP),'landscape_csv_sha256':S.sha(path),'script_sha256':S.sha(__file__),'scope':'Canonical labels evaluation-only; objective comparisons do not choose or repair outputs. Counter algebra establishes noadditions conditionalonretainedtotalflipcounter, withindependentactualphase-transition validation on H2aa sourcefixtures using the unchanged decoder; targetperphaseeventtraces notretained.'})


if __name__=='__main__':main()
