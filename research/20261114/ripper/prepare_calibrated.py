"""Fresh independent banks from globally disjoint calibration/training pools."""
import csv
import json
import numpy as np
from run import ROOT,HERE,sha,save,now

OUT=ROOT/'artifacts/research/20261114/ripper/calibrated'


def main():
    assert np.__version__=='2.3.4';started=now();pp=HERE/'protocol-calibrated-inputs.json';p=json.loads(pp.read_text())
    source=ROOT/p['source'];assert sha(source)==p['source_sha256']
    raw=ROOT/p['prior_raw_manifest'];assert sha(raw)==p['prior_raw_manifest_sha256']
    proof=json.loads(raw.read_text());assert proof['source_rows']==proof['unique_complete_boards']==p['pool_size']
    excluded=np.zeros(p['pool_size'],bool);exclusions=[]
    for spec in p['exclusion_manifests']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];manifest=json.loads(path.read_text())
        for entry in manifest['inputs']:
            path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256']
            excluded[np.load(path)['source_row']]=True;exclusions.append({'path':entry['completion_path'],'sha256':entry['completion_sha256']})
    assert excluded.sum()==356066
    eligible=np.flatnonzero(~excluded);permutation=np.random.default_rng(20276000).permutation(eligible)
    half=len(permutation)//2;calpool=permutation[:half];trainpool=permutation[half:]
    selected_cal=np.array([np.random.default_rng(20277000+s).choice(calpool,10000,replace=False) for s in range(32)])
    selected_train=np.array([np.random.default_rng(20278000+s).choice(trainpool,1500,replace=False) for s in range(32)])
    assert not np.isin(selected_cal,selected_train).any()
    OUT.mkdir(exist_ok=False)
    np.savez_compressed(OUT/'selected_sources.npz',calibration_source_row=selected_cal,training_source_row=selected_train,
        all_source_row=np.unique(np.concatenate([selected_cal.ravel(),selected_train.ravel()])))
    print(json.dumps({'stage':'source_rows_selected','path':str((OUT/'selected_sources.npz').relative_to(ROOT)),
        'sha256':sha(OUT/'selected_sources.npz'),'calibration_unique':len(np.unique(selected_cal)),
        'training_unique':len(np.unique(selected_train))}),flush=True)
    arrays={'calibration':np.zeros((32,10000,81),np.int8),'training':np.zeros((32,1500,81),np.int8)}
    wanted={}
    for role,selected in [('calibration',selected_cal),('training',selected_train)]:
        for seed in range(32):
            for index,row in enumerate(selected[seed]):wanted.setdefault(int(row),[]).append((role,seed,index))
    remaining=set(wanted)
    with source.open(newline='') as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            if row_index not in wanted:continue
            answer=np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1')
            assert answer.shape==(81,) and np.all((answer>=0)&(answer<9))
            for role,seed,index in wanted[row_index]:arrays[role][seed,index]=answer
            remaining.remove(row_index)
            if not remaining:break
    assert not remaining
    entries=[];all_entries=[]
    for role,selected in [('calibration',selected_cal),('training',selected_train)]:
        flat=arrays[role].reshape(-1,81);boards=flat.reshape(-1,9,9)
        boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):
            assert np.all((boards==digit).sum(1)==1) and np.all((boards==digit).sum(2)==1)
            assert np.all((boxes==digit).sum(2)==1)
        for seed in range(32):
            path=OUT/f'{role}-s{seed}.npz';np.savez_compressed(path,solutions=arrays[role][seed],source_row=selected[seed])
            all_entries.append({'seed':seed,'role':role,'n':len(selected[seed]),'completion_path':str(path.relative_to(ROOT)),'completion_sha256':sha(path)})
    for seed in range(32):
        t=next(e for e in all_entries if e['seed']==seed and e['role']=='training');c=next(e for e in all_entries if e['seed']==seed and e['role']=='calibration')
        entries.append({'seed':seed,'training_path':t['completion_path'],'training_sha256':t['completion_sha256'],
            'calibration_path':c['completion_path'],'calibration_sha256':c['completion_sha256']})
    save(OUT/'all_source_manifest.json',{'status':'prepared_without_calibration_or_training_outcomes','protocol_sha256':sha(pp),
        'source_sha256':sha(source),'inputs':all_entries,'purpose':'Future exclusions must include BOTH calibration and training boards.'})
    result={'status':'prepared_without_outcomes','utc':now(),'preparation_started_utc':started,'protocol_sha256':sha(pp),
        'wrapper_sha256':sha(__file__),'source_sha256':sha(source),'excluded_unique_source_rows':int(excluded.sum()),
        'exclusion_sources':exclusions,'eligible_rows':len(eligible),'calibration_pool_rows':len(calpool),'training_pool_rows':len(trainpool),
        'calibration_row_instances':320000,'training_row_instances':48000,'calibration_unique_rows':len(np.unique(selected_cal)),
        'training_unique_rows':len(np.unique(selected_train)),'calibration_training_overlap':0,'within_bank_overlap':0,'prior_overlap':0,
        'selected_sources_sha256':sha(OUT/'selected_sources.npz'),'all_source_manifest_sha256':sha(OUT/'all_source_manifest.json'),
        'inputs':entries,'numpy':np.__version__}
    save(OUT/'input_manifest.json',result);print({k:v for k,v in result.items() if k not in ('inputs','exclusion_sources')},flush=True)


if __name__=='__main__':main()
