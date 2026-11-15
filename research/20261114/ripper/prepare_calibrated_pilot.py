"""Old-board implementation inputs; calibration sources exclude pilot training."""
import json
import numpy as np
from run import ROOT,HERE,sha,save,now

OUT=ROOT/'artifacts/research/20261114/ripper/calibrated_pilot'


def main():
    assert np.__version__=='2.3.4';pp=HERE/'protocol-calibrated-pilot.json';p=json.loads(pp.read_text())
    tm=ROOT/p['training_manifest'];cm=ROOT/p['calibration_pool_manifest']
    assert sha(tm)==p['training_manifest_sha256'] and sha(cm)==p['calibration_pool_manifest_sha256']
    train=json.loads(tm.read_text());cal=json.loads(cm.read_text());ids=[];solutions=[]
    for e in cal['inputs']:
        path=ROOT/e['completion_path'];assert sha(path)==e['completion_sha256'];z=np.load(path)
        ids.append(z['source_row']);solutions.append(z['solutions'])
    ids=np.concatenate(ids);solutions=np.concatenate(solutions);unique,first=np.unique(ids,return_index=True)
    pool=solutions[first];assert len(unique)==150504
    OUT.mkdir(exist_ok=False);entries=[]
    for seed in range(6):
        e=next(e for e in train['inputs'] if e['seed']==seed);path=ROOT/e['completion_path'];assert sha(path)==e['completion_sha256']
        t=np.load(path);assert not np.isin(t['source_row'],unique).any()
        ix=np.random.default_rng(20284000+seed).choice(len(unique),10000,replace=False)
        cp=OUT/f'calibration-s{seed}.npz';tp=OUT/f'training-s{seed}.npz'
        np.savez_compressed(cp,solutions=pool[ix],source_row=unique[ix]);np.savez_compressed(tp,solutions=t['solutions'][:1500],source_row=t['source_row'][:1500])
        entries.append({'seed':seed,'training_path':str(tp.relative_to(ROOT)),'training_sha256':sha(tp),
            'calibration_path':str(cp.relative_to(ROOT)),'calibration_sha256':sha(cp)})
    save(OUT/'input_manifest.json',{'status':'prepared_without_calibration_or_training_outcomes','utc':now(),
        'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'calibration_pool_unique_rows':len(unique),
        'global_calibration_training_overlap':0,'inputs':entries,'numpy':np.__version__})
    print({'pilot_banks':6,'calibration_boards_each':10000,'training_boards_each':1500},flush=True)


if __name__=='__main__':main()
