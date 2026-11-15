"""Only old boards, with enough rows for a fixed4440training prefix."""
import json
import numpy as np
from run import ROOT,HERE,sha,save,now

OUT=ROOT/'artifacts/research/20261114/ripper/scaled_noise_pilot'


def main():
    assert np.__version__=='2.3.4';pp=HERE/'protocol-scaled-noise-pilot.json';p=json.loads(pp.read_text());old={}
    for name,e in p['old_manifests'].items():
        assert sha(ROOT/e['path'])==e['sha256'];old[name]=json.loads((ROOT/e['path']).read_text())
    OUT.mkdir(exist_ok=False);entries=[];allcal=[];alltrain=[]
    for seed in p['seeds']:
        arrays={}
        for name,m in old.items():
            e=next(x for x in m['inputs'] if x['seed']==seed);role='calibration' if name=='calibrated' else 'training';path=ROOT/e[role+'_path']
            assert sha(path)==e[role+'_sha256'];z=np.load(path);arrays[name]={k:z[k] for k in z.files}
        training={k:np.concatenate([arrays['equal_budget'][k],arrays['rate_transfer'][k]])[:4440] for k in ['source_row','solutions']}
        calibration={k:arrays['calibrated'][k][:1000] for k in ['source_row','solutions']}
        assert len(np.unique(training['source_row']))==4440 and len(np.unique(calibration['source_row']))==1000
        paths={}
        for role,data in [('training',training),('calibration',calibration)]:
            path=OUT/f'{role}-s{seed}.npz';np.savez_compressed(path,**data);paths[role+'_path']=str(path.relative_to(ROOT));paths[role+'_sha256']=sha(path)
        alltrain.extend(training['source_row']);allcal.extend(calibration['source_row']);entries.append(dict(seed=seed,**paths))
    assert not np.isin(allcal,alltrain).any()
    r={'status':'prepared_without_calibration_or_training_outcomes','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),
        'global_calibration_training_overlap':0,'inputs':entries,'numpy':np.__version__};save(OUT/'input_manifest.json',r)
    print({'pilot_banks':6,'calibration_boards_each':1000,'training_boards_each':4440},flush=True)


if __name__=='__main__':main()
