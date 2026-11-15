"""Old H1ad pilot source prefixes, before any new selector outcomes."""
import json
import numpy as np
from run import ROOT,HERE,sha,save,now

OUT=ROOT/'artifacts/research/20261114/ripper/mixture_law_pilot'


def main():
    assert np.__version__=='2.3.4';pp=HERE/'protocol-mixture-law-pilot.json';p=json.loads(pp.read_text())
    path=ROOT/p['prior_input_manifest'];assert sha(path)==p['prior_input_manifest_sha256'];old=json.loads(path.read_text())
    OUT.mkdir(exist_ok=False);entries=[];cal=[];train=[]
    for seed in p['seeds']:
        e=next(e for e in old['inputs'] if e['seed']==seed);entry={'seed':seed}
        for role,n in [('calibration',1000),('training',2220)]:
            path=ROOT/e[role+'_path'];assert sha(path)==e[role+'_sha256'];z=np.load(path);data={k:z[k][:n] for k in z.files}
            dest=OUT/f'{role}-s{seed}.npz';np.savez_compressed(dest,**data);entry[role+'_path']=str(dest.relative_to(ROOT));entry[role+'_sha256']=sha(dest)
            (cal if role=='calibration' else train).extend(data['source_row'])
        entries.append(entry)
    assert not np.isin(cal,train).any()
    save(OUT/'input_manifest.json',{'status':'prepared_without_calibration_or_training_outcomes','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'global_calibration_training_overlap':0,'inputs':entries,'numpy':np.__version__})
    print({'pilot_banks':6,'calibration_boards_each':1000,'training_boards_each':2220},flush=True)


if __name__=='__main__':main()
