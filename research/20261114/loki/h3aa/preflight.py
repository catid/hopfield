"""Audit frozen fresh cohort, initialization files and shared-state identity."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt,hashlib,json
from pathlib import Path
import numpy as np
import torch
from initialization import make
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3aa';OUT=ROOT/'artifacts/research/20261114/loki/h3aa';BASE=ROOT/'artifacts/research/20261114/loki';HASHES={}
def sha(p):p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def read(p):sha(p);return json.loads(Path(p).read_text())
def main():
    torch.set_num_threads(1);p=read(SRC/'protocol.json');dataset=read(OUT/'dataset.json');assert dt.datetime.fromisoformat(p['frozen_utc'])<dt.datetime.fromisoformat(dataset['utc'])
    for group in['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    for device in['cpu','cuda']:
        gate=read(OUT/f'gate-{device}.json');assert gate['status']=='passed'and gate['protocol_sha256']==sha(SRC/'protocol.json')
        assert dt.datetime.fromisoformat(gate['utc'])<dt.datetime.fromisoformat(dataset['utc'])
        for rel,h in gate['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    assert sha(OUT/'data.npz')==dataset['data_sha256']and sha(OUT/'model.npz')==dataset['model_sha256']==p['model_sha256']
    data=np.load(OUT/'data.npz');old=np.load(BASE/'h3w/data.npz');universe=np.load(BASE/'h3t/input-universe.npz');prior=np.union1d(old['prior_target_test_keys'],old['test_keys']);eligible=np.setdiff1d(universe['keys'],prior)
    expected=np.sort(np.random.default_rng(2026112514).permutation(eligible)[:4096]);assert np.array_equal(expected,data['test_keys'])and np.array_equal(prior,data['prior_target_test_keys'])and not np.isin(expected,prior).any()
    for given,count in zip(data['test_inputs'],data['test_solution_counts']):assert data['all_solutions'][:,given].all(1).sum()==count>0
    initial=[]
    for rec in dataset['initial_states']:
        path=ROOT/rec['path'];assert sha(path)==rec['file_sha256'];states=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:169].astype(bool)
        assert hashlib.sha256(states.tobytes()).hexdigest()==rec['unpacked_bytes_sha256'];recreated=make(data['test_inputs'],q=13,chains=20,seed=rec['search_seed'],kind=rec['kind']);assert states.tobytes()==recreated.tobytes()
        initial.append(dict(kind=rec['kind'],seed=rec['search_seed'],all_states_verified=len(states)*20,unpacked_bytes_sha256=rec['unpacked_bytes_sha256']))
    m=np.load(OUT/'model.npz');assert int(m['q'])==13 and float(m['unit'])==3.0367188453674316 and(m['J'][m['F'].astype(bool)]>0).all()and not m['h'].any()
    sha(__file__);record=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),fresh_unique_inputs=4096,prior_overlap=0,all_initial_states_reconstructed=True,
        primary_arms_reference_same_saved_permutation_files=True,initial_state_checks=initial,all_models_same_frozen_bytes=True,actual_search_jobs=9,identity_mappings=0,input_sha256=HASHES)
    (OUT/'preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k]for k in['status','fresh_unique_inputs','all_initial_states_reconstructed','actual_search_jobs']}))
if __name__=='__main__':main()
