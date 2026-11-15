"""H3w exact model reuse, inferred-axis and fresh-cohort validation."""
import datetime as dt,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3w';OUT=ROOT/'artifacts/research/20261114/loki/h3w';BASE=ROOT/'artifacts/research/20261114/loki';HASHES={}
def sha(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def main():
    p=json.loads((SRC/'protocol.json').read_text())
    for group in ['source_sha256','data_sha256']:
        for rel,h in p[group].items():assert sha(ROOT/rel)==h,rel
    source=np.load(BASE/'h3u/training.npz')['n12_all'];offsets=[(r,c)for r in range(12)for c in range(-11,12)if r>0 or c>0];counts={o:0 for o in offsets}
    for row in source:
        for i,j in itertools.combinations(np.flatnonzero(row),2):counts[j//12-i//12,j%12-i%12]+=1
    inferred=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if(a,b)==(0,0)or math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            values=[v for(r,c),v in counts.items()if a*r+b*c==0]
            if len(values)>=3 and not any(values):inferred.append((a,b))
    assert(0,1)in inferred and(1,0)in inferred;assert np.array_equal(inferred,np.load(BASE/'h3u/source-design.npz')['lines']);summaries=[]
    prior=json.loads((BASE/'h3v/independent-preflight.json').read_text());assert prior['status']=='passed'
    for rel,h in prior['input_sha256'].items():assert sha(ROOT/rel)==h,rel
    for cfg in p['arms']:
        model=OUT/f"models/{cfg['arm']}/model.npz";parent=BASE/f"h3v/models/{cfg['model']}/model.npz";assert sha(model)==sha(parent);m=np.load(model);assert int(m['q'])==13 and(m['J'][m['F'].astype(bool)]>0).all()
        summaries.append(dict(arm=cfg['arm'],kernel=cfg['kernel'],model_sha256=sha(model),parent_sha256=sha(parent),unit=float(m['unit']),target_pairs=int(m['F'].sum()//2)))
    data=np.load(OUT/'data.npz');previous=np.load(BASE/'h3v/data.npz');prior_keys=np.union1d(previous['prior_target_test_keys'],previous['test_keys']);universe=np.load(BASE/'h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],prior_keys);keys=np.sort(np.random.default_rng(2026112513).permutation(eligible)[:4096]);assert np.array_equal(data['test_keys'],keys)
    for given,count in zip(data['test_inputs'],data['test_solution_counts']):assert data['all_solutions'][:,given].all(1).sum()==count>0
    g=data['test_inputs'].reshape(-1,13,13);assert(g.sum(1)<=1).all()and(g.sum(2)<=1).all()
    for device in ['cpu','cuda']:
        path=OUT/f'kernel-validation-{device}.json';v=json.loads(path.read_text());assert v['status']=='passed'
        for rel,h in v['input_sha256'].items():assert sha(ROOT/rel)==h,rel
        sha(path)
    sha(SRC/'protocol.json');sha(__file__);sha(OUT/'data.npz');sha(BASE/'h3u/training.npz')
    rec=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),independently_inferred_lines=inferred,row_column_prior_from_source=True,all_models_byte_identical_to_frozen_H3v=True,all_inputs_clues_row_column_consistent=True,models=summaries,actual_search_jobs=12,outcome_identity_mapping=False,input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
if __name__=='__main__':main()
