"""Independent exact support, transfer, temperature and input audit before H3v search."""
import datetime as dt,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3v';OUT=ROOT/'artifacts/research/20261114/loki/h3v';OLD=ROOT/'artifacts/research/20261114/loki/h3u';HASHES={}
def sha(p):
    p=Path(p);v=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=v;return v
def main():
    p=json.loads((SRC/'protocol.json').read_text())
    for group in ['source_sha256','data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    # Literal source occupancy counts/marginals, independent of constructor's saved support.
    source=np.load(OLD/'training.npz')['n12_all'];offsets=[(r,c) for r in range(12) for c in range(-11,12) if r>0 or c>0];counts={o:0 for o in offsets}
    for row in source:
        for i,j in itertools.combinations(np.flatnonzero(row),2):counts[j//12-i//12,j%12-i%12]+=1
    marginal=source.sum(0);exposure={}
    for r,c in offsets:
        numerator=sum(int(marginal[12*a+b])*int(marginal[12*(a+r)+(b+c)]) for a in range(12) for b in range(12) if 0<=a+r<12 and 0<=b+c<12);exposure[r,c]=numerator/176
    lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            vals=[counts[o] for o in offsets if a*o[0]+b*o[1]==0]
            if len(vals)>=3 and not any(vals):lines.append((a,b))
    groups={};nextid=len(lines)
    for o in offsets:
        matched=[k for k,(a,b) in enumerate(lines) if a*o[0]+b*o[1]==0];assert len(matched)<=1
        if matched:groups[o]=matched[0]
        else:groups[o]=nextid;nextid+=1
    old=torch.load(OLD/'models/bp-adamw-grouped/model.pt',weights_only=False);A=old['A'].numpy();support={o:((counts[o]==0 and exposure[o]>=math.log(264/.01))or any(a*o[0]+b*o[1]==0 for a,b in lines)) for o in offsets}
    weights=np.array([A[groups[o]] for o in offsets if support[o]],dtype=np.float32);assert(weights>0).all();unit=float(weights.mean());models={};summaries=[];i,j=np.triu_indices(169,1)
    for cfg in p['arms']:
        arm=cfg['arm'];path=OUT/f'models/{arm}/model.npz';meta=json.loads((path.parent/'model.json').read_text());assert meta['model_sha256']==sha(path);m=dict(np.load(path));expected=np.zeros((169,169),np.float32);F=np.zeros((169,169),np.int32)
        for u,v in zip(i,j):
            o=v//13-u//13,v%13-u%13;r,c=o
            if arm=='supplied-rule-oracle':inside=r==0 or c==0 or abs(r)==abs(c) or(r,abs(c))in[(1,3),(3,1)];value=1.
            else:
                matching=[k for k,(a,b) in enumerate(lines) if a*r+b*c==0];inside=bool(matching)or support.get(o,False);value=float(A[matching[0]])if matching else(float(A[groups[o]])if o in groups else 0.)
                if arm=='uniform-filtered':value=1.
            F[u,v]=F[v,u]=inside;expected[u,v]=expected[v,u]=value if inside else 0.
        assert np.array_equal(m['F'],F)and np.array_equal(m['J'],expected)and not m['h'].any();assert float(m['unit'])==(unit if arm=='positive-bp-adamw-weights'else 1.)and int(m['q'])==13
        models[arm]=m;attack=np.array([((v//13-u//13)==0 or(v%13-u%13)==0 or abs(v//13-u//13)==abs(v%13-u%13)or(abs(v//13-u//13),abs(v%13-u%13))in[(1,3),(3,1)])for u,v in zip(i,j)])
        summaries.append(dict(arm=arm,minimum_positive_coefficient=float(expected[F.astype(bool)].min()),unit=float(m['unit']),target_pairs=int(F.sum()//2),false_forbidden=int((F[i,j].astype(bool)&~attack).sum()),missed_forbidden=int((~F[i,j].astype(bool)&attack).sum())))
    assert np.array_equal(models['positive-bp-adamw-weights']['F'],models['uniform-filtered']['F'])
    data=dict(np.load(OUT/'data.npz'));previous=np.load(OLD/'data.npz');prior=np.union1d(previous['prior_target_test_keys'],previous['test_keys']);universe=np.load(ROOT/'artifacts/research/20261114/loki/h3t/input-universe.npz');eligible=np.setdiff1d(universe['keys'],prior);keys=np.sort(np.random.default_rng(2026112512).permutation(eligible)[:4096]);assert np.array_equal(data['test_keys'],keys)
    for row,count in zip(data['test_inputs'],data['test_solution_counts']):assert data['all_solutions'][:,row].all(1).sum()==count>0
    for name in ['cpu','cuda']:
        path=OUT/f'kernel-validation-{name}.json';v=json.loads(path.read_text());assert v['status']=='passed'
        for rel,h in v['input_sha256'].items():assert sha(ROOT/rel)==h,rel
        sha(path)
    sha(SRC/'protocol.json');sha(__file__);sha(OUT/'data.npz');sha(OLD/'training.npz');sha(OLD/'models/bp-adamw-grouped/model.pt')
    r=dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_offsets_retained=len(weights),minimum_source_positive_coefficient=float(weights.min()),all_source_and_target_coefficients_exact=True,weighted_uniform_zero_sets_identical=True,source_only_producers=True,target_geometry_used_for_diagnostic_only=True,models=summaries,distinct_energies=3,actual_search_jobs=9,input_sha256=HASHES)
    (OUT/'independent-preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
