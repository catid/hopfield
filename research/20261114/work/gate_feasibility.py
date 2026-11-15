"""No-LP independent source parity gate for linear constraint extraction."""
import json
import numpy as np
import study as S
import feasibility_features as F
from fractions import Fraction

DEST=S.OUT/'training_feasibility'


def main():
    torch,_,_=S.imports();source=S.OUT/'contrastive_training';records=[]
    with np.load(source/'training-arrays.npz') as z:tokens=z['tokens'][:4];labels=z['labels'][:4]
    for seed in range(3):
        model=S.model_for(seed,30,'original','cpu',torch.float64)
        with np.load(source/f'train-readout-seed{seed}.npz') as z:attention=z['couplings'][:4]
        rng=np.random.default_rng(2026111510+seed);errors=[];rational_errors=[]
        for i in range(4):
            matrix,kind,flat=F.constraints(tokens[i],labels[i],attention[i],model.mu_offsets,model.aoffs)
            for row in [0,len(matrix)//2,len(matrix)-3,len(matrix)-2,len(matrix)-1]:
                exact=F.exact_row(tokens[i],labels[i],attention[i],model.mu_offsets,model.aoffs,int(kind[row]),int(flat[row]));rational_errors.append(float(np.max(abs(np.array(list(map(float,exact)))-matrix[row]))))
            t=torch.tensor(tokens[i].astype(np.int64))[None];y=torch.tensor(labels[i],dtype=torch.float64)[None];a=torch.tensor(attention[i],dtype=torch.float64)[None];on=int(((t==1)&y.bool()).sum());off=int(((t==1)&~y.bool()).sum())
            for _ in range(3):
                theta=rng.normal(0,4,23)
                with torch.no_grad():
                    model.T.copy_(torch.tensor(theta[:16].reshape(4,2,2)));model.b.copy_(torch.tensor(theta[16:18]));model.kappa.copy_(torch.tensor(theta[18:]));flip=model.flip_delta(t,y,a)[t==1].numpy();e=model.energy(t,y,a);ep=float((model.energy(t,torch.zeros_like(y),a)-e)/on);dense=float((model.energy(t,torch.ones_like(y),a)-e)/off);expected=np.concatenate([flip,[ep,dense]])
                errors.append(float(np.max(abs(matrix@theta-expected))))
        assert max(errors)<1e-10 and max(rational_errors)<1e-12;records.append({'seed':seed,'source_parameter_probes':12,'maximum_linear_source_error':max(errors),'exact_row_checks':20,'maximum_exact_row_float_difference':max(rational_errors)})
    result={'status':'passed','utc':S.utc(),'records':records,'feature_sha256':S.sha(F.__file__),'gate_sha256':S.sha(__file__),'scientific_lp_solves':0};S.save(DEST/'feature-gate.json',result);S.emit('feasibility_feature_gate',**result)


if __name__=='__main__':main()
