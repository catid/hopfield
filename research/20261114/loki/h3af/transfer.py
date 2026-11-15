"""Unchanged N12 positive-F producer extended to a new target size."""
import math
import numpy as np
from common import ROOT,sha

def make(cfg,n):
    params=np.load(ROOT/cfg['parameters_path']);d=np.load(ROOT/cfg['source_design_path']);A=params['A'];offsets=d['offsets'];coefficient=A[d['groups']];mask=(d['counts']==0)&(d['exposure']>=math.log(264/.01));mask|=(offsets@d['lines'].T==0).any(1);unit=float(coefficient[mask].mean());assert np.isfinite(coefficient[mask]).all()and(coefficient[mask]>0).all();lookup={tuple(o):(bool(f),float(w))for o,f,w in zip(offsets,mask,coefficient)}
    def build(q):
        F=np.zeros((q*q,q*q),np.int32);J=np.zeros((q*q,q*q),np.float32)
        for i,j in zip(*np.triu_indices(q*q,1)):
            dr,dc=j//q-i//q,j%q-i%q;line=next((k for k,(a,b)in enumerate(d['lines'])if a*dr+b*dc==0),None);inside,w=(True,float(A[line]))if line is not None else lookup.get((dr,dc),(False,0.))
            if inside:F[i,j]=F[j,i]=1;J[i,j]=J[j,i]=w
        return dict(J=J,F=F,h=np.zeros(q*q,np.float32),q=np.array(q),unit=np.array(unit))
    check=build(13);old=np.load(ROOT/cfg['inherited_n13_model_path'])
    for key,value in check.items():assert np.array_equal(value,old[key]),key
    model=build(n);return model,dict(source_patterns=len(d['labels']),source_retained_offsets=int(mask.sum()),unit=unit,source_parameters_sha256=sha(ROOT/cfg['parameters_path']),source_design_sha256=sha(ROOT/cfg['source_design_path']),inherited_N13_model_identically_reconstructed=True,target_forbidden_pairs=int(model['F'].sum()//2),new_target_size=n,no_refit_or_temperature_recalibration=True)
