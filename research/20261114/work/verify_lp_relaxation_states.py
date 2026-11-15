"""Public-only independent direct fields and binary flips for144H2zstates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_lp_dynamics import direct_field
from verify_training_feasibility import flip


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();manifest=json.loads((directory/'manifest.json').read_text());rows=[];maximum={k:0. for k in ['minimum_flip_margin','original_field_residual','modified_field_residual']}
    for item in manifest['fixtures']:
        path=directory/item['file'];assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:x={k:z[k] for k in z.files}
        t=x['tokens'];free=t==1
        assert x['final_logits'].shape==(3,1,120,120) and np.array_equal(x['theta'][1],x['theta'][2])
        for v,arm in enumerate(manifest['arms']):
            u=x['final_logits'][v].astype(np.float64);p=1/(1+np.exp(-np.clip(u,-700,700)))
            original=direct_field(t,p,x['couplings'],x['theta'][v],x['mu_offsets'],x['aoffs']);modified=original+x['lambdas'][v]*(2*p-1)
            delta=flip(t,u>0,x['couplings'].astype(np.longdouble),x['theta'][v].astype(np.longdouble),x['mu_offsets'],x['aoffs'])
            values={'minimum_flip_margin':float(delta[free].min()),'original_field_residual':float(abs(original-u)[free].max()),'modified_field_residual':float(abs(modified-u)[free].max())}
            errors={key:abs(value-float(x[key][v,0])) for key,value in values.items()}
            for key,error in errors.items():assert error<2e-4,(item,arm,key,error);maximum[key]=max(maximum[key],error)
            # Saved fractional counts usefloat32sigmoid; no equality assumption
            # near threshold. Recompute and record any threshold disagreements.
            counts={'near_half_free':int(((p>.45)&(p<.55)&free).sum()),'fractional_free':int(((p>.05)&(p<.95)&free).sum())}
            rows.append({'input_index':item['input_index'],'model_seed':item['model_seed'],'arm':arm,'independent_values':values,'errors':errors,'fractional_count_discrepancies':{k:value-int(x[k][v,0]) for k,value in counts.items()}})
    assert len(rows)==144
    return {'status':'passed','scope':'144deterministic retainedfinalstates, firstinputeachshard/allmodels/allarms. DirectNumPy pairtable/link field andextendedprecisionbinaryflip checks; no newtrajectories, targetsselectedforoutcome, Torch, SciPy, solver or featurebuilder. Wholecohort diagnostics separately recheckedonGPU.','states_checked':len(rows),'maximum_errors':maximum,'rows':rows,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'field_helper_sha256':sha(Path(__file__).with_name('verify_lp_dynamics.py')),'flip_helper_sha256':sha(Path(__file__).with_name('verify_training_feasibility.py')),'manifest_sha256':sha(directory/'manifest.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(verify(args.directory),indent=2))
