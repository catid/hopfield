"""Public-only independent originalbinaryenergy/flip checks forH2ae."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();manifest=json.loads((directory/'manifest.json').read_text());rows=[];label_rows=[];maximum={'minimum_flip_margin64':0.,'energy64':0.,'label_minimum_flip64':0.,'label_energy64':0.,'label_minus_output64':0.}
    for item in manifest['fixtures']:
        path=directory/item['file'];assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:x={k:z[k] for k in z.files}
        t=x['tokens'];free=t==1;assert x['predictions'].shape==(3,1,120,120)
        for v,arm in enumerate(manifest['arms']):
            board=x['predictions'][v];assert board.dtype==bool and np.array_equal(board[~free],(t>=2)[~free])
            a=x['couplings'].astype(np.longdouble);theta=x['theta'][v].astype(np.longdouble)
            delta=flip(t,board,a,theta,x['mu_offsets'],x['aoffs']);binary_energy=energy(t,board,a,theta,x['mu_offsets'],x['aoffs'])
            # Upstream retains the same tiny log(1+1e-12) entropy constant
            # at everybinaryfreecell; account for itsfloat64literal exactly.
            entropy=np.longdouble(float(np.log(np.float64(1)+np.float64(1e-12))))*int(free.sum())
            values={'minimum_flip_margin64':float(delta[free].min()),'energy64':float(binary_energy[0]+entropy)}
            errors={key:abs(value-float(x[key][v,0])) for key,value in values.items()}
            assert errors['minimum_flip_margin64']<1e-10 and errors['energy64']<1e-7,(item,arm,errors)
            for key,error in errors.items():maximum[key]=max(maximum[key],error)
            rows.append({'input_index':item['input_index'],'model_seed':item['model_seed'],'arm':arm,'independent_values':values,'errors':errors})
        y=x['labels'];theta=x['theta'][2].astype(np.longdouble);delta=flip(t,y,a,theta,x['mu_offsets'],x['aoffs']);le=energy(t,y,a,theta,x['mu_offsets'],x['aoffs'])[0]+entropy
        values={'label_minimum_flip64':float(delta[free].min()),'label_energy64':float(le),'label_minus_output64':float(le-(binary_energy[0]+entropy))}
        errors={key:abs(value-float(x[key][0])) for key,value in values.items()}
        assert errors['label_minimum_flip64']<1e-10 and errors['label_energy64']<1e-7 and errors['label_minus_output64']<1e-7,(item,errors)
        for key,error in errors.items():maximum[key]=max(maximum[key],error)
        label_rows.append({'input_index':item['input_index'],'model_seed':item['model_seed'],'independent_values':values,'errors':errors})
    assert len(rows)==144
    return {'status':'passed','scope':'144deterministicretainedbinaryboards plus48LPcanonical-label diagnostics; directextendedprecisionpairtable/link energies andeveryfreeflip checked. No Torch/SciPy/solver/featurebuilder, no newinference or independentinputs. Labels evaluation-only. Wholecohort exactGPUdiagnosticrecheck separate.','states_checked':144,'canonical_label_states_checked':48,'maximum_errors':maximum,'rows':rows,'label_rows':label_rows,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'helper_sha256':sha(Path(__file__).with_name('verify_training_feasibility.py')),'manifest_sha256':sha(directory/'manifest.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(verify(args.directory),indent=2))
