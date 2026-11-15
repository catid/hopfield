"""Public-only direct NumPy field and map audit of H2y saved-state probes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_field(t, p, a, theta, mu_offsets, aoffs):
    """Negative derivative of explicit pair-table and directed-link energies."""
    p=np.where(t==1,p,t>=2).astype(np.float64)
    theta=theta.astype(np.float64);a=a.astype(np.float64)
    h,w=t.shape[-2:];table=theta[:16].reshape(4,2,2);bias=theta[16:18];kappa=theta[18:]
    result=np.full_like(p,bias[1]-bias[0])
    def slices(dr,dc):
        return (slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc))), (slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))
    for k,(dr,dc) in enumerate(mu_offsets):
        l,r=slices(int(dr),int(dc));x=p[:,l[0],l[1]];y=p[:,r[0],r[1]];v=table[k]
        result[:,l[0],l[1]]-=((v[1,0]-v[0,0])*(1-y)+(v[1,1]-v[0,1])*y)
        result[:,r[0],r[1]]-=((v[0,1]-v[0,0])*(1-x)+(v[1,1]-v[1,0])*x)
    for k,(dr,dc) in enumerate(aoffs):
        l,r=slices(int(dr),int(dc));edge=kappa[k]*a[:,k,l[0],l[1]]
        result[:,l[0],l[1]]+=edge*p[:,r[0],r[1]]
        result[:,r[0],r[1]]+=edge*p[:,l[0],l[1]]
    return result


def verify(directory):
    started=time.perf_counter();receipts=[];rows=[];representatives=[];errors=[]
    for gpu in [0,1]:
        receipt=json.loads((directory/f'gpu{gpu}.json').read_text());assert receipt['status']=='complete'
        path=directory/f'per-state-gpu{gpu}.csv';assert sha(path)==receipt['per_state_csv_sha256']
        with path.open(newline='') as f:part=list(csv.DictReader(f))
        assert len(part)==receipt['source_states']==18432
        receipts.append(receipt);rows+=part;representatives+=receipt['representatives']
    assert receipts[0]['protocol_sha256']==receipts[1]['protocol_sha256']
    assert receipts[0]['script_sha256']==receipts[1]['script_sha256']
    keyed={(int(r['input_index']),int(r['model_seed']),r['arm']):r for r in rows};assert len(keyed)==36864
    for r in representatives:
        path=directory/r['file'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:x={k:z[k] for k in z.files}
        t=x['tokens'];free=t==1
        fields={}
        for k in range(3):
            fields[k]=direct_field(t,x[f'p{k}'],x['couplings'],x['theta'],x['mu_offsets'],x['aoffs'])
            err=float(abs(fields[k]-x[f'field{k}'])[free].max());assert err<1e-4,(r,k,err)
            errors.append(err)
            sig=1/(1+np.exp(-np.clip(x[f'u{k}'].astype(np.float64),-700,700)))
            assert abs(sig-x[f'p{k}'])[free].max()<1e-7
        binary=(x['u0']>0).astype(np.float64)
        fb=direct_field(t,binary,x['couplings'],x['theta'],x['mu_offsets'],x['aoffs'])
        err=float(abs(fb-x['binary_field0'])[free].max());assert err<1e-4
        errors.append(err)
        for k in [0,1]:
            u=x[f'u{k}'].astype(np.float64);direct=u+.5*(fields[k]-u)
            assert abs(direct-x[f'u{k+1}'])[free].max()<1e-4
            # The retained native float32 map is also exactly replayed.
            native=x[f'u{k}']+np.float32(.5)*(x[f'field{k}']-x[f'u{k}'])
            assert np.array_equal(native,x[f'u{k+1}'])
        row=keyed[(r['input_index'],r['model_seed'],r['arm'])]
        checks={
            'residual_u0_inf':abs(fields[0]-x['u0'])[free].max(),
            'residual_u1_inf':abs(fields[1]-x['u1'])[free].max(),
            'residual_u2_inf':abs(fields[2]-x['u2'])[free].max(),
            'one_step_displacement_inf':abs(x['u1']-x['u0'])[free].max(),
            'two_step_return_error_inf':abs(x['u2']-x['u0'])[free].max(),
            'rounded_field_residual_inf':abs(fb-x['u0'])[free].max(),
            'rounding_field_change_inf':abs(fb-fields[0])[free].max(),
            'one_step_changed_rounded_free':(((x['u1']>0)!=(x['u0']>0))&free).sum(),
            'two_step_changed_rounded_free':(((x['u2']>0)!=(x['u0']>0))&free).sum(),
        }
        # Binary self-links use p^2=p, so their exact flip differs from the
        # continuous derivative by minus kappa0*a0 for either flip direction.
        binary_flips=-(1-2*binary)*fb-x['theta'][-5]*x['couplings'][:,0]
        checks['minimum_binary_flip_u0']=binary_flips[free].min()
        checks['negative_binary_flips_u0']=((binary_flips < -1e-6)&free).sum()
        for key,value in checks.items():assert abs(float(value)-float(row[key]))<2e-4,(r,key,value,row[key])
    assert len(representatives)==9
    arms={}
    metric_names=[k for k in rows[0] if k not in ['input_index','model_seed','arm']]
    for arm in ['pl','lp_scaled','unchanged']:
        part=[r for r in rows if r['arm']==arm];assert len(part)==12288
        values={k:np.array([float(r[k]) for r in part]) for k in metric_names}
        assert all(np.isfinite(v).all() for v in values.values())
        for k,name in [(1,'one_step_changed_rounded_free'),(2,'two_step_changed_rounded_free')]:
            same=values[name]==0
            for prefix in ['minimum_binary_flip','negative_binary_flips']:
                assert np.array_equal(values[f'{prefix}_u0'][same],values[f'{prefix}_u{k}'][same])
        arms[arm]={'states':len(part),'metrics':{k:{'minimum':float(v.min()),'median':float(np.median(v)),'mean':float(v.mean()),'maximum':float(v.max())} for k,v in values.items()},
          'residual_at_most_1e_minus4':int((values['residual_u0_inf']<=1e-4).sum()),
          'two_step_return_at_most_1e_minus4':int((values['two_step_return_error_inf']<=1e-4).sum()),
          'nonfixed_two_cycle_candidate_at_1e_minus4':int(((values['residual_u0_inf']>1e-4)&(values['two_step_return_error_inf']<=1e-4)).sum()),
          'unchanged_rounding_one_probe':int((values['one_step_changed_rounded_free']==0).sum()),
          'unchanged_rounding_two_probes':int((values['two_step_changed_rounded_free']==0).sum()),
          'binary_uncertified_u0':int((values['minimum_binary_flip_u0'] < -1e-6).sum()),
          'binary_uncertified_u1':int((values['minimum_binary_flip_u1'] < -1e-6).sum()),
          'binary_uncertified_u2':int((values['minimum_binary_flip_u2'] < -1e-6).sum())}
    return {'status':'passed','scope':'Post-outcome descriptive final-state diagnostics only. 36864 retained states, two native-map probes each; no new primary trajectories or independent inputs. Threshold1e-4 is descriptive and does not prove exact equilibrium, global convergence or absence of longer cycles. Stopped controls are probed without stop masks; their primary certification rule did not require continuous convergence.',
      'arms':arms,'source_states':36864,'map_probes':73728,'new_primary_trajectories':0,'independent_representatives':9,'independent_full_field_checks':36,'independent_binary_flip_boards':9,'same_board_binary_margin_consistency_checks':73728,'maximum_independent_float64_vs_gpu_float32_field_error':max(errors),'representative_selection':'First input of shard0, all3models and all3arms; deterministic, no outcome selection.','receipts':receipts,'verifier_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))
