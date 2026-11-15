"""Independent direct factors, label-free routing and graph checks on public fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_hard_negative_feasibility import valid_shortest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();m=json.loads((directory/'manifest.json').read_text());rows=[];maxmargin=maxenergy=0.;graph_cache={};labels=set();routed_count=0
    for row in m['fixtures']:
        path=directory/row['file'];assert sha(path)==row['sha256']
        with np.load(path) as z:d={k:z[k] for k in z.files}
        t=d['tokens'];a=d['couplings'].astype(LD);theta=d['theta'].astype(LD);mu=d['mu_offsets'];ao=d['aoffs'];token_hash=hashlib.sha256(t.tobytes()).digest()
        def graph(b):
            key=(token_hash,np.packbits(b).tobytes())
            if key not in graph_cache:graph_cache[key]=valid_shortest(t[0],b[0])
            return graph_cache[key]
        assert graph(d['labels']);labels.add(token_hash);valid=[]
        for v in range(4):
            b=d['predictions'][v];assert np.array_equal(b[t!=1],(t>=2)[t!=1]);margin=float(np.where(t==1,flip(t,b,a,theta[v],mu,ao),np.inf).min());err=abs(margin-float(d['minimum_flip_margin64'][v,0]));assert err<1e-9;maxmargin=max(maxmargin,err)
            entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));e=float((energy(t,b,a,theta[v],mu,ao)+entropy)[0]);err=abs(e-float(d['energy64'][v,0]));assert err<1e-6;maxenergy=max(maxenergy,err);valid.append(graph(b))
        numerical=d['hidden_good'][0]&d['finite'][:,0]&d['stopped'][:,0]&np.isfinite(d['minimum_flip_margin64'][:,0])&(d['minimum_flip_margin64'][:,0]>=-1e-6)
        lp=bool(numerical[1] and not d['lp_block_cap'][0] and not d['lp_native_cap'][0]);assert lp==bool(d['lp_accepted'][0]) and row['routed']==(not lp)
        assert d['hybrid_output_model'][0]==(0 if lp else 1)
        assert float(d['source_lp_margin'][0])==float(d['minimum_flip_margin64'][1,0])
        assert np.array_equal(d['predictions'][0],d['bp_control_final_logits']>0)
        if lp:
            for k in ['predictions','iterations','finite','stopped','minimum_flip_margin64','energy64']:assert np.array_equal(d[k][2],d[k][1])
            assert len(d['subset_final_logits'])==0
        else:
            routed_count+=1;assert np.array_equal(d['predictions'][2],d['subset_final_logits']>0)
            assert bool(d['finite'][2,0])==bool(np.isfinite(d['subset_final_logits']).all())
            assert bool(d['stopped'][2,0])==bool(d['finite'][2,0] and d['subset_minimum_margin32'][0]>=-1e-6)
        for k in ['predictions','iterations','finite','stopped','minimum_flip_margin64','energy64']:assert np.array_equal(d[k][2],d[k][3])
        cross=float(np.where(t==1,flip(t,d['predictions'][2],a,theta[1],mu,ao),np.inf).min());assert abs(cross-float(d['hybrid_lp_margin'][0]))<1e-9
        main=bool(numerical[2]);dual=bool(main and np.isfinite(cross) and cross>=-1e-6);assert main==bool(d['accepted_main'][0]) and dual==bool(d['accepted_dual'][0])
        success=[bool(valid[0] and numerical[0]),bool(valid[1] and numerical[1]),bool(valid[2] and main),bool(valid[3] and dual)]
        assert list(map(int,success))==row['expected_success_by_arm'];rows.append({'input_index':row['input_index'],'model_seed':row['model_seed'],'routed':row['routed'],'success_by_arm':list(map(int,success)),'hybrid_lp_margin':cross})
    return {'status':'passed','fixtures':len(rows),'reported_state_checks':4*len(rows),'distinct_shortest_labels':len(labels),'routed_model_cases':routed_count,
        'maximum_margin_error':maxmargin,'maximum_energy_error':maxenergy,'rows':rows,'manifest_sha256':sha(directory/'manifest.json'),'source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started,
        'scope':m['selection']+' Every published original energy/margin, numerical route, own/dual certificate and independent shortest graph outcome checked. No Torch or production decoder.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);print(json.dumps(verify(parser.parse_args().directory),indent=2))
