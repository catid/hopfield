"""Independent grouped PL, optimizer, transfer and source-calibration checks."""
import argparse,datetime as dt,hashlib,importlib.util,itertools,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3u';OUT=ROOT/'artifacts/research/20261114/loki/h3u'
HASHES={}
def sha(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASHES[str(p.relative_to(ROOT))]=h;return h
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load_driver():
    spec=importlib.util.spec_from_file_location('h3u_driver',SRC/'driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def independent_design():
    source=np.load(OUT/'training.npz')['n12_all'];assert source.shape==(176,144) and len(np.unique(source,axis=0))==176 and (source.sum(1)==12).all()
    offsets=[(r,c) for r in range(12) for c in range(-11,12) if r>0 or c>0];counts={o:0 for o in offsets}
    for board in source:
        for i,j in itertools.combinations(np.flatnonzero(board),2):counts[j//12-i//12,j%12-i%12]+=1
    lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or (a<0 or (a==0 and b<0)):continue
            values=[counts[o] for o in offsets if a*o[0]+b*o[1]==0]
            if len(values)>=3 and not any(values):lines.append((a,b))
    group={};nxt=len(lines)
    for o in offsets:
        matches=[k for k,(a,b) in enumerate(lines) if a*o[0]+b*o[1]==0];assert len(matches)<=1
        if matches:group[o]=matches[0]
        else:group[o]=nxt;nxt+=1
    assert len(lines)==4 and nxt==224
    m=source.sum(0).astype(np.int64);exposure={}
    for dr,dc in offsets:
        total=0
        for r in range(12):
            for c in range(12):
                rr,cc=r+dr,c+dc
                if 0<=rr<12 and 0<=cc<12:total+=int(m[12*r+c])*int(m[12*rr+cc])
        exposure[dr,dc]=total/176
    return source,offsets,counts,lines,group,exposure

def literal_J(A,b,n,lines,group):
    # Independent per-pair assignment, including every target-only offset.
    rows=[]
    for i in range(n*n):
        row=[]
        for j in range(n*n):
            if i==j:row.append(A.new_zeros(()));continue
            u,v=min(i,j),max(i,j);o=v//n-u//n,v%n-u%n
            line=[k for k,(a,z) in enumerate(lines) if a*o[0]+z*o[1]==0]
            g=line[0] if line else group.get(o)
            row.append(A[g] if g is not None else A.new_tensor(-1.))
        rows.append(torch.stack(row))
    return torch.stack(rows)

def loss_check():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;D=load_driver();p=D.frozen();source,offsets,counts,lines,groups,exposure=independent_design();d=D.source_design()
    assert np.array_equal(d['groups'],[groups[o] for o in offsets])
    subsets=[c for k in [1,2,3] for c in itertools.combinations(range(12),k)];assert len(subsets)==298
    ix=[];givens=[]
    for k,board in enumerate(source[:3]):
        cells=np.flatnonzero(board)
        for subset in subsets:
            x=np.zeros(144,bool);x[cells[list(subset)]]=True;givens.append(x);ix.append(k)
    free=~np.array(givens);expanded=torch.tensor(source[np.array(ix)],dtype=torch.float64)
    A=torch.linspace(-.23,.31,224,dtype=torch.float64,requires_grad=True);b=torch.tensor([-.17],dtype=torch.float64,requires_grad=True)
    J=literal_J(A,b,12,lines,groups);field=expanded@J+b
    literal=torch.where(expanded.bool(),torch.nn.functional.softplus(field),torch.nn.functional.softplus(-field))[torch.tensor(free)].mean()
    lg=torch.autograd.grad(literal,(A,b));target=torch.tensor(source[:3],dtype=torch.float64)
    weights=torch.where(target.bool(),target.new_tensor(231.),target.new_tensor(298.));weights/=weights.sum()
    aggregated=D.source_loss(A,b,target,weights,torch.tensor(d['pair_groups']));ag=torch.autograd.grad(aggregated,(A,b))
    torch.testing.assert_close(literal,aggregated,atol=1e-10,rtol=1e-9)
    for u,v in zip(lg,ag):torch.testing.assert_close(u,v,atol=1e-10,rtol=1e-9)
    param=torch.tensor([.2,-.3,.4],dtype=torch.float64,requires_grad=True);expected=param.detach().clone();m=torch.zeros_like(param);v=torch.zeros_like(param)
    opt=torch.optim.AdamW([param],lr=.3,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False)
    for t,g in enumerate([torch.tensor([.13,-.27,.08],dtype=torch.float64),torch.tensor([-.1,.2,.31],dtype=torch.float64)],1):
        param.grad=g;opt.step();expected*=1-.3*.01;m=.9*m+.1*g;v=.999*v+.001*g*g;expected-=.3*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)
        torch.testing.assert_close(param,expected,atol=1e-12,rtol=1e-12)
    target=torch.tensor(source,dtype=torch.float64);weights=torch.where(target.bool(),target.new_tensor(231.),target.new_tensor(298.));weights/=weights.sum()
    cpu=D.source_loss(A,b,target,weights,torch.tensor(d['pair_groups']));cg=torch.autograd.grad(cpu,(A,b))
    AA=A.detach().float().cuda().requires_grad_();bb=b.detach().float().cuda().requires_grad_();tt=target.float().cuda();ww=weights.float().cuda();gg=torch.tensor(d['pair_groups'],device='cuda')
    actual=D.source_loss(AA,bb,tt,ww,gg);gpu=torch.autograd.grad(actual,(AA,bb));loss_error=abs(float(cpu.detach())-float(actual.detach()));gradient_error=max(float((u-v.cpu().double()).abs().max()) for u,v in zip(cg,gpu))
    assert loss_error<=1e-5 and gradient_error<=1e-5
    # Bounded, actual-source forward/backward benchmark. No optimizer step or search.
    full_labels=np.repeat(source,298,axis=0);full_free=np.ones_like(full_labels,bool)
    for k,board in enumerate(source):
        cells=np.flatnonzero(board)
        for t,subset in enumerate(subsets):full_free[k*298+t,cells[list(subset)]]=False
    ft=torch.tensor(full_labels,device='cuda',dtype=torch.float32);ff=torch.tensor(full_free,device='cuda');i,j=torch.triu_indices(144,144,1,device='cuda');timings={};full_loss=None;full_grad=None
    for path in ['expanded_full_rows','aggregated_complete_patterns']:
        times=[]
        for repeat in range(4):
            torch.cuda.synchronize();start=time.perf_counter()
            if path=='expanded_full_rows':
                J=AA.new_zeros(144,144).index_put((i,j),AA[gg]).index_put((j,i),AA[gg]);fields=ft@J+bb
                value=torch.nn.functional.binary_cross_entropy_with_logits(-fields,ft,reduction='none')[ff].mean()
            else:value=D.source_loss(AA,bb,tt,ww,gg)
            grad=torch.autograd.grad(value,(AA,bb));torch.cuda.synchronize()
            if repeat:times.append(time.perf_counter()-start)
        if path=='expanded_full_rows':full_loss=float(value.detach());full_grad=[z.detach().cpu() for z in grad]
        timings[path]=times
    assert abs(full_loss-float(actual.detach()))<=1e-5
    assert max(float((a-b.detach().cpu()).abs().max()) for a,b in zip(full_grad,gpu))<=1e-5
    save(OUT/'loss-validation.json',dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),literal_expanded_rows=len(ix),
       cpu_loss_absolute_error=float(abs(literal.detach()-aggregated.detach())),cpu_gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(lg,ag)),
       adamw_two_step_moment_parity=True,actual_source_cpu_gpu_loss_error=loss_error,actual_source_cpu_gpu_gradient_error=gradient_error,
       actual_full_expanded_rows=len(full_labels),actual_compressed_patterns=len(source),timings_seconds=timings,median_forward_backward_speedup=float(np.median(timings['expanded_full_rows'])/np.median(timings['aggregated_complete_patterns'])),
       no_optimizer_fit_or_target_search=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    print((OUT/'loss-validation.json').read_text())

def models_check():
    torch.set_num_threads(1);D=load_driver();p=D.frozen();source,offsets,counts,lines,groups,exposure=independent_design();design=dict(np.load(OUT/'source-design.npz'))
    assert np.array_equal(design['counts'],[counts[o] for o in offsets]);assert np.array_equal(design['exposure'],[exposure[o] for o in offsets]);assert np.array_equal(design['groups'],[groups[o] for o in offsets]);assert np.array_equal(design['lines'],lines)
    summaries=[];energy_groups=[];models={}
    for cfg in p['arms']:
        arm=cfg['arm'];folder=OUT/f'models/{arm}';rec=json.loads((folder/'model.json').read_text());assert rec['model_sha256']==sha(folder/'model.pt');assert rec['source_parameters_sha256']==sha(folder/'source-parameters.npz')
        model=torch.load(folder/'model.pt',weights_only=False);A=model['A'].double();b=model['b'].double();J=literal_J(A,b,12,lines,groups);x=source.astype(np.float64);field=x@J.numpy()+b.numpy();cal=dict(delta=float(((1-2*x)*field).min()),unit=float(abs(field).mean()),tau=float((.5*x*(field+b.numpy())).sum(1).max()))
        assert all(cal[k]==model[k] for k in cal),(arm,cal)
        eligible=all(np.isfinite(v) for v in cal.values()) and cal['delta']>0 and cal['unit']>0 and bool(torch.isfinite(A).all() and torch.isfinite(b).all());assert eligible==rec['source_eligible']
        if arm!='bp-adamw-grouped':
            expected=np.full(224,-1.);expected[:4]=11
            for o,g in groups.items():
                if g<4:continue
                r,c=o
                if arm=='count-hybrid':forbidden=counts[o]==0
                elif arm=='exposure-filtered':forbidden=counts[o]==0 and exposure[o]>=math.log(264/.01)
                else:forbidden=r==0 or c==0 or abs(r)==abs(c) or (r,abs(c)) in [(1,3),(3,1)]
                expected[g]=12*forbidden-1
            assert np.array_equal(A.numpy(),expected) and not b.any()
        Jt=literal_J(A,b,13,lines,groups).numpy();assert np.array_equal(Jt,model['S'][:,:,1,1].numpy())
        assert not model['S'][:,:,0,:].any() and not model['S'][:,:,:,0].any() and not model['B'][:,0].any();assert np.array_equal(model['B'][:,1].numpy(),np.repeat(b.numpy(),169));assert model['q']==13
        rng=np.random.default_rng(83201);fixtures=rng.integers(0,2,(13,169));i,j=np.triu_indices(169,1)
        for row in fixtures:
            literal=(row[i]*row[j]*Jt[i,j]).sum()+(row*b.numpy()).sum();matrix=.5*row@Jt@row+(row*b.numpy()).sum();assert abs(literal-matrix)<1e-9
            for cell in range(169):
                other=row.copy();other[cell]=1-other[cell];delta=(other[i]*other[j]*Jt[i,j]).sum()+(other*b.numpy()).sum()-literal
                assert abs(delta-(1-2*row[cell])*(row@Jt[:,cell]+b.item()))<1e-8
        summaries.append(dict(arm=arm,source_eligible=eligible,calibration_float64=cal,all_transferred_coefficients_exact=True,synthetic_boards=13,synthetic_flip_checks=13*169,model_sha256=sha(folder/'model.pt')))
        models[arm]=model
        if eligible:
            found=None
            for g in energy_groups:
                other=models[g['representative']]
                if all(torch.equal(model[k],other[k]) for k in ['S','B']) and all(model[k]==other[k] for k in ['q','delta','tau','unit']):found=g;break
            if found is None:found=dict(representative=arm,members=[]);energy_groups.append(found)
            found['members'].append(arm)
    data=dict(np.load(OUT/'data.npz'));old=np.load(ROOT/'artifacts/research/20261114/loki/h3t/input-universe.npz');prior=np.union1d(old['prior_target_keys'],np.load(ROOT/'artifacts/research/20261114/loki/h3t/data.npz')['test_keys']);eligible=np.setdiff1d(old['keys'],prior)
    expected=np.sort(np.random.default_rng(2026112511).permutation(eligible)[:4096]);assert np.array_equal(data['test_keys'],expected) and not np.intersect1d(expected,prior).size
    for given,count in zip(data['test_inputs'],data['test_solution_counts']):assert data['all_solutions'][:,given].all(1).sum()==count>0
    for name in ['data.npz','training.npz','source-design.npz','loss-validation.json','backend-validation-cpu.json','backend-validation-cuda.json']:sha(OUT/name)
    logical=sum(len(g['members']) for g in energy_groups)*3
    save(OUT/'energy-groups.json',dict(status='grouped_before_target_inference',utc=dt.datetime.now(dt.timezone.utc).isoformat(),groups=energy_groups,logical_models=logical//3,actual_unique_search_jobs=3*len(energy_groups),avoided_identical_jobs=logical-3*len(energy_groups)))
    sha(OUT/'energy-groups.json');sha(SRC/'protocol.json');sha(SRC/'driver.py');sha(__file__)
    save(OUT/'independent-preflight.json',dict(status='passed',utc=dt.datetime.now(dt.timezone.utc).isoformat(),all_identities_exact=True,models=summaries,all_source_and_target_calibrations_verified=True,complete_input_reconstruction=True,input_sha256=HASHES))
    print(json.dumps(dict(status='passed',models=summaries,actual_search_jobs=3*len(energy_groups))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['loss','models']);a=p.parse_args()
    if a.mode=='loss':loss_check()
    else:models_check()
