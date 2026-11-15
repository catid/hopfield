"""H4ap independent resampled counts, all LP proofs and paired decisions."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json,time
import numpy as np
from bootstrap_consensus import ROOT,HERE,OUT,PUBLIC,PARENT,PROTOCOL,FAMILIES,sha,read,save,npz,verify
from audit_degree_prior import check_certificate
from audit_objective_gap import reconstruct


def reference_indices(index,p):
    draws=[]
    for rep in range(32):draws.append(np.random.default_rng(p['bootstrap_seed_base']+index*32+rep).integers(0,600,size=(600,),dtype=np.int16))
    return np.array(draws,np.int16)
def reference_counts(labels,draws):
    result=[]
    for chosen in draws:
        sample=labels[chosen];result.append(np.concatenate([(sample[:,left,None]==sample[:,left+1:]).sum(0,dtype=np.int64) for left in range(80)]))
    return np.array(result,np.int64)
def reference_decisions(rows):
    good=sum(r['classification']=='success' for r in rows);unknown=sum(r['classification']=='unresolved' for r in rows);lost=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in rows);unknown_old=sum(r['baseline_exact'] and r['classification']=='unresolved' for r in rows)
    return dict(recovery=dict(lower=good,upper=good+unknown,required=145,decision='supported' if good>=145 else 'falsified' if good+unknown<145 else 'inconclusive'),paired_loss=dict(lower=lost,upper=lost+unknown_old,maximum=8,decision='supported' if lost+unknown_old<=8 else 'falsified' if lost>8 else 'inconclusive'))
def aggregate(rows):
    assert len(rows)==256 and [r['index'] for r in rows]==list(range(256)) and sum(r['baseline_exact'] for r in rows)==129
    grouped=[]
    for family in FAMILIES:
        part=[r for r in rows if r['family']==family];assert len(part)==64
        grouped.append(dict(family=family,baseline_exact=sum(r['baseline_exact'] for r in part),**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},gains=sum(not r['baseline_exact'] and r['classification']=='success' for r in part),losses=sum(r['baseline_exact'] and r['classification'].startswith('resolved_failure') for r in part)))
    return dict(grouped=grouped,primary=reference_decisions(rows),certified_LPs=sum(r['certified_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),fractional_bootstrap_LPs=sum(r['fractional_bootstrap_LPs'] for r in rows),all256_banks_retained=True)

def audit_bank(index,p,public=False,labels=None):
    job=p['jobs'][index];assert job['index']==index and job['group']==index//4 and job['family']==FAMILIES[index%4] and job['rows']==600 and job['parent_prefix_index']==12*(index//4)+3*(index%4)+1
    folder=(PUBLIC if public else OUT)/f'b{index:03d}';draws=npz(folder/'indices.npz')['indices'];assert np.array_equal(draws,reference_indices(index,p));seal=read((PUBLIC if public else OUT)/'indices-seal.json');assert seal['protocol_sha256']==sha(PROTOCOL) and seal['all256_before_any_LPs'];assert seal['entries'][index]==dict(index=index,indices_sha256=sha(folder/'indices.npz'))
    if labels is None:
        from audit_prefix_budget import regenerated_source
        labels=regenerated_source(job)[:600]
    assert labels.shape==(600,81) and ((labels>=0)&(labels<9)).all();costs=reference_counts(labels,draws);a=npz(folder/'models.npz');r=read(folder/'receipt.json');d=job['inferred_degree']
    assert a['costs'].dtype==np.int64 and np.array_equal(costs,a['costs']) and costs.shape==(32,3240)
    assert r['index']==index and r['degree']==d and r['protocol_sha256']==sha(PROTOCOL) and r['source_sha256']==job['source_sha256'] and r['indices_sha256']==sha(folder/'indices.npz') and r['models_sha256']==sha(folder/'models.npz') and len(r['records'])==33
    g=npz(ROOT/f"results/research/foureyes/{job['parent']}/controls/graphs.npz");pi=job['parent_index'];assert g['groups'][pi]==job['group'] and g['conditions'][pi]==job['family'];graph=reconstruct(g['orders'][pi],job['family']);ii,jj=np.triu_indices(81,1);truth=graph[ii,jj];assert np.array_equal(truth,g['truths'][pi]) and (graph.sum(0)==d).all()
    oldindex=job['parent_prefix_index'];old=read(PARENT/'summary.json')['rows'][oldindex];oldarrays=npz(PARENT/f'j{oldindex:03d}/models.npz');assert old['selected_degree']==d and old['classification']!='unresolved';assert np.array_equal(oldarrays['cost'],reference_counts(labels,np.array([np.arange(600)],np.int16))[0]);assert check_certificate(81,d,ii,jj,oldarrays['cost'],*[oldarrays[f'd{d:02d}_'+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]);assert bool(np.array_equal(oldarrays[f'd{d:02d}_primal_twice'],2*truth))==job['baseline_exact']==(old['classification']=='success')
    valid=[];voters=[];certified=mutations=fractional=0;final=None
    for rep,record in enumerate(r['records']):
        assert record['rep']==rep
        if rep==32 and not all(valid):assert record==dict(rep=32,skipped_due_to_uncertified_bootstrap=True);break
        cost=costs[rep] if rep<32 else np.array([64-sum(int(v[e]) for v in voters) for e in range(3240)],np.int64)
        if rep==32:assert np.array_equal(a['consensus_cost'],cost) and np.array_equal(a['votes_twice'],64-cost) and ((cost>=0)&(cost<=64)).all()
        t,y,z=[a[f'r{rep:02d}_'+name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']];ok=check_certificate(81,d,ii,jj,cost,t,y,z);cert=record['certificate'];meta=record['solver'];assert ok==cert['certified'];valid.append(ok)
        assert meta['highs_version']==p['solver']['highspy'] and meta['numpy']==p['solver']['numpy']
        for key,value in p['solver'].items():
            if key in ['highspy','numpy']:continue
            assert meta['solver_options'][('random_seed' if key=='random_seed_base' else key)]==(value+33*index+rep if key=='random_seed_base' else value)
        if not public:assert record['log_sha256']==sha(folder/f'r{rep:02d}.log')
        if not ok:continue
        certified+=1;x=a[f'r{rep:02d}_primal_values'];dual=a[f'r{rep:02d}_node_dual_values'];assert x.dtype==dual.dtype==np.float64 and np.array_equal(2*x,t) and np.isin(x,[0,.5,1]).all() and np.max(np.abs(2*dual-y))<=1e-6
        obj=sum(int(c)*int(v) for c,v in zip(cost,t));assert cert['primal_objective_twice']==cert['dual_objective_twice']==obj and meta['solver_objective']==obj/2 and meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid'];assert cert['integral']==bool(np.isin(t,[0,2]).all()) and cert['primal_rounding_used'] is False
        for which in range(3):
            bad=[t.copy(),y.copy(),z.copy()];bad[which][0]+=1;assert not check_certificate(81,d,ii,jj,cost,*bad);mutations+=1
        if rep<32:voters.append(t);fractional+=int(not cert['integral'])
        else:final=t
    classification='unresolved';missing=extra=None
    if len(valid)==33 and all(valid):
        assert final is not None
        if np.isin(final,[0,2]).all():missing=int((truth&(final!=2)).sum());extra=int((~truth&(final==2)).sum());classification='success' if missing==extra==0 else 'resolved_failure_nonexact'
        else:classification='resolved_failure_fractional'
    row=dict(index=index,group=job['group'],family=job['family'],degree=d,baseline_exact=job['baseline_exact'],classification=classification,missing_true=missing,extra=extra,certified_LPs=certified,mutations_rejected=mutations,fractional_bootstrap_LPs=fractional,uncertified_repetitions=[i for i,v in enumerate(valid) if not v],consensus_skipped=len(valid)==32 and not all(valid))
    return dict(row=row,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),independent_resampled_counts=True,artifact_sha256={name:sha(folder/name) for name in ['indices.npz','models.npz','receipt.json']})

def public_job(index):return audit_bank(index,read(PROTOCOL),True)
def main():
    start=time.monotonic();p=read(PROTOCOL);missing=verify(p,True);summary=read(PUBLIC/'summary.json');assert summary['protocol_sha256']==sha(PROTOCOL)
    for name,h in summary['artifact_sha256'].items():assert sha(PUBLIC/name)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(256)))
    for i,a in enumerate(audits):assert a==read(PUBLIC/f'b{i:03d}/audit.json')
    rows=[a['row'] for a in audits];assert rows==summary['rows']
    for key,value in aggregate(rows).items():assert summary[key]==value
    runtime=read(PUBLIC/'runtime.json');assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0]*8 and runtime['wall_seconds']<180 and runtime['aggregate_cpu_seconds']<900;assert runtime['samples'][-1]['audited_banks']==256 and runtime['samples'][-1]['active_pids']==[]
    print(json.dumps(dict(status='passed',primary=summary['primary'],grouped=summary['grouped'],certified_LPs=summary['certified_LPs'],mutations_rejected=summary['mutations_rejected'],unavailable_raw_dependencies=missing,wall_seconds=time.monotonic()-start)))
if __name__=='__main__':main()
