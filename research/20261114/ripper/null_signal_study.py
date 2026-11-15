"""H1br: curvature concentration on examined real banks and matched IID nulls."""
from datetime import datetime,timezone
import argparse,json,os,shutil
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz,zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate,top_count_support
from laplace_rate_math import rate_costs
OUT=ROOT/'artifacts/research/20261114/ripper/null_signal'
PUBLIC=ROOT/'results/research/ripper/null_signal'
PARENT=ROOT/'results/research/ripper/higher_noise'
PROTOCOL=HERE/'protocol-null-signal.json'
ARMS=['uniform','balanced']


def sample_null(seed):return np.random.default_rng(seed).integers(0,9,size=(180,81),dtype=np.int8)


def observe(given,labels):
    assert given.shape==labels.shape==(180,81)
    values=np.full(given.shape,-1,np.int8);joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint)
    for row in range(180):
        cells=np.flatnonzero(given[row]);board=labels[row];values[row,cells]=board[cells];block=np.ix_(cells,cells);joint[block]+=1;equal[block]+=board[cells,None]==board[None,cells]
    ii,jj=np.triu_indices(81,1)
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj])


def concentration(objectives):
    if any(v is None for v in objectives):return None
    values=list(map(int,objectives));assert len(values)==41
    curvature=[values[d+1]+values[d-1]-2*values[d] for d in range(1,40)];assert min(curvature)>=0
    denominator=sum(curvature);return dict(numerator=max(curvature) if denominator else 0,denominator=denominator or 1,all_zero_curvature=denominator==0)


def strict_greater(real,null):
    if real is None or null is None:return None
    return real['numerator']*null['denominator']>null['numerator']*real['denominator']


def verify():
    p=json.loads(PROTOCOL.read_text())
    for rel,h in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==h,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=HERE/'null-signal-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,h in g['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    parent=json.loads((HERE/'protocol-higher-noise.json').read_text());proposal=HERE/'null-signal-proposal.json';design=json.loads(proposal.read_text())
    inputs=[gate,proposal,HERE/'protocol-higher-noise.json',PARENT/'summary.json']
    for seed in range(32):inputs+=[PARENT/f's{seed}'/name for name in ['masks.npz']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]]
    p=dict(design,frozen_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_null_draws_and_concentration_ratios',source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           degrees=parent['degrees'],candidate_degrees=parent['candidate_degrees'],costs=parent['costs'],selector=parent['selector'],objective=parent['objective'],proof=parent['proof'],prior=parent['prior'],
           parent_solver=parent['solver'],solver=dict(parent['solver'],random_seed_base=2026261502),parent_protocol_sha256=sha(HERE/'protocol-higher-noise.json'))
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),new_LPs=2560,archived_LPs_rechecked=2560)))


def bank(seed):
    p=verify();d=OUT/f's{seed}';d.mkdir(parents=True,exist_ok=False);shutil.copyfile(PARENT/f's{seed}/masks.npz',d/'masks.npz');masks=npz(d/'masks.npz');labels=sample_null(p['null_seed_base']+seed);np.savez_compressed(d/'null-labels.npz',labels=labels);ii,jj=np.triu_indices(81,1)
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;observed=observe(masks[arm],labels);np.savez_compressed(d/f'{arm}-observations.npz',**observed);cost=rate_costs(observed['joint_count'],observed['equal_count'])
        arrays=dict(cost=cost,joint_count=observed['joint_count'],equal_count=observed['equal_count'],top810_support=top_count_support(np.ones(3240,bool),cost));records=[];arrays.update({'d00_'+k:v for k,v in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,150000+index*41+degree,d/f'{arm}-d{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            arrays.update({f'd{degree:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**witness).items()});records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'{arm}-d{degree:02d}.log')))
        np.savez_compressed(d/f'{arm}-models.npz',**arrays)
        save(d/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/f'{arm}-models.npz'),observations_sha256=sha(d/f'{arm}-observations.npz'),masks_sha256=sha(d/'masks.npz'),null_labels_sha256=sha(d/'null-labels.npz')))
    from audit_null_signal import audit_bank
    audit=audit_bank(seed);save(d/'audit.json',audit);print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def summary():
    from audit_null_signal import aggregate
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);audits=[];files={}
    for seed in range(32):
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_null_signal.py')
        for name,h in a['artifact_sha256'].items():assert sha(d/name)==h
        audits.append(a);dest=PUBLIC/f's{seed}';dest.mkdir()
        for name in ['masks.npz','null-labels.npz']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]:shutil.copyfile(d/name,dest/name);files[f's{seed}/{name}']=sha(d/name)
    rows=[r for a in audits for r in a['rows']];outcomes=aggregate(rows);q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert q==dict(uniform_queries=414720,balanced_queries=414720,shared_queries=368719,new_union_queries=460721)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],rows=rows,**outcomes,raw_audits=audits,query_accounting=p['query_accounting'],literal_null_query_accounting=q,
                null_certified_LPs=sum(r['null_grid']['certified_grid_LPs'] for r in rows),real_certified_LPs_rechecked=sum(r['real_grid']['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['null_grid']['mutations_rejected']+r['real_grid']['mutations_rejected'] for r in rows),analytic_zero_witnesses_checked=128,complete_null_label_gates=64,artifact_sha256=files,all_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(endpoint=result['endpoint'],grouped=result['grouped'],joint_banks=result['joint_banks'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','bank','summary']);ap.add_argument('--seed',type=int);args=ap.parse_args()
    bank(args.seed) if args.action=='bank' else globals()[args.action]()
