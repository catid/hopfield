"""H1bi: infer a common degree from partial-observation LP curvature."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
OUT=ROOT/'artifacts/research/20261114/ripper/partial_degree'
PUBLIC=ROOT/'results/research/ripper/partial_degree'
PARENT=ROOT/'results/research/ripper/fresh_transfer_certificates.npz'
PROTOCOL=HERE/'protocol-partial-degree.json'
ARMS=['aligned','permuted']


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def curvature(values):
    v=np.asarray(values,np.int64);second=v[2:]-2*v[1:-1]+v[:-2]
    return int(np.argmax(second))+1,second


def zero_witness(n,cost):
    assert cost.dtype==np.int64 and (cost>=0).all()
    return dict(primal_twice=np.zeros(len(cost),np.int64),node_dual_twice=np.zeros(n,np.int64),upper_dual_twice=np.zeros(len(cost),np.int64))


def verify():
    p=json.loads(PROTOCOL.read_text())
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=HERE/'partial-degree-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    inputs=[gate,PARENT,HERE/'protocol-fresh-transfer.json',ROOT/'results/research/ripper/fresh_transfer.json',ROOT/'results/research/ripper/fresh_transfer_public_manifest.json']
    p=dict(id='H1bi-partial-observation-degree-curvature',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,arms=ARMS,degrees=list(range(41)),candidate_degrees=list(range(1,40)),
           scope='Adaptive follow-up on all64 examined H1bh partial-observation cost matrices; no new source draws, labels, corruptions, masks, donor data or puzzle inputs. Aligned primaries only; permuted descriptive.',
           prior='Common degree and bounded grid0..40 on81nodes; no supplied20,810,geometry or preferred-degree tie rule. Historical donor feedback and its label cost remain part of acquisition.',
           costs='Unchanged exact half-even integer Laplace rates round_even(1000000*(equal+1)/(joint+2)); all3240pairs eligible.',
           objective='For each degree independently minimize cost*x with0<=x<=1 and each node degree d. New solver instance at every d including20; no cached ancestor primal.',
           selector='Vd=2F(d), analytic V0=0. Maximize V[d+1]-2V[d]+V[d-1] over1..39; ties smallest d. No filtering odd degrees.',
           degree_endpoint=dict(arm='aligned',required=31,success='Selected degree exactly20 after all grid proofs certify.'),
           graph_endpoint=dict(arm='aligned',required=27,success='Selected graph certified integral and exactly equals evaluation truth; at most one fewer aggregate exact recovery than archived28/32 fixeddegree result.'),
           decision='Separate finite targets: lower>=required supported; lower+unresolved<required falsified; otherwise inconclusive. Any uncertified grid degree makes bank unresolved for both endpoints. Selected fractional is a graph failure; intermediate fractional objectives valid evidence. Resource failure means incomplete.',
           diagnostics='All41objectives,39curvatures,ties,selecteddegree,graph errors,all proof counts and intermediate fractionality. Permuted32banks descriptive only; no alignment endpoint or pooling. BP first, archived fixeddegree20 and top810 controls.',
           algebra='Additive cell cost bi+bj adds d*sum(b) to F(d), leaving exact curvature unchanged. No universal degree-identification or unique-optimum claim.',
           proof='Literal half-integral primal only, exact scale-two dual feasibility/complementarity/objective equality; no primal snapping. Analytic d0 zero witness valid for nonnegative costs.',
           solver=dict(highspy='1.12.0',numpy='2.3.4',threads=1,parallel='off',solver='simplex',simplex_strategy=1,presolve='off',time_limit=1.,random_seed_base=2026151502,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9),
           seed_rule='Index=2*bank+arm_index; seed=base+index*41+d. Increasing degree order; distinct fresh solver instance per LP.',
           budget=dict(max_cpu_workers=8,new_LPs=2560,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0),
           budget_scope='Worker imports/hash checks, all LPs, independent per-bank exact audit and mutation tests, ordered aggregation/export monitored together. Registration, synthetic gate, archived timing and later public verification separate.',
           query_accounting=dict(new_labels=0,new_sources=0,new_donor_queries=0,each_alternative_historical_recipient=720000,each_alternative_historical_donor=240000,each_alternative_total=960000,unchanged_ripper_ledger_union=2045471,unchanged_h1az_through_h1bh_sequence_union=5148181),
           source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def bank(index):
    p=verify();seed,arm_index=divmod(index,2);arm=ARMS[arm_index];prefix=f's{seed}_{arm}_';start=time.monotonic()
    with np.load(PARENT,allow_pickle=False) as z:arrays={name:z[prefix+name].copy() for name in ['cost','joint_count','equal_count']}
    cost=arrays['cost'];assert cost.dtype==np.int64 and cost.shape==(3240,) and ((cost>=0)&(cost<=1000000)).all()
    ii,jj=np.triu_indices(81,1);d=OUT/f'j{index:03d}';d.mkdir(parents=True,exist_ok=False)
    arrays.update({'d00_'+key:value for key,value in zero_witness(81,cost).items()});records=[]
    for degree in range(1,41):
        x,y,solver=solve(81,degree,ii,jj,cost,40000+index*41+degree,d/f'd{degree:02d}.log')
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
        else:cert=dict(certified=False,reason='solver_status_or_validity');w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
        arrays.update({f'd{degree:02d}_'+key:value for key,value in dict(primal_values=x,node_dual_values=y,**w).items()})
        records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'd{degree:02d}.log')))
        save(d/'progress.json',dict(index=index,pid=os.getpid(),complete_degrees=degree,phase='fit'))
    np.savez_compressed(d/'models.npz',**arrays)
    save(d/'receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/'models.npz'),fit_wall_seconds=time.monotonic()-start))
    from audit_partial_degree import audit_bank
    result=audit_bank(index,False)
    save(d/'audit.json',result);save(d/'progress.json',dict(index=index,pid=os.getpid(),complete_degrees=40,phase='audited'))
    print(json.dumps(dict(index=index,selected_degree=result['selected_degree'],classification=result['classification'])),flush=True)


def summary():
    import shutil
    from audit_partial_degree import aggregate
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);rows=[];files={}
    for index in range(64):
        d=OUT/f'j{index:03d}';a=json.loads((d/'audit.json').read_text());assert a['index']==index and a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_partial_degree.py')
        assert a['models_sha256']==sha(d/'models.npz') and a['receipt_sha256']==sha(d/'receipt.json');rows.append(a)
        dest=PUBLIC/f'j{index:03d}';dest.mkdir()
        for name in ['models.npz','receipt.json']:
            shutil.copyfile(d/name,dest/name);files[f'j{index:03d}/{name}']=sha(d/name)
    grouped,endpoints=aggregate(rows);result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(HERE/'audit_partial_degree.py'),scope=p['scope'],grouped=grouped,endpoints=endpoints,rows=rows,
        certified_LPs=sum(r['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),analytic_zero_proofs=64,query_accounting=p['query_accounting'],artifact_sha256=files,all_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(grouped=grouped,endpoints=endpoints)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','bank','summary']);ap.add_argument('--index',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.index)
    else:globals()[a.action]()
