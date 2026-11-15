"""Synthetic correctness and archived serial/parallel audit parity before H1bi."""
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,platform,time
import numpy as np
from partial_degree_study import ROOT,HERE,OUT,sha,save,npz,curvature,zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_partial_degree import grid,literal_curvature
ARCHIVE=ROOT/'results/research/foureyes/degree_curvature'
ARCHIVE_PROTOCOL=ROOT/'research/20261114/foureyes/degree_curvature.json'


def archived_job(index):
    p=json.loads(ARCHIVE_PROTOCOL.read_text());d=ARCHIVE/f'j{index:03d}';a=npz(d/'models.npz');r=json.loads((d/'receipt.json').read_text())
    assert r['models_sha256']==sha(d/'models.npz')
    return grid(a,r['records'],a['cost'],p['solver'],index)


def main():
    assert platform.node()=='ripper';path=HERE/'partial-degree-preflight.json';assert not path.exists()
    folder=OUT.parent/'partial_degree_preflight';folder.mkdir(exist_ok=False);base=json.loads(ARCHIVE_PROTOCOL.read_text())['solver'];cases=[]
    for case,n in enumerate([4,5]):
        ii,jj=np.triu_indices(n,1);cost=np.ones(len(ii),np.int64)
        if n==4:cost=np.where((jj-ii==1)|((ii==0)&(jj==3)),1,10).astype(np.int64)
        b=np.arange(n,dtype=np.int64)+1;results=[]
        for variant,weights in enumerate([cost,cost+b[ii]+b[jj]]):
            config=dict(base,random_seed_base=2026201502+case*100+variant*10);arrays={'d00_'+k:v for k,v in zero_witness(n,weights).items()};records=[]
            for degree in range(1,n):
                x,y,solver=solve(n,degree,ii,jj,weights,90000+case*100+variant*10+degree,folder/f'{case}-{variant}-{degree}.log')
                cert,w=attempt_certificate(n,degree,ii,jj,weights,x,y);assert cert['certified']
                arrays.update({f'd{degree:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**w).items()});records.append(dict(degree=degree,solver=solver,certificate=cert))
            result=grid(arrays,records,weights,config,0,n);chosen,second=curvature(result['objective_twice']);ref,expected=literal_curvature(result['objective_twice'])
            assert chosen==ref==result['selected_degree'] and second.tolist()==expected==result['curvature_twice']
            assert chosen==(2 if n==4 else 1);assert np.isin(arrays[f'd{chosen:02d}_primal_twice'],[0,2]).all()==(n==4)
            results.append(result)
        assert np.array_equal(np.array(results[1]['objective_twice'])-results[0]['objective_twice'],2*np.arange(n,dtype=np.int64)*b.sum())
        assert results[0]['curvature_twice']==results[1]['curvature_twice']
        cases.append(dict(n=n,selected_degree=results[0]['selected_degree'],selected_integral=n==4,additive_cell_invariance=True,proof_mutations_rejected=sum(r['mutations_rejected'] for r in results)))
    start=time.monotonic();serial=[archived_job(i) for i in range(8)];serial_wall=time.monotonic()-start
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:parallel=list(pool.map(archived_job,range(8)))
    parallel_wall=time.monotonic()-start;assert serial==parallel
    ancestor=json.loads((ARCHIVE/'summary.json').read_text())
    for index,row in enumerate(serial):assert row=={k:ancestor['rows'][index][k] for k in row}
    # Bound the consequential parallelization claim with an exact archived parity gate.
    assert parallel_wall<serial_wall
    files=[HERE/name for name in ['partial_degree_study.py','audit_partial_degree.py','preflight_partial_degree.py','run_partial_degree.py','run.py','degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py','audit_laplace_rate.py']]
    inputs=[ARCHIVE_PROTOCOL,ARCHIVE/'summary.json']+[ARCHIVE/f'j{i:03d}'/name for i in range(8) for name in ['models.npz','receipt.json']]
    result=dict(status='passed',host=platform.node(),synthetic_cases=cases,archived_subset=list(range(8)),archived_proofs_per_mode=320,archived_mutation_rejections_per_mode=960,
                serial_parallel_exact_parity=True,ancestor_summary_exact_parity=True,serial_wall_seconds=serial_wall,parallel_wall_seconds=parallel_wall,speedup=serial_wall/parallel_wall,
                parity_sha256=hashlib.sha256(json.dumps(serial,sort_keys=True).encode()).hexdigest(),scope='Synthetic newLPs and archived H4aj proofs only; no H1bi cost-curve fits before freeze.',
                source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
