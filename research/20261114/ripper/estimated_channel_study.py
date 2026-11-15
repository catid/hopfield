"""H1ce: pairwise composite likelihood from fixed H1ca noisy query records."""
from datetime import datetime,timezone
import argparse,json,os,shutil
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz,zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate,top_count_support
from estimated_channel_math import score,independent_score,UncertifiedCost,estimate,independent_estimate
PARENT=ROOT/'results/research/ripper/posterior_fresh_confirm'
OUT=ROOT/'artifacts/research/20261114/ripper/estimated_channel'
PUBLIC=ROOT/'results/research/ripper/estimated_channel'
PROTOCOL=HERE/'protocol-estimated-channel.json'
NOMINAL=ROOT/'results/research/ripper/pair_likelihood'
ARMS=['estimated_channel']
EXPOSURE={arm:71 for arm in ARMS}


def read(path):return json.loads(path.read_text())


def arm_queries(records,arm):
    assert arm in ARMS
    return np.stack([records['first'][:90],records['repeat1'],records['repeat2']]),records['given'][:90]


def verify():
    p=read(PROTOCOL)
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    return p


def independent_numeric(values,given,channel):
    try:return dict(certified=True,output=independent_score(values,given,channel),error=None)
    except UncertifiedCost as error:return dict(certified=False,output=None,error=str(error))


def numeric_state(arrays,reference):
    production=bool(arrays['cost_certified'].all());independent=bool(reference['certified'])
    return dict(producer_all_costs_certified=production,reference_all_costs_certified=independent,
                fit_allowed=production and independent,reference_error=reference['error'],
                producer_uncertified_edges=int((~arrays['cost_certified']).sum()),
                producer160_digit_edges=int((arrays['cost_precision']==160).sum()))


def fit_models(arrays,numeric,index,p,directory,arm):
    # A stored zero placeholder for an uncertified cost is never given to a solver.
    if not numeric['fit_allowed']:
        return arrays,[]
    assert numeric['producer_all_costs_certified'] and numeric['reference_all_costs_certified'] and arrays['cost_certified'].all()
    cost=arrays['cost'];ii,jj=np.triu_indices(81,1)
    arrays['top810_support']=top_count_support(np.ones(3240,bool),cost)
    arrays.update({'d00_'+key:value for key,value in zero_witness(81,cost).items()});records=[]
    for degree in range(1,41):
        log=directory/f'{arm}-d{degree:02d}.log'
        x,y,solver=solve(81,degree,ii,jj,cost,p['solver']['random_seed_base']-2026111502+41*index+degree,log)
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
            certificate,witness=attempt_certificate(81,degree,ii,jj,cost,x,y)
        else:
            certificate=dict(certified=False,reason='solver_status_or_validity')
            witness={key:np.empty(0,np.int64) for key in ['primal_twice','node_dual_twice','upper_dual_twice']}
        arrays.update({f'd{degree:02d}_'+key:value for key,value in dict(primal_values=x,node_dual_values=y,**witness).items()})
        records.append(dict(degree=degree,solver=solver,certificate=certificate,log_sha256=sha(log)))
    return arrays,records


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gatepath=HERE/'estimated-channel-preflight.json';gate=read(gatepath)
    assert gate['status']=='passed' and not gate['study_scores_read']
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    mathpath=HERE/'estimated-channel-math-check.json';seedpath=HERE/'estimated-channel-seed-audit.json'
    assert read(mathpath)['status']=='passed' and gate['math_gate_sha256']==sha(mathpath)
    assert read(seedpath)['status']=='passed' and gate['seed_audit_sha256']==sha(seedpath)
    assert gate['proposal_sha256']==sha(HERE/'estimated-channel-proposal.json')
    for rel,digest in read(seedpath)['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==digest,rel
    parent=read(HERE/'protocol-pair-likelihood.json');prior=read(NOMINAL/'summary.json')
    assert len(prior['rows'])==128 and [r['success'] for r in prior['grouped']]==[64,62]
    assert all(r['selected_degree']==20 and r['classification']!='unresolved' for r in prior['rows'])
    design=read(HERE/'estimated-channel-proposal.json')
    inputs=[gatepath,mathpath,seedpath,HERE/'estimated-channel-proposal.json',HERE/'protocol-pair-likelihood.json',
            HERE/'protocol-posterior-fresh-confirm.json',NOMINAL/'summary.json',ROOT/'results/research/ripper/pair_likelihood_public_manifest.json',
            PARENT/'query-record-seal.json',ROOT/'results/research/ripper/posterior_fresh_confirm_public_manifest.json',HERE/'pair-likelihood-math-check.json']
    for seed in range(64):
        inputs += [PARENT/f's{seed}/{name}' for name in ['query-records.npz','query-record-audit.json','masks.npz']]
        inputs += [NOMINAL/f's{seed}/triple90_pair_likelihood-{suffix}' for suffix in ['models.npz','receipt.json']]
    p=dict(design,status='frozen_before_cohort_likelihood_scores_or_new_LPs',frozen_utc=datetime.now(timezone.utc).isoformat(),
           solver=dict(parent['solver'],random_seed_base=design['candidate_solver_seed_base']),source_sha256=gate['source_sha256'],
           input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),input_files=len(inputs))),flush=True)


def prepare_and_fit(values,given,index,p,directory,arm):
    estimator=estimate(values,given);assert estimator==independent_estimate(values,given)
    assert estimator['total_comparisons']==19440
    if not estimator['fit_allowed']:
        return dict(estimator_unexecuted=np.array(True)),estimator,None,[],None
    channel=(estimator['rate_numerator'],estimator['rate_denominator'])
    arrays=score(values,given,channel);assert (arrays['joint_count']==71).all()
    reference=independent_numeric(values,given,channel)
    if reference['certified']:
        expected,joint,_=reference['output'];assert np.array_equal(joint,arrays['joint_count'])
        assert np.array_equal(expected[arrays['cost_certified']],arrays['cost'][arrays['cost_certified']])
    numeric=numeric_state(arrays,reference)
    arrays,records=fit_models(arrays,numeric,index,p,directory,arm)
    return arrays,estimator,numeric,records,reference


def bank(seed,p):
    directory=OUT/f's{seed}';directory.mkdir()
    seal=read(PARENT/'query-record-seal.json');assert seal['record_sha256'][str(seed)]==sha(PARENT/f's{seed}/query-records.npz')
    queries=npz(PARENT/f's{seed}/query-records.npz');references={};completed=0
    for ai,arm in enumerate(ARMS):
        index=seed;values,given=arm_queries(queries,arm)
        arrays,estimator,numeric,records,reference=prepare_and_fit(values,given,index,p,directory,arm)
        references[arm]=reference
        np.savez_compressed(directory/f'{arm}-models.npz',**arrays)
        save(directory/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),estimator=estimator,numeric=numeric,
             records=records,models_sha256=sha(directory/f'{arm}-models.npz'),parent_query_records_sha256=sha(PARENT/f's{seed}/query-records.npz'),
             parent_query_seal_sha256=sha(PARENT/'query-record-seal.json'),
             unexecuted_grid_reason='inadmissible_channel_estimate' if not estimator['fit_allowed'] else (None if numeric['fit_allowed'] else 'uncertified_producer_or_independent_log_cost_interval')))
        completed+=len(records);save(directory/'progress.json',dict(completed_LPs=completed,pid=os.getpid()))
    from audit_estimated_channel import audit_bank
    checked=audit_bank(seed,False,p,references);save(directory/'audit.json',checked)
    print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def worker(worker_id):
    assert 0<=worker_id<8;p=verify()
    for seed in range(worker_id,64,8):bank(seed,p)


def summary():
    from audit_estimated_channel import aggregate
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True)
    rows=[];files={}
    for seed in range(64):
        directory=OUT/f's{seed}';checked=read(directory/'audit.json')
        assert checked['status']=='validated' and checked['protocol_sha256']==sha(PROTOCOL) and checked['auditor_sha256']==sha(HERE/'audit_estimated_channel.py')
        for name,digest in checked['artifact_sha256'].items():assert sha(directory/name)==digest
        rows+=checked['rows'];target=PUBLIC/f's{seed}';target.mkdir()
        for name in ['audit.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz','receipt.json']]:
            shutil.copyfile(directory/name,target/name);files[f's{seed}/{name}']=sha(target/name)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],rows=rows,**aggregate(rows),
                attempted_LPs=sum(r['attempted_LPs'] for r in rows),new_certified_LPs=sum(r['certified_grid_LPs'] for r in rows),
                archived_selected_certificates=64,mutations_rejected=sum(r['mutations_rejected']+3*r['archived_selected_certificates'] for r in rows),
                analytic_zero_proofs=sum(r['analytic_zero_proof'] for r in rows),artifact_sha256=files,
                new_source_IDs=0,new_noisy_queries=0,parent_physical_noisy_union=2073600,official_noisy_label_sequence=16760919,all64_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result)
    print(json.dumps(dict(endpoints=result['endpoints'],grouped=result['grouped'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker','summary']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    worker(args.seed) if args.action=='worker' else globals()[args.action]()
