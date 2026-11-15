"""H1av: remove premature hard exclusions while keeping fixed observations."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
from degree_prior_math import integer_costs,top_count_support,attempt_certificate
from degree_prior_solver import solve

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/permuted_noise'
OUT=ROOT/'artifacts/research/20261114/ripper/exclusion_repair'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-exclusion-repair.json'
ARMS=['original_support','top810_unrestricted','degree20_hard','degree20_unrestricted']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in z.files}


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for name,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/name)==digest,name
    for bank in (range(32) if full else ([] if seed is None else [seed])):
        for name,digest in p['bank_sha256'][str(bank)].items():assert sha(PARENT/f's{bank}'/name)==digest,(bank,name)
    return p


def degree_deficit(n,degree,ii,jj,allowed):
    counts=np.zeros(n,np.int64);np.add.at(counts,ii[allowed],1);np.add.at(counts,jj[allowed],1)
    bad=np.flatnonzero(counts<degree)
    if not len(bad):return None
    vertex=int(bad[0])
    return dict(type='integer_degree_deficit',vertex=vertex,available_neighbors=int(counts[vertex]),required_degree=degree,
                implication='Even selecting every allowed incident edge cannot meet this degree equality; this certifies LP infeasibility.')


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    pre=json.loads((HERE/'exclusion-repair-preflight.json').read_text());assert pre['status']=='validated_synthetic_only'
    for name,digest in pre['code_sha256'].items():assert sha(HERE/name)==digest,name
    previous=json.loads((HERE/'protocol-degree-prior.json').read_text())
    code=[HERE/name for name in ['exclusion_repair_study.py','audit_exclusion_repair.py','preflight_exclusion_repair.py','run_exclusion_repair.py',
                                'degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json',HERE/'audit_permuted_noise_acquisition.py',
            HERE/'protocol-degree-prior.json',HERE/'exclusion-repair-preflight.json']
    p=dict(id='H1av-premature-exclusion-repair',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),arms=ARMS,
           scope='All32 previously examined H1ap observed streams, consumed only through independently archived final counts and supports. No acquisition replay, new observations, calibration, source rows, solver puzzle inputs or GPU use.',
           comparison='Same frozen exact-null costs and supplied degree20 prior. Hard LP allows only original retained pairs; unrestricted LP allows all3240 pairs. Unrestricted top810 same-cost control supplies graph cardinality. No cost tuning or new latent draws.',
           supplied_prior=previous['supplied_prior'],cost=previous['cost'],tie_convention=previous['tie_convention'],
           certificate=previous['certificate'],certificate_rationale=previous['certificate_rationale'],
           hard_infeasibility_shortcut='Before solving hard arm, count allowed incident edges. If any cell has <20, emit exact integer deficit certificate for first cell index and skip LP. This proves LP infeasibility. Otherwise solve; an uncertified solver infeasible status remains unresolved.',
           omitted_true_edge_diagnostic='After optimization/audit, report original missing true edges. Their presence proves exact truth recovery impossible under those hard restrictions; it alone does not prove LP infeasibility.',
           outcome_rules='Exact certified integral graph => success. Certified integral nonexact, certified fractional output or exact degree-deficit infeasibility => resolved failure. All other solver/time/numerical/certificate gaps unresolved. No primal rounding or fallback denominator. Fractional output does not exclude an integral co-optimum; no uniqueness claim.',
           primary='Unrestricted degree20 arm has >=29/32 certified integral exact graphs. Supported if successes>=29; falsified only if successes+unresolved<29; otherwise inconclusive. Hard/cardinality/original controls descriptive, all32 retained.',
           controls='BP AdamW outside source-only scope, not run. Controls differ only in explicit prior/candidate availability as stated. Same paired source counts; no independent-arm or fresh-source generalization claim.',
           solver_options=previous['solver_options'],solver_version=previous['solver_version'],numpy_version=previous['numpy_version'],
           budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,gpu_seconds=0,max_solver_calls=64,
                       scope='Monitored prepare, all32 paired-bank LP/audit jobs and summary. Each actual LP time limit1s; known degree deficits skip redundant calls. Thread variables set before numerical imports. Synthetic preflight/packaging separate.'),
           query_accounting=dict(historical_training_queries_per_bank=27750,historical_training_queries_total=888000,
                                  historical_calibration_queries_per_bank=2000,historical_calibration_queries_total=64000,
                                  new_queries=0,new_calibrations=0,new_source_rows=0,new_solver_inputs=0),
           code_sha256={str(f.relative_to(ROOT)):sha(f) for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(seed):{name:sha(PARENT/f's{seed}'/name) for name in ['fit.npz','receipt.json','audit.json']} for seed in range(32)})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'audit.json').read_text());assert parent['status']=='validated'
    assert parent['protocol_sha256']==sha(HERE/'protocol-permuted-noise.json')
    rows=[]
    for seed in p['seeds']:
        d=PARENT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['auditor_sha256']==sha(HERE/'audit_permuted_noise_acquisition.py')
        assert a['protocol_sha256']==r['protocol_sha256']==parent['protocol_sha256']
        assert r['artifact_sha256']['fit.npz']==sha(d/'fit.npz') and r['queries']==29750
        assert a['training_queries']==27750 and a['calibration_queries']==2000
        old=load_npz(d/'fit.npz');assert old['support'].dtype==bool and old['support'].shape==(3240,)
        assert np.array_equal(old['support'],old['first_crossing']<0)
        rows.append(dict(seed=seed,parent_fit_sha256=sha(d/'fit.npz'),parent_audit_sha256=sha(d/'audit.json'),
                         candidate_pairs=int(old['support'].sum()),historical_training_queries=27750,historical_calibration_queries=2000))
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),banks=rows,new_queries=0))


def fit_arm(candidate,cost,seed,d,arm):
    ii,jj=np.triu_indices(81,1);indices=np.flatnonzero(candidate)
    witness={name:np.empty(0,np.int64) for name in ['primal_twice','node_dual_twice','upper_dual_twice']}
    deficit=degree_deficit(81,20,ii,jj,candidate)
    if deficit is not None:
        x=y=np.empty(0,np.float64);solver=None;certificate=dict(certified=False,reason='certified_degree_deficit')
        category='resolved_failure_infeasible'
    else:
        x,y,solver=solve(81,20,ii[indices],jj[indices],cost[indices],seed,d/f'{arm}-highs.log')
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
            certificate,witness=attempt_certificate(81,20,ii[indices],jj[indices],cost[indices],x,y)
        else:certificate=dict(certified=False,reason='solver_status_or_solution_validity')
        if certificate['certified'] and certificate['integral']:
            support=np.zeros(3240,bool);support[indices]=x==1
            truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
            category='success' if np.array_equal(support,truth) else 'resolved_failure_nonexact'
        elif certificate['certified']:category='resolved_failure_fractional'
        else:category='unresolved'
    arrays=dict(candidate_indices=indices,primal_values=x,row_dual_values=y,**witness)
    metadata=dict(arm=arm,solver=solver,certificate=certificate,infeasibility_certificate=deficit,classification=category)
    return arrays,metadata


def bank(seed):
    started=time.monotonic();p=verify(seed);assert seed in p['seeds'];d=OUT/f's{seed}';d.mkdir(exist_ok=False)
    old=load_npz(PARENT/f's{seed}'/'fit.npz');cost=integer_costs(old['joint_count'],old['equal_count'])
    allpairs=np.ones(3240,bool);top=top_count_support(allpairs,cost)
    arrays=dict(cost=cost,joint_count=old['joint_count'],equal_count=old['equal_count'],original_support=old['support'],top810_support=top)
    metadata={}
    for arm,candidate in [('hard',old['support']),('unrestricted',allpairs)]:
        result,meta=fit_arm(candidate,cost,seed,d,arm);arrays.update({arm+'_'+name:value for name,value in result.items()})
        save(d/f'{arm}-fit.json',meta);metadata[arm]=meta
    np.savez_compressed(d/'model.npz',**arrays)
    receipt=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),wall_s=time.monotonic()-started,
                 parent_fit_sha256=sha(PARENT/f's{seed}'/'fit.npz'),artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()})
    save(d/'receipt.json',receipt)
    from audit_exclusion_repair import audit_bank
    a=audit_bank(seed);print(json.dumps(dict(seed=seed,status=a['status'],classification=a['arms']['unrestricted']['classification'])),flush=True)


def summary():
    p=verify();audits=[];rows=[];hashes={};arrays={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
        assert a['auditor_sha256']==sha(HERE/'audit_exclusion_repair.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['model_sha256']==sha(d/'model.npz');audits.append(a);rows+=a['rows']
        for name,value in load_npz(d/'model.npz').items():arrays[f's{seed}_{name}']=value
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories=[a['arms']['unrestricted']['classification'] for a in audits]
    successes=categories.count('success');unresolved=categories.count('unresolved');failures=32-successes-unresolved
    decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive')
    primary=dict(decision=decision,certified_integral_exact_graphs=successes,resolved_failures=failures,unresolved=unresolved,
                 success_lower_bound=successes,success_upper_bound=successes+unresolved,required=29)
    grouped=[]
    for arm in ARMS[:2]:
        part=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,banks=32,exact_graphs=sum(r['exact_graph'] for r in part),
                            missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part)))
    for arm in ['hard','unrestricted']:
        part=[a['arms'][arm] for a in audits];counts=[a['classification'] for a in part]
        grouped.append(dict(arm='degree20_'+arm,banks=32,certified_integral_exact=counts.count('success'),
                            certified_integral_nonexact=counts.count('resolved_failure_nonexact'),certified_fractional=counts.count('resolved_failure_fractional'),
                            certified_infeasible=counts.count('resolved_failure_infeasible'),unresolved=counts.count('unresolved'),
                            exact_integral_candidates=sum(bool(a['integral_candidate_graph'] and a['integral_candidate_graph']['exact_graph']) for a in part)))
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS)
    np.savez_compressed(PUBLIC/'exclusion_repair_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],supplied_prior=p['supplied_prior'],
                primary=primary,grouped=grouped,rows=rows,bank_audits=audits,source_audit=json.loads((OUT/'source-audit.json').read_text()),
                original_missing_true_banks=sum(a['original_missing_true_edges']>0 for a in audits),
                original_missing_true_total=sum(a['original_missing_true_edges'] for a in audits),artifact_sha256=hashes,
                certificate_export_sha256=sha(PUBLIC/'exclusion_repair_certificates.npz'),all_banks_retained=True,**p['query_accounting'],gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'exclusion_repair.json',result)
    print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.action=='bank':bank(args.seed)
    else:globals()[args.action]()
