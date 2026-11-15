"""H1au: explicit degree/cardinality priors on fixed H1as candidate graphs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/adaptive_null'
OUT=ROOT/'artifacts/research/20261114/ripper/degree_prior'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-degree-prior.json'
ARMS=['original_support','top810_cardinality','degree20_lp']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in z.files}


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    for bank in (range(32) if full else ([] if seed is None else [seed])):
        for name,digest in p['bank_sha256'][str(bank)].items():assert sha(PARENT/f's{bank}'/name)==digest,(bank,name)
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    pre=json.loads((HERE/'degree-prior-preflight.json').read_text());assert pre['status']=='validated_synthetic_only'
    for name,digest in pre['code_sha256'].items():assert sha(HERE/name)==digest,name
    code=[HERE/name for name in ['degree_prior_math.py','degree_prior_solver.py','degree_prior_study.py',
                                 'audit_degree_prior.py','preflight_degree_prior.py','run_degree_prior.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-adaptive-null.json',HERE/'protocol-adaptive-null-continuation.json',
            HERE/'degree-prior-preflight.json',HERE/'audit_adaptive_null_eager.py']
    p=dict(id='H1au-explicit-degree20-prior',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),arms=ARMS,
           scope='All 32 previously examined H1as candidate supports and final counts; no observation replay, new labels, source rows, calibration, solver puzzle inputs or GPU work.',
           supplied_prior='Exactly 20 incident retained edges per cell, implying exactly 810 undirected edges. The top810 control receives that cardinality directly. These are explicit structural priors; no prior-free or uniqueness claim.',
           objective='Use only original retained candidate edges. Minimize sum_e w_e*x_e subject to 0<=x_e<=1 and sum_incident x_e=20 at each of 81 cells. No row/column/block geometry enters construction.',
           cost='w_e = round-half-even(1000000 * [c*ln(1600/81)+(m-c)*ln(12800/14319)]), from original final equal/coquery counts. Producer Decimal80, independent auditor Decimal110; no cost/scale tuning.',
           tie_convention='Top810 sorts (integer cost, lexicographic unordered pair index). LP uses deterministic column pair-index order and frozen HiGHS dual simplex seed/options. No lexicographic objective perturbation or uniqueness claim.',
           certificate='No primal rounding/snapping. Literal primal values must belong to {0,0.5,1}. Multiply by two exactly; require exact degree40. Reconstruct node dual y2=round-to-nearest-even(2*y_float), requiring absolute reconstruction error<=1e-6. Set upper dual z2=min(0,2*w-y2_i-y2_j). Check integer dual feasibility, both complementarity conditions and exact equality sum(w*x2)=20*sum(y2)+sum(z2). Costs and duals restricted to absolute value<2^48. No alternate denominators or fallback scheme.',
           certificate_rationale='Unsigned graph incidence LPs with integral costs and degree RHS admit half-integral basic primal/dual witnesses via tree/odd-cycle components. The fixed scale-two reconstruction is a practical scheme, not an assumption that every solver output is certifiable. Acceptance rests only on exact integer checks. No uniqueness conclusion follows.',
           outcome_rules='Certified integral exact graph => success. Certified integral nonexact or certified fractional output => resolved failure of this fixed procedure. Time/numerical/infeasible-status/nonrepresentable/uncertified results => unresolved, retaining raw values/status. Integral graph recovery is reported separately even if certificate is unavailable. Fractional returned optimum does not prove no integral optimum exists.',
           primary='At least 29/32 successful certified integral exact graphs. Supported if successes>=29; falsified only if successes+unresolved<29; otherwise inconclusive. All 32 outcomes retained; no rounding, reruns or eligible-subset denominator.',
           controls='BP AdamW is outside this source-only study, not run. Original supports and same-cost top810 cardinality control are descriptive. If both LP and top810 recover graphs, do not attribute success uniquely to degree regularity.',
           solver_options=dict(threads=1,parallel='off',solver='simplex',simplex_strategy=1,presolve='off',time_limit=1.,
                               random_seed_base=2026111502,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,log_to_console=False),
           solver_version='HiGHS 1.12.0',numpy_version='2.3.4',
           budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,gpu_seconds=0,
                       scope='Monitored prepare, all 32 single-thread LP/certificate/auditor jobs and summary. Synthetic development preflight and publication packaging separate.'),
           query_accounting=dict(historical_training_queries_per_bank=27750,historical_training_queries_total=888000,
                                  new_queries=0,new_calibrations=0,new_source_rows=0,new_solver_inputs=0),
           code_sha256={str(f.relative_to(ROOT)):sha(f) for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(seed):{name:sha(PARENT/f's{seed}'/name) for name in ['fit.npz','receipt.json','audit.json']} for seed in range(32)})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'audit.json').read_text());assert parent['status']=='validated'
    assert parent['protocol_sha256']==sha(HERE/'protocol-adaptive-null.json')
    rows=[]
    for seed in p['seeds']:
        d=PARENT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['auditor_sha256']==sha(HERE/'audit_adaptive_null_eager.py')
        assert a['protocol_sha256']==r['protocol_sha256']==parent['protocol_sha256']
        assert r['artifact_sha256']['fit.npz']==sha(d/'fit.npz') and r['queries']==27750
        old=load_npz(d/'fit.npz');assert old['support'].dtype==bool and old['support'].shape==(3240,)
        assert old['support'].sum()>=810 and np.array_equal(old['support'],old['first_crossing']<0)
        rows.append(dict(seed=seed,parent_fit_sha256=sha(d/'fit.npz'),parent_audit_sha256=sha(d/'audit.json')))
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),banks=rows,new_queries=0))


def bank(seed):
    from degree_prior_math import integer_costs,top_count_support,attempt_certificate
    from degree_prior_solver import solve
    started=time.monotonic();p=verify(seed);assert seed in p['seeds']
    d=OUT/f's{seed}';d.mkdir(exist_ok=False);old=load_npz(PARENT/f's{seed}'/'fit.npz')
    candidate=old['support'];cost=integer_costs(old['joint_count'],old['equal_count']);top=top_count_support(candidate,cost)
    ii,jj=np.triu_indices(81,1);indices=np.flatnonzero(candidate)
    x,y,solver=solve(81,20,ii[indices],jj[indices],cost[indices],seed,d/'highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
        certificate,witness=attempt_certificate(81,20,ii[indices],jj[indices],cost[indices],x,y)
    else:
        certificate=dict(certified=False,reason='solver_status_or_solution_validity')
        witness={name:np.empty(0,np.int64) for name in ['primal_twice','node_dual_twice','upper_dual_twice']}
    np.savez_compressed(d/'model.npz',cost=cost,original_support=candidate,top810_support=top,candidate_indices=indices,
                        primal_values=x,row_dual_values=y,**witness)
    # Ground-truth geometry is used only after optimization/certification for outcome labeling.
    if certificate['certified'] and certificate['integral']:
        selected=np.zeros(3240,bool);selected[indices]=x==1
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
        category='success' if np.array_equal(selected,truth) else 'resolved_failure_nonexact'
    elif certificate['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    receipt=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),
                 wall_s=time.monotonic()-started,solver=solver,certificate=certificate,classification=category,
                 artifact_sha256={name:sha(d/name) for name in ['model.npz','highs.log']})
    save(d/'receipt.json',receipt)
    from audit_degree_prior import audit_bank
    a=audit_bank(seed);print(json.dumps(dict(seed=seed,status=a['status'],classification=category)),flush=True)


def summary():
    p=verify();audits=[];rows=[];hashes={};arrays={};categories=[]
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
        assert a['auditor_sha256']==sha(HERE/'audit_degree_prior.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['model_sha256']==sha(d/'model.npz');audits.append(a);rows+=a['rows'];categories.append(a['classification'])
        model=load_npz(d/'model.npz')
        for name,value in model.items():arrays[f's{seed}_{name}']=value
        for name in ['model.npz','highs.log','receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    successes=categories.count('success');unresolved=categories.count('unresolved');failures=32-successes-unresolved
    decision='supported' if successes>=29 else ('falsified' if successes+unresolved<29 else 'inconclusive')
    primary=dict(decision=decision,certified_integral_exact_graphs=successes,resolved_failures=failures,unresolved=unresolved,
                 success_lower_bound=successes,success_upper_bound=successes+unresolved,required=29)
    grouped=[]
    for arm in ARMS[:2]:
        part=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,banks=32,exact_graphs=sum(r['exact_graph'] for r in part),
                            missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part)))
    grouped.append(dict(arm=ARMS[2],banks=32,certified_integral_exact=successes,
                        certified_integral_nonexact=categories.count('resolved_failure_nonexact'),
                        certified_fractional=categories.count('resolved_failure_fractional'),unresolved=unresolved,
                        feasible_integral_candidates=sum(a['candidate_feasible'] for a in audits),
                        exact_integral_candidates=sum(bool(a['integral_candidate_graph'] and a['integral_candidate_graph']['exact_graph']) for a in audits)))
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);arrays['classification']=np.array(categories)
    np.savez_compressed(PUBLIC/'degree_prior_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],
                supplied_prior=p['supplied_prior'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),artifact_sha256=hashes,
                certificate_export_sha256=sha(PUBLIC/'degree_prior_certificates.npz'),all_banks_retained=True,
                **p['query_accounting'],gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'degree_prior.json',result)
    with (PUBLIC/'degree_prior_controls_per_bank.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
