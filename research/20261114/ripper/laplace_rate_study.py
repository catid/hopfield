"""H1aw: a frozen smoothed-rate replacement for exact-law LP evidence."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/permuted_noise'
BASELINE=ROOT/'artifacts/research/20261114/ripper/exclusion_repair'
OUT=ROOT/'artifacts/research/20261114/ripper/laplace_rate'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-laplace-rate.json'
ARMS=['original_support','archived_exact_law_degree20','laplace_rate_top810','laplace_rate_degree20']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in z.files}


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    for bank in (range(32) if full else ([] if seed is None else [seed])):
        for path,digest in p['bank_sha256'][str(bank)].items():assert sha(ROOT/path)==digest,path
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    pre=json.loads((HERE/'laplace-rate-preflight.json').read_text());assert pre['status']=='validated_synthetic_only'
    for name,digest in pre['code_sha256'].items():assert sha(HERE/name)==digest,name
    previous=json.loads((HERE/'protocol-exclusion-repair.json').read_text())
    sources=[HERE/name for name in ['laplace_rate_math.py','laplace_rate_study.py','audit_laplace_rate.py','preflight_laplace_rate.py',
                                   'run_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json',HERE/'audit_permuted_noise_acquisition.py',
            BASELINE/'audit.json',HERE/'protocol-exclusion-repair.json',HERE/'laplace-rate-preflight.json']
    p=dict(id='H1aw-laplace-rate-replacement',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),arms=ARMS,
           scope='All32 previously examined H1ap streams and their archived counts. New unrestricted LP and same-cost cardinality control, compared with archived H1av exact-law LP. No acquisition replay or new observations.',
           supplied_prior=previous['supplied_prior'],
           cost='Integer round-half-even(1000000*(c+1)/(m+2)). Exact integer quotient/remainder producer; independent Fraction rounding auditor. Laplace-smoothed empirical equality rate; no supplied q0 or q1, no tuning. Unexposed pair cost500000.',
           objective='All3240 unordered pairs eligible; minimize sum(cost*x) with 0<=x<=1 and degree20 at every cell. No row/column/box geometry in construction.',
           tie_convention=previous['tie_convention'],certificate=previous['certificate'],certificate_rationale=previous['certificate_rationale'],
           primary='At least29/32 certified integral exact graphs under the smoothed-rate degree20 LP. Supported if successes>=29; falsified only if successes+unresolved<29; otherwise inconclusive. All32 retained.',
           outcome_rules='Certified integral exact => success; certified integral nonexact or certified fractional => resolved failure of this procedure; numerical/time/nonrepresentable/certification gaps => unresolved. No primal rounding or fallback denominator. No uniqueness claim.',
           interpretation='This replaces likelihood/exposure weighting and introduces Laplace smoothing. Success shows this specific procedure does not require supplied q0/q1 on these fixed streams. Failure would NOT establish calibration necessity. No isolated-noise-law-effect, universal robustness or fresh-source claim.',
           controls='BP AdamW outside source-only scope, not run. Original supports, archived exact-law degree20 LP, and same-rate top810 cardinality control retained. Archived LP is reverified, not rerun. Structural degree/cardinality information remains explicit.',
           solver_options=previous['solver_options'],solver_version=previous['solver_version'],numpy_version=previous['numpy_version'],
           budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,gpu_seconds=0,max_solver_calls=32,
                       scope='Prepare, all32 LP/auditor jobs including archived certificate verification and summary. One-second LP cap; thread limits before imports. Synthetic preflight and packaging separate.'),
           query_accounting=previous['query_accounting'],
           code_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(seed):{str(f.relative_to(ROOT)):sha(f) for f in
                         [PARENT/f's{seed}'/name for name in ['fit.npz','receipt.json','audit.json']]+
                         [BASELINE/f's{seed}'/name for name in ['model.npz','receipt.json','audit.json','unrestricted-fit.json']]} for seed in range(32)})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'audit.json').read_text());baseline=json.loads((BASELINE/'audit.json').read_text())
    assert parent['status']==baseline['status']=='validated'
    assert parent['protocol_sha256']==sha(HERE/'protocol-permuted-noise.json') and baseline['protocol_sha256']==sha(HERE/'protocol-exclusion-repair.json')
    assert baseline['primary']['certified_integral_exact_graphs']==32
    rows=[]
    for seed in p['seeds']:
        d=PARENT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['receipt_sha256']==sha(d/'receipt.json') and a['auditor_sha256']==sha(HERE/'audit_permuted_noise_acquisition.py')
        assert a['protocol_sha256']==r['protocol_sha256']==parent['protocol_sha256']
        assert r['artifact_sha256']['fit.npz']==sha(d/'fit.npz') and r['queries']==29750
        old=load_npz(d/'fit.npz');archived=load_npz(BASELINE/f's{seed}'/'model.npz')
        for name in ['joint_count','equal_count']:assert np.array_equal(old[name],archived[name])
        assert np.array_equal(old['support'],archived['original_support'])
        br=json.loads((BASELINE/f's{seed}'/'receipt.json').read_text());ba=json.loads((BASELINE/f's{seed}'/'audit.json').read_text())
        assert ba['receipt_sha256']==sha(BASELINE/f's{seed}'/'receipt.json') and ba['model_sha256']==sha(BASELINE/f's{seed}'/'model.npz')
        assert br['artifact_sha256']['unrestricted-fit.json']==sha(BASELINE/f's{seed}'/'unrestricted-fit.json')
        rows.append(dict(seed=seed,parent_fit_sha256=sha(d/'fit.npz'),baseline_model_sha256=sha(BASELINE/f's{seed}'/'model.npz')))
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),banks=rows,new_queries=0))


def bank(seed):
    start=time.monotonic();p=verify(seed);assert seed in p['seeds'];d=OUT/f's{seed}';d.mkdir(exist_ok=False)
    old=load_npz(PARENT/f's{seed}'/'fit.npz');archived=load_npz(BASELINE/f's{seed}'/'model.npz')
    cost=rate_costs(old['joint_count'],old['equal_count']);top=top_count_support(np.ones(3240,bool),cost)
    ii,jj=np.triu_indices(81,1);x,y,solver=solve(81,20,ii,jj,cost,seed,d/'highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
        certificate,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
    else:
        certificate=dict(certified=False,reason='solver_status_or_solution_validity')
        witness={name:np.empty(0,np.int64) for name in ['primal_twice','node_dual_twice','upper_dual_twice']}
    arrays=dict(cost=cost,joint_count=old['joint_count'],equal_count=old['equal_count'],original_support=old['support'],top810_support=top,
                primal_values=x,row_dual_values=y,**witness)
    for name in ['cost','unrestricted_primal_values','unrestricted_row_dual_values','unrestricted_primal_twice','unrestricted_node_dual_twice','unrestricted_upper_dual_twice']:
        arrays['baseline_'+name]=archived[name]
    np.savez_compressed(d/'model.npz',**arrays)
    if certificate['certified'] and certificate['integral']:
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
        category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
    elif certificate['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    r=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),wall_s=time.monotonic()-start,
           solver=solver,certificate=certificate,classification=category,parent_fit_sha256=sha(PARENT/f's{seed}'/'fit.npz'),
           baseline_model_sha256=sha(BASELINE/f's{seed}'/'model.npz'),artifact_sha256={name:sha(d/name) for name in ['model.npz','highs.log']})
    save(d/'receipt.json',r)
    from audit_laplace_rate import audit_bank
    a=audit_bank(seed);print(json.dumps(dict(seed=seed,status=a['status'],classification=category)),flush=True)


def summary():
    p=verify();rows=[];audits=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_laplace_rate.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows']
        for name,value in load_npz(d/'model.npz').items():arrays[f's{seed}_{name}']=value
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories=[a['classification'] for a in audits];success=categories.count('success');unresolved=categories.count('unresolved')
    decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive')
    primary=dict(decision=decision,certified_integral_exact_graphs=success,resolved_failures=32-success-unresolved,unresolved=unresolved,
                 success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    grouped=[]
    for arm in ARMS[:3]:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        grouped.append(dict(arm=arm,banks=32,exact_graphs=sum(r['exact_graph'] for r in part),missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part)))
    grouped.append(dict(arm=ARMS[3],banks=32,certified_integral_exact=success,certified_integral_nonexact=categories.count('resolved_failure_nonexact'),
                        certified_fractional=categories.count('resolved_failure_fractional'),unresolved=unresolved,
                        exact_integral_candidates=sum(bool(a['integral_candidate_graph'] and a['integral_candidate_graph']['exact_graph']) for a in audits)))
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);arrays['classification']=np.array(categories)
    np.savez_compressed(PUBLIC/'laplace_rate_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],supplied_prior=p['supplied_prior'],
                interpretation=p['interpretation'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),artifact_sha256=hashes,
                certificate_export_sha256=sha(PUBLIC/'laplace_rate_certificates.npz'),all_banks_retained=True,**p['query_accounting'],gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'laplace_rate.json',result);print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
