"""H1ax: fixed-prefix query budget of smoothed-rate degree20 recovery."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
from prefix_rate_features import prefix_counts
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/permuted_noise'
BASELINE=ROOT/'artifacts/research/20261114/ripper/laplace_rate'
OUT=ROOT/'artifacts/research/20261114/ripper/prefix_rate'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-prefix-rate.json'
PREFIXES=[100,300,600];ARMS=['top810','degree20']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {name:z[name].copy() for name in z.files}


def observations(seed):
    # Materialize only the two needed arrays once; do not decompress archived log-LRs.
    with np.load(PARENT/f's{seed}'/'observations.npz',allow_pickle=False) as z:
        given=z['given'].copy();values=z['values'].copy()
    assert given.shape==values.shape==(1110,81) and given.dtype==bool and values.dtype==np.int8
    given.setflags(write=False);values.setflags(write=False)
    return given,values


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    for bank in (range(32) if full else ([] if seed is None else [seed])):
        for path,digest in p['bank_sha256'][str(bank)].items():assert sha(ROOT/path)==digest,path
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    pre=json.loads((HERE/'prefix-rate-preflight.json').read_text());assert pre['status']=='validated_synthetic_only'
    for name,digest in pre['code_sha256'].items():assert sha(HERE/name)==digest,name
    previous=json.loads((HERE/'protocol-laplace-rate.json').read_text())
    source=[HERE/name for name in ['prefix_rate_features.py','prefix_rate_study.py','audit_prefix_rate.py','preflight_prefix_rate.py','run_prefix_rate.py',
                                  'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json',HERE/'audit_permuted_noise_acquisition.py',
            BASELINE/'audit.json',HERE/'protocol-laplace-rate.json',HERE/'prefix-rate-preflight.json']
    p=dict(id='H1ax-prefix-smoothed-rate-query-budget',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),
           prefixes=PREFIXES,fit_order=[600,300,100],primary_rows=600,archived_rows=1110,arms=ARMS,
           scope='Fixed first100/300/600 rows of all32 predictable H1ap observed streams; inherited1110 H1aw proof/control reverified, not refitted. Previously examined source banks; no new queries, acquisition or latent draws.',
           supplied_prior=previous['supplied_prior'],cost=previous['cost'],objective=previous['objective'],
           tie_convention=previous['tie_convention'],certificate=previous['certificate'],certificate_rationale=previous['certificate_rationale'],
           outcome_rules=previous['outcome_rules'],
           primary='N600: >=29/32 independently certified integral exact graphs. Supported if successes>=29; falsified only if successes+unresolved<29; otherwise inconclusive. N100/N300 and top810 controls descriptive; all32 retained at each prefix.',
           leakage='Eager-load only given/values arrays once perbank. Feature builder cumulatively reads rows0..599 and copies independent snapshots at100/300/600. Each LP receives only its snapshot costs, all3240 candidates and fixed degree20. No final supports, later observations, future-derived costs, warm starts or later-prefix outputs enter earlier fits. Archived1110 model is loaded only after new fits. Independent pair-vector recount plus future-suffix substitution for each prefix must leave counts/costs unchanged.',
           predictability='H1ap current masks were finalized before current labels. Reused prefix masks remain predictable. Their acquisition used the historical calibration; this is not end-to-end calibration-free acquisition, only q0/q1-free recovery costs.',
           interpretation='Finite budget-prefix mechanism on nested already-seen streams. Prefixes are dependent, not independent replications; shorter labels are literal subsets. No multiple-prefix primary, fresh-source or universal-noise claim.',
           solver_options=previous['solver_options'],solver_version=previous['solver_version'],numpy_version=previous['numpy_version'],
           budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,max_solver_calls=96,gpu_seconds=0,
                       scope='Preparation, three fits perbank, independent recount/suffix gates/certificates, archived proof checks and summary. Each actualLP time_limit1s; eight one-thread workers. Synthetic preflight and packaging separate.'),
           query_accounting=[dict(rows=n,training_per_bank=25*n,training_total=32*25*n,shared_calibration_per_bank=2000,
                                  shared_calibration_total=64000,total_historical_acquisition_per_bank=25*n+2000,
                                  total_historical_acquisition_all_banks=32*(25*n+2000)) for n in [100,300,600,1110]],
           query_scope='Nested historical observations; prefix totals are alternative budgets, not additive new query spend. Shared calibration charged once perbank per hypothetical truncated acquisition. No new queries/source rows/solver puzzle inputs.',
           code_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(seed):{str(f.relative_to(ROOT)):sha(f) for f in
             [PARENT/f's{seed}'/name for name in ['observations.npz','receipt.json','audit.json','events.json']]+
             [BASELINE/f's{seed}'/name for name in ['model.npz','receipt.json','audit.json']]} for seed in range(32)})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'audit.json').read_text());baseline=json.loads((BASELINE/'audit.json').read_text())
    assert parent['status']==baseline['status']=='validated'
    assert parent['protocol_sha256']==sha(HERE/'protocol-permuted-noise.json') and baseline['protocol_sha256']==sha(HERE/'protocol-laplace-rate.json')
    rows=[]
    for seed in p['seeds']:
        d=PARENT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text());events=json.loads((d/'events.json').read_text())
        assert a['status']=='validated' and a['receipt_sha256']==sha(d/'receipt.json') and a['auditor_sha256']==sha(HERE/'audit_permuted_noise_acquisition.py')
        assert r['queries']==29750 and r['mask_finalized_before_current_noise']==1110
        assert r['artifact_sha256']['observations.npz']==sha(d/'observations.npz') and r['artifact_sha256']['events.json']==sha(d/'events.json')
        assert len(events)==1110
        for row,event in enumerate(events[:600]):assert event['row']==row and event['mask_finalized_before_current_corruption'] is True
        br=json.loads((BASELINE/f's{seed}'/'receipt.json').read_text());ba=json.loads((BASELINE/f's{seed}'/'audit.json').read_text())
        assert ba['receipt_sha256']==sha(BASELINE/f's{seed}'/'receipt.json') and ba['model_sha256']==sha(BASELINE/f's{seed}'/'model.npz')
        assert br['artifact_sha256']['model.npz']==sha(BASELINE/f's{seed}'/'model.npz')
        rows.append(dict(seed=seed,observations_sha256=sha(d/'observations.npz'),parent_audit_sha256=sha(d/'audit.json'),baseline_model_sha256=sha(BASELINE/f's{seed}'/'model.npz'),predictable_prefix_rows=600))
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),banks=rows,new_queries=0))


def bank(seed):
    start=time.monotonic();p=verify(seed);d=OUT/f's{seed}';d.mkdir(exist_ok=False);given,values=observations(seed)
    snapshots=prefix_counts(given,values,p['prefixes']);arrays={};records={};ii,jj=np.triu_indices(81,1)
    for n in p['fit_order']:
        m,c=snapshots[n];cost=rate_costs(m,c);top=top_count_support(np.ones(3240,bool),cost)
        x,y,solver=solve(81,20,ii,jj,cost,seed,d/f'n{n}-highs.log')
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
            certificate,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
        else:
            certificate=dict(certified=False,reason='solver_status_or_solution_validity')
            witness={name:np.empty(0,np.int64) for name in ['primal_twice','node_dual_twice','upper_dual_twice']}
        if certificate['certified'] and certificate['integral']:
            truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
            category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
        elif certificate['certified']:category='resolved_failure_fractional'
        else:category='unresolved'
        model=dict(joint_count=m,equal_count=c,cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness)
        arrays.update({f'n{n}_'+name:value for name,value in model.items()})
        record=dict(rows=n,solver=solver,certificate=certificate,classification=category)
        save(d/f'n{n}-fit.json',record);records[n]=record
    # Late comparator load: full-stream coefficients cannot enter any earlier fit.
    baseline=load_npz(BASELINE/f's{seed}'/'model.npz')
    for name in ['joint_count','equal_count','cost','top810_support','primal_values','row_dual_values','primal_twice','node_dual_twice','upper_dual_twice']:
        arrays['baseline_'+name]=baseline[name]
    np.savez_compressed(d/'model.npz',**arrays)
    receipt=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),wall_s=time.monotonic()-start,
                 observations_sha256=sha(PARENT/f's{seed}'/'observations.npz'),baseline_model_sha256=sha(BASELINE/f's{seed}'/'model.npz'),
                 artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()})
    save(d/'receipt.json',receipt)
    from audit_prefix_rate import audit_bank
    a=audit_bank(seed,(given,values));print(json.dumps(dict(seed=seed,status=a['status'],primary_classification=a['prefix_audits']['600']['classification'])),flush=True)


def summary():
    p=verify();audits=[];arrays={};hashes={};rows=[]
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
        assert a['auditor_sha256']==sha(HERE/'audit_prefix_rate.py') and a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows']
        for name,value in load_npz(d/'model.npz').items():arrays[f's{seed}_'+name]=value
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    grouped=[];primary=None
    for n in p['prefixes']:
        chosen=[a['prefix_audits'][str(n)] for a in audits];categories=[a['classification'] for a in chosen]
        success=categories.count('success');unresolved=categories.count('unresolved');top=[r for r in rows if r['rows']==n and r['arm']=='top810']
        grouped.append(dict(rows=n,arm='top810',exact_graphs=sum(r['exact_graph'] for r in top),missing_true_total=sum(r['missing_true_edges'] for r in top),extra_total=sum(r['extra_edges'] for r in top),banks=32))
        grouped.append(dict(rows=n,arm='degree20',certified_integral_exact=success,certified_integral_nonexact=categories.count('resolved_failure_nonexact'),
                            certified_fractional=categories.count('resolved_failure_fractional'),unresolved=unresolved,banks=32,
                            exact_integral_candidates=sum(bool(a['integral_candidate_graph'] and a['integral_candidate_graph']['exact_graph']) for a in chosen)))
        if n==600:
            decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive')
            primary=dict(rows=600,decision=decision,certified_integral_exact_graphs=success,resolved_failures=32-success-unresolved,
                         unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    for arm in ARMS:
        part=[r for r in rows if r['rows']==1110 and r['arm']==arm]
        grouped.append(dict(rows=1110,arm=arm,archived=True,exact_graphs=sum(r['exact_graph'] for r in part),
                            missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part),banks=32))
    arrays['banks']=np.arange(32);arrays['prefixes']=np.array(PREFIXES,np.int64);arrays['arms']=np.array(ARMS)
    np.savez_compressed(PUBLIC/'prefix_rate_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],supplied_prior=p['supplied_prior'],
                primary=primary,grouped=grouped,rows=rows,bank_audits=audits,source_audit=json.loads((OUT/'source-audit.json').read_text()),
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'prefix_rate_certificates.npz'),all_banks_retained=True,
                query_accounting=p['query_accounting'],query_scope=p['query_scope'],new_queries=0,new_calibrations=0,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0,
                suffix_substitution_gates=96,independent_prefix_pair_states_checked=32*1110*3240)
    save(OUT/'audit.json',result);save(PUBLIC/'prefix_rate.json',result);print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
