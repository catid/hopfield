"""H1be fixed feedback-prefix intervention on the examined H1bd streams."""
from datetime import datetime,timezone
import argparse,json
import numpy as np
from run import ROOT,HERE,sha,save
from frozen_feedback_policy import collect
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve
PARENT=ROOT/'artifacts/research/20261114/ripper/fresh_soft_feedback'
OUT=ROOT/'artifacts/research/20261114/ripper/frozen_feedback';PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-frozen-feedback.json';ARMS=['frozen','aligned']


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    for bank in (range(32) if full else ([] if seed is None else [seed])):
        for path,digest in p['bank_sha256'][str(bank)].items():assert sha(ROOT/path)==digest,path
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    gate=json.loads((HERE/'frozen-feedback-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    old=json.loads((HERE/'protocol-fresh-soft-feedback.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    names=['frozen_feedback_policy.py','frozen_feedback_study.py','audit_frozen_feedback.py','preflight_frozen_feedback.py','run_frozen_feedback.py',
           'soft_feedback_policy.py','audit_soft_feedback.py','laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py','run.py']
    inputs=[PARENT/'audit.json',PARENT/'source-audit.json',PARENT/'input_manifest.json',HERE/'protocol-fresh-soft-feedback.json',HERE/'frozen-feedback-preflight.json']
    p=dict(id='H1be-freeze-policy-feedback-after300',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,freeze_at=300,suffix_checkpoint=450,arms=ARMS,
           tie_seed_base=old['tie_seed_base'],noise_seed_base=old['noise_seed_base'],
           scope='Adaptive fixed-prefix mechanism test on the now-examined H1bd900row banks and exact archived mask-independent noise/tie streams. No new source IDs, latent draws, calibration or solver inputs. Not a fresh confirmation.',
           policy='Unchanged greedy weights(8/9)^actual_joint/(1+feedback). Feedback equals actual past equality counts for first300 masks; snapshot after row300 is immutable for masks301..900. Actual joint AND actual equality counts continue updating through900; final recovery uses actual complete counts, not snapshot. One fixed cutoff, no sweep.',
           predictability='Finalize actual-state and feedback digests and mask before accessing current source/noise; independent sequential replay. Future-suffix source/noise substitution from row450 must preserve first450 masks, labels, counts and frozen snapshot.',
           primary='At least29/32 certified integral exact frozen-feedback graphs. Supported ifS>=29; falsified ifS+U<29; otherwise inconclusive. Certified fractional is resolved procedure failure, no snapping. All32 retained.',
           mechanism_limit='Tests sufficiency of this first300-feedback intervention at fixed900queries and fixed target. Does not establish universal necessity of continuedfeedback, causal mediation, population success probability or query savings.',
           recovery='Unchanged all3240pair Laplace-rate degree20 LP and same-arm top810. Reverify32archived H1bdaligned proofs without refitting. All actual counts use all900rows.',
           supplied_prior=old['supplied_prior'],cost=old['cost'],certificate=old['certificate'],solver_options=old['solver_options'],
           query_accounting=dict(alternative_training=720000,alternative_calibration=0,prior_h1bd_union=1216156,prior_h1az_through_h1bd_sequence_union=3238105),
           physical_query_rule='Prior set perbank is union(H1bdaligned,H1bdpermuted). Newphysicalqueries=sum(frozen AND NOT prior). Reused=sum(frozen AND prior); reused+new=720000. Updated H1bdstreamunion=1216156+new; H1az-through-presentsequenceunion=3238105+new. No newcalibration; alternative budgets not additive.',
           budget=dict(aggregate_cpu_seconds=240,wall_seconds=120,max_cpu_workers=8,max_solver_calls=32,gpu_seconds=0,
                       scope='Preparation,32acquisitions/LPs,independent sequentialreplay,32suffix450gates,32new+32archivedproofs,accounting andsummary. Syntheticgate/freezeregistration/packaging separate.'),
           code_sha256={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(e['seed']):{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/e['completion_path']]+[PARENT/f"s{e['seed']}"/n for n in
               ['potential_noise.npz','aligned-observations.npz','permuted-observations.npz','aligned-fit.json','receipt.json','audit.json','model.npz']]} for e in manifest['inputs']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir()
    old=json.loads((PARENT/'audit.json').read_text());source=json.loads((PARENT/'source-audit.json').read_text())
    assert old['status']==source['status']=='validated' and old['protocol_sha256']==source['protocol_sha256']==sha(HERE/'protocol-fresh-soft-feedback.json')
    assert source['input_manifest_sha256']==sha(PARENT/'input_manifest.json') and old['source_audit']==source and old['physical_query_accounting']['new_union_queries']==1216156
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),parent_audit_sha256=sha(PARENT/'audit.json'),
         parent_source_audit_sha256=sha(PARENT/'source-audit.json'),input_manifest_sha256=sha(PARENT/'input_manifest.json'),scope=p['scope'],new_source_ids=0,new_latent_draws=0,new_solver_inputs=0))


def bank(seed):
    p=verify(seed);d=OUT/f's{seed}';d.mkdir(exist_ok=False);entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed
    clean=load_npz(ROOT/entry['completion_path'])['solutions'];noise=load_npz(PARENT/f's{seed}'/'potential_noise.npz')
    observed,events=collect(clean,noise['uniforms'],noise['permutations'],p['tie_seed_base']+seed,p['freeze_at'],p['suffix_checkpoint'])
    np.savez_compressed(d/'observations.npz',**observed);save(d/'events.json',events)
    m,c=observed['joint_count'],observed['equal_count'];cost=rate_costs(m,c);top=top_count_support(np.ones(3240,bool),cost);ii,jj=np.triu_indices(81,1)
    x,y,solver=solve(81,20,ii,jj,cost,seed,d/'highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
    else:cert=dict(certified=False,reason='solver_status_or_solution_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    if cert['certified'] and cert['integral']:
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
    elif cert['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    arrays={'frozen_'+k:v for k,v in dict(joint_count=m,equal_count=c,cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness).items()}
    arrays['frozen_mask']=observed['given'];arrays['feedback_equal_count']=observed['feedback_equal_count']
    baseline=load_npz(PARENT/f's{seed}'/'model.npz');arrays.update({k:v for k,v in baseline.items() if k.startswith('aligned_')});arrays['permuted_mask']=baseline['permuted_mask']
    previous=baseline['aligned_mask']|baseline['permuted_mask'];given=observed['given']
    for arm in ['aligned','permuted']:
        old=load_npz(PARENT/f's{seed}'/f'{arm}-observations.npz');overlap=old['given']&given;assert np.array_equal(old['values'][overlap],observed['values'][overlap])
    accounting=dict(frozen_queries=int(given.sum()),reused_queries=int((given&previous).sum()),new_queries=int((given&~previous).sum()),prior_h1bd_union=int(previous.sum()),combined_h1bd_union=int((given|previous).sum()),new_calibration_queries=0)
    np.savez_compressed(d/'model.npz',**arrays);save(d/'fit.json',dict(solver=solver,certificate=cert,classification=category))
    save(d/'receipt.json',dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),source_sha256=entry['completion_sha256'],noise_sha256=sha(PARENT/f's{seed}'/'potential_noise.npz'),
         query_accounting=accounting,artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()}))
    from audit_frozen_feedback import audit_bank
    audit_bank(seed,(clean,noise))


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['seed']==seed and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_frozen_feedback.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows'];arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'model.npz').items()})
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories={arm:[a['arm_audits'][arm]['classification'] for a in audits] for arm in ARMS};success=categories['frozen'].count('success');unresolved=categories['frozen'].count('unresolved')
    primary=dict(decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive'),success_lower=success,success_upper=success+unresolved,required=29,unresolved=unresolved)
    grouped=[]
    for arm in ARMS:
        top=[r for r in rows if r['arm']==arm+'_top810'];grouped.append(dict(arm=arm+'_top810',exact_graphs=sum(r['exact_graph'] for r in top),missing_true_total=sum(r['missing_true_edges'] for r in top),extra_total=sum(r['extra_edges'] for r in top),banks=32))
        cs=categories[arm];grouped.append(dict(arm=arm,certified_integral_exact=cs.count('success'),certified_integral_nonexact=cs.count('resolved_failure_nonexact'),certified_fractional=cs.count('resolved_failure_fractional'),unresolved=cs.count('unresolved'),banks=32))
    accounting={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert accounting['frozen_queries']==accounting['reused_queries']+accounting['new_queries']==720000 and accounting['prior_h1bd_union']==1216156
    assert accounting['combined_h1bd_union']==1216156+accounting['new_queries'];accounting['h1az_through_present_sequence_union']=3238105+accounting['new_queries']
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);np.savez_compressed(PUBLIC/'frozen_feedback_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),physical_query_accounting=accounting,alternative_query_accounting=p['query_accounting'],physical_query_rule=p['physical_query_rule'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'frozen_feedback_certificates.npz'),all_banks_retained=True,suffix_substitution_gates=32,new_source_ids=0,new_latent_draws=0,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'frozen_feedback.json',result);print(json.dumps(dict(primary=primary,grouped=grouped,physical_query_accounting=accounting)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
