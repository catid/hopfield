"""H1bc aligned versus cell-permuted soft equality feedback at fixed N900."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
from run import ROOT,HERE,sha,save
from soft_feedback_policy import collect
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve
PARENT=ROOT/'artifacts/research/20261114/ripper/fresh_rate'
UNIFORM=ROOT/'artifacts/research/20261114/ripper/uniform_mask'
BALANCED=ROOT/'artifacts/research/20261114/ripper/balanced_mask'
OUT=ROOT/'artifacts/research/20261114/ripper/soft_feedback';PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-soft-feedback.json';ARMS=['aligned','permuted']


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
    gate=json.loads((HERE/'soft-feedback-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    old=json.loads((HERE/'protocol-fresh-rate.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    source=['soft_feedback_policy.py','soft_feedback_study.py','audit_soft_feedback.py','preflight_soft_feedback.py','run_soft_feedback.py',
            'laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py','run.py']
    inputs=[PARENT/'audit.json',PARENT/'source-audit.json',PARENT/'input_manifest.json',UNIFORM/'audit.json',BALANCED/'audit.json',
            HERE/'protocol-fresh-rate.json',HERE/'protocol-uniform-mask.json',HERE/'protocol-balanced-mask.json',HERE/'soft-feedback-preflight.json']
    permutations=[np.random.default_rng(202611800000+s).permutation(81).tolist() for s in range(32)]
    p=dict(id='H1bc-soft-feedback-alignment-control',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,arms=ARMS,fit_order=ARMS,
           tie_seed_base=202611790000,feedback_permutation_seed_base=202611800000,feedback_permutations=permutations,suffix_checkpoint=450,
           scope='Adaptive mechanism follow-up on already-examined H1az source identities and same archived mask-independent noise. No fresh identities/latents or distribution-generalization claim.',
           policy='Greedy pair weights(8/9)^actual_joint_count/(1+feedback_count), exponent1. Aligned feedback is actual past equality matrix; control uses actual past equality matrix[fixed_permutation,fixed_permutation]. Each arm has its own actual observed history. No hard exclusions, calibration or suppliednoiseprobabilities. Both use H1ba/H1bb tie permutations. Final recovery always uses true-coordinate observed counts.',
           predictability='At eachrow hash prior actual joint/equality state, compute/finalize mask using only thatstate/tieorder/frozenfeedbackpermutation, then access current source/noiselatent. Immutable-state checks and independent sequential mask/noise/state replay. Alter sources+noise only atrows>=450 and rerun; first450 masks/labels/counts must remain identical.',
           primary='Endpoint1 procedure: aligned>=29/32 certified integral exact. Success bounds S_aligned to S_aligned+U_aligned. Supported if lower>=29; falsified if upper<29; otherwise inconclusive.',
           alignment_endpoint='Endpoint2 finite alignment advantage: aligned-minus-permuted exact-bank count>=4/32. Lower=S_aligned-(S_permuted+U_permuted); upper=S_aligned+U_aligned-S_permuted. Supported if lower>=4; falsified if upper<4; otherwise inconclusive. Report endpoints separately, no combined success claim.',
           mechanism_limit='Cellpermutation changes feedback location and trajectory, including exposure/magnitude pathways. Endpoint2 is a procedure-level alignment intervention, not proof of a unique causal mediator. Endpoint1 alone establishes no alignment mechanism. No posthoc exponent/mask tuning.',
           recovery='Unchanged all3240-pair Laplace-rate degree20 LP; same-arm top810 controls. Reverify archived adaptive and balanced LP proofs without refitting. Fractional returnedoptima resolved procedure failures, not proof of no integral co-optimum. No snapping or uniqueness claim.',
           supplied_prior=old['supplied_prior'],cost=old['cost'],certificate=old['certificate'],solver_options=old['solver_options'],
           query_accounting=dict(alternative_each_arm_training=720000,alternative_each_arm_calibration=0,prior_cumulative_union=1619152),
           physical_query_rule='Previous=union(adaptive,uniform,balanced masks). For eachnewarm count previous-overlap and new relativeprevious; physical increment is union(aligned,permuted) minus previous, counted once. Combined cumulative union=1619152+increment. Marginal arm budgets720000 each are alternatives, not additive physical spend.',
           budget=dict(aggregate_cpu_seconds=240,wall_seconds=120,max_cpu_workers=8,max_solver_calls=64,gpu_seconds=0,
                       scope='Preparation,64acquisitions/LPs, independent sequential replays,64future-suffix gates,new+archivedproofs,query accounting/summary. Synthetic gate and packaging separate.'),
           code_sha256={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(e['seed']):{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/e['training_path']]+
               [PARENT/f"s{e['seed']}"/n for n in ['potential_noise.npz','observations.npz','recovery.npz','recovery.json','recovery-audit.json']]+
               [UNIFORM/f"s{e['seed']}"/'observations.npz']+[BALANCED/f"s{e['seed']}"/n for n in ['observations.npz','model.npz','receipt.json','audit.json']]} for e in manifest['inputs']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir();audits={}
    for name,path,protocol in [('adaptive',PARENT,'fresh-rate'),('uniform',UNIFORM,'uniform-mask'),('balanced',BALANCED,'balanced-mask')]:
        a=json.loads((path/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(HERE/f'protocol-{protocol}.json');audits[name]=sha(path/'audit.json')
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),parent_audit_sha256=audits,scope=p['scope'],new_source_ids=0,new_latent_draws=0,new_solver_inputs=0))


def bank(seed):
    p=verify(seed);d=OUT/f's{seed}';d.mkdir(exist_ok=False);entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed
    clean=load_npz(ROOT/entry['training_path'])['solutions'];noise=load_npz(PARENT/f's{seed}'/'potential_noise.npz');permutation=np.array(p['feedback_permutations'][seed],np.int64)
    arrays={};newmasks={};observations={};ii,jj=np.triu_indices(81,1)
    for arm in p['fit_order']:
        observed,events=collect(clean,noise['uniforms'],noise['permutations'],p['tie_seed_base']+seed,permutation if arm=='permuted' else None,p['suffix_checkpoint'])
        np.savez_compressed(d/f'{arm}-observations.npz',**observed);save(d/f'{arm}-events.json',events)
        newmasks[arm]=observed['given'];observations[arm]=observed
        m,c=observed['joint_count'],observed['equal_count'];cost=rate_costs(m,c);top=top_count_support(np.ones(3240,bool),cost)
        x,y,solver=solve(81,20,ii,jj,cost,seed,d/f'{arm}-highs.log')
        if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
        else:
            cert=dict(certified=False,reason='solver_status_or_solution_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
        if cert['certified'] and cert['integral']:
            truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
        elif cert['certified']:category='resolved_failure_fractional'
        else:category='unresolved'
        arrays.update({arm+'_'+k:v for k,v in dict(joint_count=m,equal_count=c,cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness).items()})
        save(d/f'{arm}-fit.json',dict(solver=solver,certificate=cert,classification=category))
    # Comparators and physical-overlap accounting load only after both new fits.
    adaptive=load_npz(PARENT/f's{seed}'/'recovery.npz');balanced=load_npz(BALANCED/f's{seed}'/'model.npz')
    keys=['joint_count','equal_count','cost','primal_values','row_dual_values','primal_twice','node_dual_twice','upper_dual_twice']
    for arm,model in [('adaptive',adaptive),('balanced',balanced)]:arrays.update({arm+'_'+k:model[k] for k in keys})
    previous=np.zeros((900,81),bool)
    for path in [PARENT,UNIFORM,BALANCED]:
        with np.load(path/f's{seed}'/'observations.npz',allow_pickle=False) as z:given=z['given'].copy();values=z['values'].copy()
        for arm in ARMS:
            overlap=given&newmasks[arm];assert np.array_equal(values[overlap],observations[arm]['values'][overlap])
        previous|=given
    shared=newmasks['aligned']&newmasks['permuted'];assert np.array_equal(observations['aligned']['values'][shared],observations['permuted']['values'][shared])
    accounting=dict(prior_union_queries=int(previous.sum())+2000,new_union_queries=int(((newmasks['aligned']|newmasks['permuted'])&~previous).sum()),
                    aligned_reused=int((newmasks['aligned']&previous).sum()),aligned_new_relative_prior=int((newmasks['aligned']&~previous).sum()),
                    permuted_reused=int((newmasks['permuted']&previous).sum()),permuted_new_relative_prior=int((newmasks['permuted']&~previous).sum()),
                    new_shared_between_arms=int((shared&~previous).sum()),new_calibration_queries=0)
    accounting['combined_union_queries']=accounting['prior_union_queries']+accounting['new_union_queries']
    arrays['previous_mask_union']=previous;arrays['aligned_mask']=newmasks['aligned'];arrays['permuted_mask']=newmasks['permuted'];np.savez_compressed(d/'model.npz',**arrays)
    receipt=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),source_sha256=entry['training_sha256'],
                 noise_sha256=sha(PARENT/f's{seed}'/'potential_noise.npz'),query_accounting=accounting,artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()})
    save(d/'receipt.json',receipt)
    from audit_soft_feedback import audit_bank
    a=audit_bank(seed,(clean,noise));print(json.dumps(dict(seed=seed,status=a['status'],classifications={k:v['classification'] for k,v in a['arm_audits'].items()})),flush=True)


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_soft_feedback.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows'];arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'model.npz').items()})
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories={arm:[a['arm_audits'][arm]['classification'] for a in audits] for arm in ARMS};sa,sp=[categories[arm].count('success') for arm in ARMS];ua,up=[categories[arm].count('unresolved') for arm in ARMS]
    procedure=dict(decision='supported' if sa>=29 else ('falsified' if sa+ua<29 else 'inconclusive'),success_lower=sa,success_upper=sa+ua,required=29,unresolved=ua)
    low=sa-sp-up;high=sa+ua-sp;alignment=dict(decision='supported' if low>=4 else ('falsified' if high<4 else 'inconclusive'),difference_lower=low,difference_upper=high,required=4,aligned_success=sa,permuted_success=sp,aligned_unresolved=ua,permuted_unresolved=up)
    grouped=[]
    for arm in ['adaptive','balanced']:
        part=[r for r in rows if r['arm']==arm];grouped.append(dict(arm=arm,archived=True,exact_graphs=sum(r['exact_graph'] for r in part),missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part),banks=32))
    for arm in ARMS:
        top=[r for r in rows if r['arm']==arm+'_top810'];grouped.append(dict(arm=arm+'_top810',exact_graphs=sum(r['exact_graph'] for r in top),missing_true_total=sum(r['missing_true_edges'] for r in top),extra_total=sum(r['extra_edges'] for r in top),banks=32))
        cs=categories[arm];grouped.append(dict(arm=arm,certified_integral_exact=cs.count('success'),certified_integral_nonexact=cs.count('resolved_failure_nonexact'),certified_fractional=cs.count('resolved_failure_fractional'),unresolved=cs.count('unresolved'),banks=32))
    accounting={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};assert accounting['prior_union_queries']==1619152
    assert accounting['combined_union_queries']==1619152+accounting['new_union_queries']
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);np.savez_compressed(PUBLIC/'soft_feedback_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=procedure,alignment_endpoint=alignment,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),physical_query_accounting=accounting,alternative_query_accounting=p['query_accounting'],physical_query_rule=p['physical_query_rule'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'soft_feedback_certificates.npz'),all_banks_retained=True,suffix_substitution_gates=64,new_source_ids=0,new_latent_draws=0,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'soft_feedback.json',result);print(json.dumps(dict(primary=procedure,alignment_endpoint=alignment,grouped=grouped,physical_query_accounting=accounting)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
