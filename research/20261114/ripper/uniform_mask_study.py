"""H1ba: paired uniform acquisition without calibration on fixed H1az latents."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
from run import ROOT,HERE,sha,save
from uniform_mask_features import masks
from prefix_rate_features import prefix_counts
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve
PARENT=ROOT/'artifacts/research/20261114/ripper/fresh_rate'
OUT=ROOT/'artifacts/research/20261114/ripper/uniform_mask';PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-uniform-mask.json';ARMS=['adaptive_degree20','uniform_top810','uniform_degree20']


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
    gate=json.loads((HERE/'uniform-mask-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    old=json.loads((HERE/'protocol-fresh-rate.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    source=['uniform_mask_features.py','uniform_mask_study.py','audit_uniform_mask.py','preflight_uniform_mask.py','run_uniform_mask.py',
            'prefix_rate_features.py','laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py','run.py']
    inputs=[PARENT/'audit.json',PARENT/'source-audit.json',PARENT/'input_manifest.json',HERE/'protocol-fresh-rate.json',HERE/'uniform-mask-preflight.json']
    p=dict(id='H1ba-uniform-masks-zero-calibration-acquisition',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,
           mask_seed_base=202611790000,arms=ARMS,
           scope='Paired acquisition intervention on fixed already-examined H1az source identities and archived mask-independent corruption latents. No fresh identities, latent draws or distribution-generalization claim.',
           mask_rule='Independent NumPy Generator(202611790000+bank), eachrow permutation81 first25. Generate all900 masks and archive them before loading selected current source labels or archived noise latents. No calibration, LR state, past labels or geometry enters masks. Load archived adaptive masks only after the new LP fit for overlap accounting.',
           common_noise='Reconstruct the exact same full corrupted board for each H1az row using its clean solution and archived uniform/position permutation. These latents came from a stream independent of acquisition masks. Paired reused queried cells must have identical labels; newly exposed cells are the only incremental physical answers.',
           primary='At least29/32 independently certified integral exact uniform-mask graphs. Supported if successes>=29; falsified if successes+unresolved<29; otherwise inconclusive. All32 retained, no masks or sources replaced.',
           recovery='Unchanged all3240-pair Laplace-rate degree20 LP, with same uniform-count top810 control. Archived adaptive degree20 outputs/costs/proofs independently reverified, without refitting.',
           supplied_prior=old['supplied_prior'],cost=old['cost'],certificate=old['certificate'],solver_options=old['solver_options'],
           query_accounting=dict(uniform_training_per_bank=22500,uniform_calibration_per_bank=0,uniform_alternative_total=720000,
                                 adaptive_training_total=720000,adaptive_calibration_total=64000,adaptive_alternative_total=784000),
           physical_query_rule='Count reused=sum(uniform_mask AND adaptive_mask), newly_exposed=sum(uniform_mask AND NOT adaptive_mask). Reused+newly_exposed=720000; only newly_exposed are new physical queries in paired replay. Combined union including oldcalibration=784000+newly_exposed. Alternative deployment budgets are not additive.',
           budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,max_solver_calls=32,gpu_seconds=0,
                       scope='Prepare,32uniform acquisitions/fits, independent mask/common-noise/count replays and exact new+archived proofs, accounting and summary. Synthetic gate and packaging separate.'),
           code_sha256={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(e['seed']):{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/e['training_path']]+[PARENT/f"s{e['seed']}"/n for n in
               ['potential_noise.npz','observations.npz','receipt.json','audit.json','recovery.npz','recovery.json','recovery-audit.json']]} for e in manifest['inputs']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir()
    old=json.loads((PARENT/'audit.json').read_text());source=json.loads((PARENT/'source-audit.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    assert old['status']==source['status']=='validated' and old['protocol_sha256']==source['protocol_sha256']==sha(HERE/'protocol-fresh-rate.json')
    assert source['input_manifest_sha256']==sha(PARENT/'input_manifest.json') and old['source_audit']==source
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),parent_audit_sha256=sha(PARENT/'audit.json'),
         parent_source_audit_sha256=sha(PARENT/'source-audit.json'),input_manifest_sha256=sha(PARENT/'input_manifest.json'),scope=p['scope'],new_source_ids=0,new_latent_draws=0,new_solver_inputs=0))


def bank(seed):
    p=verify(seed);d=OUT/f's{seed}';d.mkdir(exist_ok=False)
    given=masks(p['mask_seed_base']+seed,p['rows']);np.savez_compressed(d/'masks.npz',given=given)
    mask_digest=sha(d/'masks.npz')
    # All masks exist before any selected labels or archived corruption arrays load.
    manifest=json.loads((PARENT/'input_manifest.json').read_text());e=manifest['inputs'][seed];assert e['seed']==seed
    clean=load_npz(ROOT/e['training_path'])['solutions'];noise=load_npz(PARENT/f's{seed}'/'potential_noise.npz')
    values=np.full((900,81),-1,np.int8)
    for row in range(900):
        observed=clean[row,noise['permutations'][row]] if noise['uniforms'][row]<9/160 else clean[row]
        values[row,given[row]]=observed[given[row]]
    m,c=prefix_counts(given,values,[900])[900];cost=rate_costs(m,c);top=top_count_support(np.ones(3240,bool),cost);ii,jj=np.triu_indices(81,1)
    x,y,solver=solve(81,20,ii,jj,cost,seed,d/'highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
    else:
        cert=dict(certified=False,reason='solver_status_or_solution_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    if cert['certified'] and cert['integral']:
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
    elif cert['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    arrays=dict(joint_count=m,equal_count=c,cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness)
    # Archived adaptive recovery and masks are strictly late comparators.
    baseline=load_npz(PARENT/f's{seed}'/'recovery.npz');arrays.update({'baseline_'+k:v for k,v in baseline.items()})
    with np.load(PARENT/f's{seed}'/'observations.npz',allow_pickle=False) as z:oldgiven=z['given'].copy();oldvalues=z['values'].copy()
    overlap=given&oldgiven;new=given&~oldgiven;assert np.array_equal(values[overlap],oldvalues[overlap])
    accounting=dict(reused_queries=int(overlap.sum()),newly_exposed_queries=int(new.sum()),uniform_training_queries=int(given.sum()),new_calibration_queries=0,
                    prior_adaptive_total_queries=24500,combined_union_queries=24500+int(new.sum()))
    np.savez_compressed(d/'observations.npz',given=given,values=values);np.savez_compressed(d/'model.npz',**arrays)
    receipt=dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),mask_sha256_before_labels=mask_digest,
                 mask_seed=p['mask_seed_base']+seed,source_sha256=e['training_sha256'],noise_sha256=sha(PARENT/f's{seed}'/'potential_noise.npz'),baseline_sha256=sha(PARENT/f's{seed}'/'recovery.npz'),
                 solver=solver,certificate=cert,classification=category,query_accounting=accounting,artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()})
    save(d/'receipt.json',receipt)
    from audit_uniform_mask import audit_bank
    a=audit_bank(seed);print(json.dumps(dict(seed=seed,status=a['status'],classification=a['classification'])),flush=True)


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_uniform_mask.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows'];arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'model.npz').items()})
        arrays[f's{seed}_uniform_mask']=load_npz(d/'masks.npz')['given']
        with np.load(PARENT/f's{seed}'/'observations.npz',allow_pickle=False) as z:arrays[f's{seed}_adaptive_mask']=z['given'].copy()
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories=[a['classification'] for a in audits];success=categories.count('success');unresolved=categories.count('unresolved')
    primary=dict(decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive'),certified_integral_exact_graphs=success,
                 resolved_failures=32-success-unresolved,unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    grouped=[]
    for arm in ARMS[:2]:
        part=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,exact_graphs=sum(r['exact_graph'] for r in part),missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part),banks=32))
    grouped.append(dict(arm=ARMS[2],certified_integral_exact=success,certified_integral_nonexact=categories.count('resolved_failure_nonexact'),certified_fractional=categories.count('resolved_failure_fractional'),unresolved=unresolved,banks=32))
    accounting={key:sum(a['query_accounting'][key] for a in audits) for key in audits[0]['query_accounting']}
    assert accounting['reused_queries']+accounting['newly_exposed_queries']==accounting['uniform_training_queries']==720000
    assert accounting['combined_union_queries']==784000+accounting['newly_exposed_queries']
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);np.savez_compressed(PUBLIC/'uniform_mask_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),physical_query_accounting=accounting,alternative_query_accounting=p['query_accounting'],physical_query_rule=p['physical_query_rule'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'uniform_mask_certificates.npz'),all_banks_retained=True,new_source_ids=0,new_latent_draws=0,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'uniform_mask.json',result);print(json.dumps(dict(primary=primary,grouped=grouped,physical_query_accounting=accounting)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
