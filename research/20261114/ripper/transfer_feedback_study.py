"""H1bf transfer another bank's300-row snapshot without recipient feedback."""
from datetime import datetime,timezone
import argparse,json
import numpy as np
from run import ROOT,HERE,sha,save
from transfer_feedback_policy import donor_map,masks,observe
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve
PARENT=ROOT/'artifacts/research/20261114/ripper/fresh_soft_feedback'
SNAPSHOT=ROOT/'artifacts/research/20261114/ripper/frozen_feedback'
OUT=ROOT/'artifacts/research/20261114/ripper/transfer_feedback';PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-transfer-feedback.json';ARMS=['transfer','frozen']


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify(seed=None,full=False):
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    banks=range(32) if full else ([] if seed is None else sorted({seed,p['donor_map'][seed]}))
    for bank in banks:
        for path,digest in p['bank_sha256'][str(bank)].items():assert sha(ROOT/path)==digest,path
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    gate=json.loads((HERE/'transfer-feedback-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    old=json.loads((HERE/'protocol-fresh-soft-feedback.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    ids=load_npz(PARENT/'selected_sources.npz')['source_rows'];mapping=donor_map(ids,300)
    if mapping is None:
        save(HERE/'transfer-feedback-registration-failure.json',dict(status='incomplete',reason='No source-disjoint derangement; rule not relaxed.'));raise RuntimeError('No valid donor assignment')
    names=['transfer_feedback_policy.py','transfer_feedback_study.py','audit_transfer_feedback.py','preflight_transfer_feedback.py','run_transfer_feedback.py',
           'soft_feedback_policy.py','audit_soft_feedback.py','prefix_rate_features.py','laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py','run.py']
    inputs=[PARENT/'audit.json',PARENT/'source-audit.json',PARENT/'input_manifest.json',PARENT/'selected_sources.npz',SNAPSHOT/'audit.json',
            HERE/'protocol-fresh-soft-feedback.json',HERE/'protocol-frozen-feedback.json',HERE/'transfer-feedback-preflight.json']
    p=dict(id='H1bf-transferred300row-feedback',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,donor_prefix=300,donor_map=mapping.tolist(),arms=ARMS,
           tie_seed_base=old['tie_seed_base'],noise_seed_base=old['noise_seed_base'],
           scope='One fixed transfer intervention on examined H1bd source/noise/tie streams. No new source IDs, latent draws, calibration or puzzle-solver inputs. Shared indexed topology; donor coupling prevents independent-bank population inference.',
           donor_rule='Deterministic augmenting-path matching: recipients ascending; donor choices ascending; forbid self and any intersection of donor first300 sourceIDs with recipient all900IDs. Exactly one donor per recipient and each donor used once. Mapping frozen before any recipient masks. No outcome-aware selection; no valid mapping is incomplete.',
           donor_feedback='Donor snapshot from archived aligned first300 observed equality counts, identical H1be snapshot. Reconstruct rawlabels independently. Symmetric81matrix with zero unused diagonal; immutable through recipient acquisition.',
           policy='Generate and persist all900 masks from fixed donorfeedback, actual recipientcoquery counts and archivedrecipient tie seed before loading recipient sourceboards or latentarrays. Greedy weight(8/9)^joint/(1+donorfeedback). No recipientlabel feedback at any point. Final recovery uses all900 actual queriedlabels.',
           predictability='Independent mask replay and complete recipient-label replacement gate: every mask unchanged while actual final equality counts change. All900 masks persisted before recipient labels/latents read.',
           primary='At least29/32 certified integral exact transfer graphs. Supported ifS>=29; falsified ifS+U<29; otherwise inconclusive. Fractional is resolvedfailure; unresolved retained; no rounding, donor/cutoff sweep or replacement.',
           mechanism_limit='Transfer from distinct source rows under common indexed graph. Not zero-cost prior, new-source confirmation, population generalization or universal transfer sufficiency. Archived same-bank prefix29/32 descriptive only.',
           recovery='All3240pair Laplace-rate degree20 LP and same-arm top810. Reverify32archived H1be frozen proofs without refitting. Actual recovery counts include all900 recipientrows.',
           supplied_prior=old['supplied_prior'],cost=old['cost'],certificate=old['certificate'],solver_options=old['solver_options'],
           query_accounting=dict(recipient_training=720000,donor_warmup=240000,alternative_before_overlap=960000,calibration=0,prior_h1bd_stream_union=1443941,prior_h1az_through_h1be_sequence_union=3465890),
           physical_query_rule='Actualnew=sum(transfer AND NOT union(H1bdaligned,H1bdpermuted,H1befrozen)) perbank,row,cell; reused+new=720000. All donorwarmups already in priorunion. Alternative deployment union=sum(transfer OR warmup),warmup=alignedmask first300 rows in each sourcebank; totaldonor240k,recipient720k,subtract exactoverlap. Each donor used once. Alternative deployment budgets not additive physicalspending.',
           budget=dict(aggregate_cpu_seconds=240,wall_seconds=120,max_cpu_workers=8,max_solver_calls=32,gpu_seconds=0,
                       scope='Preparation,32maskstreams/acquisitions/LPs,independent donor/mask/count replays,32complete-label-substitution gates,32new+32archivedproofs,accounting andsummary. Syntheticgate/registration/packaging separate.'),
           code_sha256={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},
           bank_sha256={str(e['seed']):{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/e['completion_path']]+[PARENT/f"s{e['seed']}"/n for n in ['potential_noise.npz','aligned-observations.npz','permuted-observations.npz','model.npz']]+[SNAPSHOT/f"s{e['seed']}"/n for n in ['observations.npz','model.npz','fit.json','audit.json']]} for e in manifest['inputs']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),donor_map=mapping.tolist())))


def prepare():
    p=verify(full=True);assert not OUT.exists();OUT.mkdir();ids=load_npz(PARENT/'selected_sources.npz')['source_rows']
    assert np.array_equal(donor_map(ids,300),p['donor_map'])
    for recipient,donor in enumerate(p['donor_map']):assert recipient!=donor and np.intersect1d(ids[recipient],ids[donor,:300]).size==0
    prior=json.loads((SNAPSHOT/'audit.json').read_text());assert prior['status']=='validated' and prior['protocol_sha256']==sha(HERE/'protocol-frozen-feedback.json')
    assert prior['physical_query_accounting']['combined_h1bd_union']==1443941
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),donor_map=p['donor_map'],source_disjoint_donor_recipient_pairs=32,
         parent_audit_sha256=sha(PARENT/'audit.json'),snapshot_audit_sha256=sha(SNAPSHOT/'audit.json'),selected_sources_sha256=sha(PARENT/'selected_sources.npz'),scope=p['scope'],new_source_ids=0,new_latent_draws=0,new_solver_inputs=0))


def bank(seed):
    p=verify(seed);d=OUT/f's{seed}';d.mkdir(exist_ok=False);donor=p['donor_map'][seed];ii,jj=np.triu_indices(81,1)
    donor_counts=load_npz(SNAPSHOT/f's{donor}'/'model.npz')['feedback_equal_count'];feedback=np.zeros((81,81),np.int32);feedback[ii,jj]=donor_counts;feedback[jj,ii]=donor_counts
    given,events=masks(feedback,p['tie_seed_base']+seed,900);np.savez_compressed(d/'masks.npz',given=given,donor_feedback=feedback);save(d/'events.json',events);mask_digest=sha(d/'masks.npz')
    # Persisted all900 masks precede recipient source/latent loading.
    entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed
    clean=load_npz(ROOT/entry['completion_path'])['solutions'];noise=load_npz(PARENT/f's{seed}'/'potential_noise.npz');observed=observe(given,clean,noise)
    np.savez_compressed(d/'observations.npz',**observed);m,c=observed['joint_count'],observed['equal_count'];cost=rate_costs(m,c);top=top_count_support(np.ones(3240,bool),cost)
    x,y,solver=solve(81,20,ii,jj,cost,seed,d/'highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
    else:cert=dict(certified=False,reason='solver_status_or_solution_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    if cert['certified'] and cert['integral']:
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
    elif cert['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    arrays={'transfer_'+k:v for k,v in dict(joint_count=m,equal_count=c,cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness).items()};arrays['transfer_mask']=given;arrays['donor_feedback']=feedback
    baseline=load_npz(SNAPSHOT/f's{seed}'/'model.npz');arrays.update({k:v for k,v in baseline.items() if k.startswith('frozen_') or k in ['aligned_mask','permuted_mask']})
    previous=baseline['aligned_mask']|baseline['permuted_mask']|baseline['frozen_mask'];warmup=np.zeros_like(given);warmup[:300]=baseline['aligned_mask'][:300];arrays['warmup_mask']=warmup
    for arm,path,file in [('aligned',PARENT,'aligned-observations.npz'),('permuted',PARENT,'permuted-observations.npz'),('frozen',SNAPSHOT,'observations.npz')]:
        old=load_npz(path/f's{seed}'/file);overlap=old['given']&given;assert np.array_equal(old['values'][overlap],observed['values'][overlap])
    accounting=dict(recipient_queries=int(given.sum()),reused_queries=int((given&previous).sum()),new_queries=int((given&~previous).sum()),prior_h1bd_stream_union=int(previous.sum()),combined_h1bd_stream_union=int((given|previous).sum()),
                    donor_warmup_queries=int(warmup.sum()),deployment_overlap=int((given&warmup).sum()),deployment_union=int((given|warmup).sum()),new_calibration_queries=0)
    np.savez_compressed(d/'model.npz',**arrays);save(d/'fit.json',dict(solver=solver,certificate=cert,classification=category))
    save(d/'receipt.json',dict(status='complete',seed=seed,donor=donor,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),source_sha256=entry['completion_sha256'],noise_sha256=sha(PARENT/f's{seed}'/'potential_noise.npz'),
         masks_persisted_before_recipient_sources_and_latents=True,mask_archive_sha256_before_recipient_labels=mask_digest,query_accounting=accounting,artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()}))
    from audit_transfer_feedback import audit_bank
    audit_bank(seed,(clean,noise))


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['seed']==seed and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_transfer_feedback.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows'];arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'model.npz').items()})
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories={arm:[a['arm_audits'][arm]['classification'] for a in audits] for arm in ARMS};success=categories['transfer'].count('success');unresolved=categories['transfer'].count('unresolved')
    primary=dict(decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive'),success_lower=success,success_upper=success+unresolved,required=29,unresolved=unresolved)
    grouped=[]
    for arm in ARMS:
        top=[r for r in rows if r['arm']==arm+'_top810'];grouped.append(dict(arm=arm+'_top810',exact_graphs=sum(r['exact_graph'] for r in top),missing_true_total=sum(r['missing_true_edges'] for r in top),extra_total=sum(r['extra_edges'] for r in top),banks=32))
        cs=categories[arm];grouped.append(dict(arm=arm,certified_integral_exact=cs.count('success'),certified_integral_nonexact=cs.count('resolved_failure_nonexact'),certified_fractional=cs.count('resolved_failure_fractional'),unresolved=cs.count('unresolved'),banks=32))
    accounting={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert accounting['recipient_queries']==accounting['reused_queries']+accounting['new_queries']==720000 and accounting['donor_warmup_queries']==240000
    assert accounting['deployment_union']==960000-accounting['deployment_overlap'] and accounting['prior_h1bd_stream_union']==1443941
    assert accounting['combined_h1bd_stream_union']==1443941+accounting['new_queries'];accounting['h1az_through_present_sequence_union']=3465890+accounting['new_queries']
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);arrays['donor_map']=np.array(p['donor_map'],np.int64);np.savez_compressed(PUBLIC/'transfer_feedback_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),physical_query_accounting=accounting,alternative_query_accounting=p['query_accounting'],physical_query_rule=p['physical_query_rule'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'transfer_feedback_certificates.npz'),all_banks_retained=True,complete_label_substitution_gates=32,new_source_ids=0,new_latent_draws=0,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'transfer_feedback.json',result);print(json.dumps(dict(primary=primary,grouped=grouped,physical_query_accounting=accounting)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    else:globals()[a.action]()
