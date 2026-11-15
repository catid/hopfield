"""H1bd prospective ledger-disjoint confirmation of both fixed soft-feedback arms."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json
import numpy as np
from run import ROOT,HERE,sha,save
from soft_feedback_policy import collect
from laplace_rate_math import rate_costs
from degree_prior_math import top_count_support,attempt_certificate
from degree_prior_solver import solve
OUT=ROOT/'artifacts/research/20261114/ripper/fresh_soft_feedback';PUBLIC=ROOT/'results/research/ripper'
REGISTRY=OUT.parent/'fresh_soft_feedback_registration';PROTOCOL=HERE/'protocol-fresh-soft-feedback.json';ARMS=['aligned','permuted']


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/p['ledger_path'])==p['ledger_sha256'] and sha(ROOT/p['excluded_ids_path'])==p['excluded_ids_sha256']
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists()
    gate=json.loads((HERE/'fresh-soft-feedback-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    from fresh_rate_study import registry
    prior,manifests,files,old=registry();extra=OUT.parent/'fresh_rate/all_source_manifest.json';spec=dict(path=str(extra.relative_to(ROOT)),sha256=sha(extra));manifests=manifests+[spec]
    more=[]
    for entry in json.loads(extra.read_text())['inputs']:
        path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256'];files[entry['completion_path']]=entry['completion_sha256']
        with np.load(path,allow_pickle=False) as z:more.append(z['source_row'].copy())
    excluded=np.unique(np.concatenate([prior]+more));assert len(excluded)==1988296 and old['pool_size']-len(excluded)==1843698
    assert sha(ROOT/old['source'])==old['source_sha256']
    raw=json.loads((ROOT/old['prior_raw_manifest']).read_text());assert sha(ROOT/old['prior_raw_manifest'])==old['prior_raw_manifest_sha256'] and raw['source_rows']==raw['unique_complete_boards']==old['pool_size']
    REGISTRY.mkdir();np.savez_compressed(REGISTRY/'excluded_source_ids.npz',excluded=excluded)
    ledger=dict(status='prior_registry_sealed_before_new_draws',scope='Archived Ripper completion-source ledger only; not global repository exposure.',excluded_unique=len(excluded),eligible_rows=old['pool_size']-len(excluded),
                manifests=manifests,completion_sha256=files,excluded_array_sha256=sha(REGISTRY/'excluded_source_ids.npz'),excluded_id_bytes_sha256=hashlib.sha256(excluded.tobytes()).hexdigest())
    save(REGISTRY/'ledger.json',ledger)
    previous=json.loads((HERE/'protocol-soft-feedback.json').read_text());sources=['fresh_soft_feedback_study.py','audit_fresh_soft_feedback.py','preflight_fresh_soft_feedback.py','run_fresh_soft_feedback.py',
        'soft_feedback_policy.py','audit_soft_feedback.py','soft_feedback_study.py','fresh_rate_study.py','fresh_rate_common.py','run.py','laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']
    p=dict(id='H1bd-fresh-soft-feedback-confirmation',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,arms=ARMS,fit_order=ARMS,
           source=old['source'],source_sha256=old['source_sha256'],pool_size=old['pool_size'],prior_raw_manifest=old['prior_raw_manifest'],prior_raw_manifest_sha256=old['prior_raw_manifest_sha256'],
           ledger_path=str((REGISTRY/'ledger.json').relative_to(ROOT)),ledger_sha256=sha(REGISTRY/'ledger.json'),excluded_ids_path=str((REGISTRY/'excluded_source_ids.npz').relative_to(ROOT)),excluded_ids_sha256=sha(REGISTRY/'excluded_source_ids.npz'),excluded_unique=1988296,eligible_rows=1843698,
           source_seed_base=202611810000,noise_seed_base=202611820000,tie_seed_base=202611830000,feedback_permutation_seed_base=202611840000,
           feedback_permutations=[np.random.default_rng(202611840000+s).permutation(81).tolist() for s in range(32)],suffix_checkpoint=450,
           scope='Prospective new900row banks disjoint from all1,988,296 IDs in the declared Ripper completion-source ledger. New source, noise, tie and feedbackpermutation seeds. Not globally untouched or externaldistribution evidence; no puzzle-solver inputs.',
           sampling='Perbank independently seeded choice900without replacement from all sorted1,843,698eligibleIDs; no calibration split. Crossbank source-ID overlap allowed and reported. Seal all source identities before selected labels. Feedbackpermutations fixed in this protocol before source IDs.',
           policy=previous['policy'],predictability=previous['predictability'],primary=previous['primary'],alignment_endpoint=previous['alignment_endpoint'],mechanism_limit=previous['mechanism_limit'],
           noise='Each bank,row has new independent U and uniform81positionpermutation; corruptwholeboard iffU<9/160. Identical underlying noisyboard shared across aligned/permuted arms; eachhas own pastobservedhistory. Noise RNG neverdepends on masks. No calibration collected.',
           recovery='Unchanged all3240-pair Laplace-rate degree20 LP and same-arm top810controls. No archivedsame-source comparator or extraLPs. Fractional returnedoptimum resolvedfailure, not proof no integralcooptimum; no snapping/uniqueness claim.',
           supplied_prior=previous['supplied_prior'],cost=previous['cost'],certificate=previous['certificate'],solver_options=previous['solver_options'],
           query_accounting=dict(alternative_each_arm_training=720000,alternative_each_arm_calibration=0),
           physical_query_rule='All selected identities outside priorRipperledger, so newphysicalqueries=sum(alignedmask OR permutedmask) over bank,row,cell. Shared=sum(aligned AND permuted),newunion=1440000-shared. Crossbank repeatedsourceIDs have distinct latentinstances and remain separatelycharged. Alternative720kdeploymentbudgets not additive.',
           budget=dict(aggregate_cpu_seconds=900,wall_seconds=180,max_cpu_workers=8,max_solver_calls=64,gpu_seconds=0,
                       scope='Identityseal,sourceCSVmaterialization+independentaudit,64acquisitions/LPs,sequentialreplay,64suffix450gates,proofs,queryaccounting,summary. Priorledgerregistration/syntheticgate/packaging separate.'),
           code_sha256={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in sources},preflight_sha256=sha(HERE/'fresh-soft-feedback-preflight.json'),parent_protocol_sha256=sha(HERE/'protocol-soft-feedback.json'))
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),excluded_unique=len(excluded),eligible_rows=p['eligible_rows'])))


def select():
    p=verify();assert not OUT.exists();excluded=load_npz(ROOT/p['excluded_ids_path'])['excluded'];eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==p['eligible_rows'];draws=np.stack([np.random.default_rng(p['source_seed_base']+seed).choice(eligible,900,replace=False) for seed in p['seeds']])
    OUT.mkdir();np.savez_compressed(OUT/'selected_sources.npz',source_rows=draws,all_source_row=np.unique(draws))
    seal=dict(status='identities_sealed_before_selected_labels',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PROTOCOL),selector_sha256=sha(__file__),selected_sources_sha256=sha(OUT/'selected_sources.npz'),
              ledger_sha256=p['ledger_sha256'],source_instances=28800,new_unique_source_ids=int(len(np.unique(draws))),crossbank_repeated_instances=int(draws.size-len(np.unique(draws))),within_bank_repeats=0,declared_ledger_overlap=0,scope=p['scope'])
    save(OUT/'identity-seal.json',seal);print(json.dumps(seal),flush=True)


def materialize():
    p=verify();seal=json.loads((OUT/'identity-seal.json').read_text());assert seal['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==sha(OUT/'selected_sources.npz')
    source=ROOT/p['source'];assert sha(source)==p['source_sha256'];draws=load_npz(OUT/'selected_sources.npz')['source_rows'];wanted={};solutions=np.zeros((32,900,81),np.int8)
    for seed in range(32):
        for offset,row in enumerate(draws[seed]):wanted.setdefault(int(row),[]).append((seed,offset))
    remaining=set(wanted)
    with source.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if index not in remaining:continue
            board=np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1');assert board.shape==(81,)
            for seed,offset in wanted[index]:solutions[seed,offset]=board
            remaining.remove(index)
            if not remaining:break
    assert not remaining;boards=solutions.reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    for digit in range(9):assert ((boards==digit).sum(1)==1).all() and ((boards==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
    entries=[]
    for seed in range(32):
        path=OUT/f'training-s{seed}.npz';np.savez_compressed(path,solutions=solutions[seed],source_row=draws[seed]);entries.append(dict(seed=seed,role='training',completion_path=str(path.relative_to(ROOT)),completion_sha256=sha(path)))
    manifest=dict(status='prepared_without_acquisition_outcomes',protocol_sha256=sha(PROTOCOL),identity_seal_sha256=sha(OUT/'identity-seal.json'),selected_sources_sha256=sha(OUT/'selected_sources.npz'),source_sha256=p['source_sha256'],inputs=entries)
    save(OUT/'input_manifest.json',manifest);save(OUT/'all_source_manifest.json',dict(**manifest,purpose='Future Ripper source-ID exclusions must include these32training banks; no calibration sources.'))


def bank(seed):
    p=verify();source_audit=json.loads((OUT/'source-audit.json').read_text());assert source_audit['status']=='validated' and source_audit['protocol_sha256']==sha(PROTOCOL) and source_audit['input_manifest_sha256']==sha(OUT/'input_manifest.json')
    d=OUT/f's{seed}';d.mkdir(exist_ok=False);entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed and sha(ROOT/entry['completion_path'])==entry['completion_sha256']
    clean=load_npz(ROOT/entry['completion_path'])['solutions'];rng=np.random.default_rng(p['noise_seed_base']+seed);uniforms=np.zeros(900);permutations=np.zeros((900,81),np.uint8)
    for row in range(900):uniforms[row]=rng.random();permutations[row]=rng.permutation(81)
    noise=dict(uniforms=uniforms,permutations=permutations);np.savez_compressed(d/'potential_noise.npz',**noise)
    feedback_perm=np.array(p['feedback_permutations'][seed],np.int64);arrays={};observations={};ii,jj=np.triu_indices(81,1)
    for arm in p['fit_order']:
        observed,events=collect(clean,uniforms,permutations,p['tie_seed_base']+seed,feedback_perm if arm=='permuted' else None,p['suffix_checkpoint']);observations[arm]=observed
        np.savez_compressed(d/f'{arm}-observations.npz',**observed);save(d/f'{arm}-events.json',events)
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
        arrays[arm+'_mask']=observed['given'];save(d/f'{arm}-fit.json',dict(solver=solver,certificate=cert,classification=category))
    aligned=observations['aligned']['given'];permuted=observations['permuted']['given'];shared=aligned&permuted
    assert np.array_equal(observations['aligned']['values'][shared],observations['permuted']['values'][shared])
    accounting=dict(aligned_queries=int(aligned.sum()),permuted_queries=int(permuted.sum()),shared_queries=int(shared.sum()),new_union_queries=int((aligned|permuted).sum()),new_calibration_queries=0)
    np.savez_compressed(d/'model.npz',**arrays)
    save(d/'receipt.json',dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),source_sha256=entry['completion_sha256'],
         query_accounting=accounting,artifact_sha256={f.name:sha(f) for f in d.iterdir() if f.is_file()}))
    from audit_fresh_soft_feedback import audit_bank
    audit_bank(seed,(clean,noise))


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and a['seed']==seed and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_fresh_soft_feedback.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and a['model_sha256']==sha(d/'model.npz')
        audits.append(a);rows+=a['rows'];arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'model.npz').items()})
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories={arm:[a['arm_audits'][arm]['classification'] for a in audits] for arm in ARMS};sa,sp=[categories[arm].count('success') for arm in ARMS];ua,up=[categories[arm].count('unresolved') for arm in ARMS]
    procedure=dict(decision='supported' if sa>=29 else ('falsified' if sa+ua<29 else 'inconclusive'),success_lower=sa,success_upper=sa+ua,required=29,unresolved=ua)
    low=sa-sp-up;high=sa+ua-sp;alignment=dict(decision='supported' if low>=4 else ('falsified' if high<4 else 'inconclusive'),difference_lower=low,difference_upper=high,required=4,aligned_success=sa,permuted_success=sp,aligned_unresolved=ua,permuted_unresolved=up)
    grouped=[]
    for arm in ARMS:
        top=[r for r in rows if r['arm']==arm+'_top810'];grouped.append(dict(arm=arm+'_top810',exact_graphs=sum(r['exact_graph'] for r in top),missing_true_total=sum(r['missing_true_edges'] for r in top),extra_total=sum(r['extra_edges'] for r in top),banks=32))
        cs=categories[arm];grouped.append(dict(arm=arm,certified_integral_exact=cs.count('success'),certified_integral_nonexact=cs.count('resolved_failure_nonexact'),certified_fractional=cs.count('resolved_failure_fractional'),unresolved=cs.count('unresolved'),banks=32))
    accounting={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert accounting['aligned_queries']==accounting['permuted_queries']==720000 and accounting['new_union_queries']==1440000-accounting['shared_queries']
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);np.savez_compressed(PUBLIC/'fresh_soft_feedback_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=procedure,alignment_endpoint=alignment,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),identity_seal=json.loads((OUT/'identity-seal.json').read_text()),physical_query_accounting=accounting,alternative_query_accounting=p['query_accounting'],physical_query_rule=p['physical_query_rule'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'fresh_soft_feedback_certificates.npz'),all_banks_retained=True,suffix_substitution_gates=64,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'fresh_soft_feedback.json',result);print(json.dumps(dict(primary=procedure,alignment_endpoint=alignment,grouped=grouped,physical_query_accounting=accounting)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','select','materialize','source-audit','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    elif a.action=='source-audit':
        from audit_fresh_soft_feedback import sources
        sources()
    else:globals()[a.action]()
