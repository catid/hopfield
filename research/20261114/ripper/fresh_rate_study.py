"""H1az: prospective N900 recovery on sources disjoint from the Ripper ledger."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,sys,time
import numpy as np
from fresh_rate_common import paths,load_protocol
from run import ROOT,HERE,sha,save
OUT,PROTOCOL=paths();PUBLIC=ROOT/'results/research/ripper'
LEDGER=OUT.parent/'fresh_rate_registration'
ARMS=['original','top810','degree20'];SOURCE=ROOT/'upstream/data/sudoku-extreme/train.csv'


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify():return load_protocol(False)


def registry():
    old=json.loads((HERE/'protocol-balanced-filter-inputs.json').read_text())
    manifests=old['exclusion_manifests']+[dict(path='artifacts/research/20261114/ripper/balanced_filter/all_source_manifest.json',sha256=sha(OUT.parent/'balanced_filter/all_source_manifest.json'))]
    mask=np.zeros(old['pool_size'],bool);files={}
    for spec in manifests:
        assert sha(ROOT/spec['path'])==spec['sha256'];manifest=json.loads((ROOT/spec['path']).read_text())
        for entry in manifest['inputs']:
            path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256']
            with np.load(path,allow_pickle=False) as z:ids=z['source_row'].copy()
            assert ids.dtype==np.int64 and ((0<=ids)&(ids<old['pool_size'])).all()
            mask[ids]=True;files[entry['completion_path']]=entry['completion_sha256']
    assert int(mask.sum())==1928477
    return np.flatnonzero(mask),manifests,files,old


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not LEDGER.exists()
    gate=json.loads((HERE/'fresh-rate-preflight.json').read_text());assert gate['status']=='validated_synthetic_only'
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,name
    excluded,manifests,files,old=registry();assert old['pool_size']-len(excluded)>=2*32000
    assert sha(SOURCE)==old['source_sha256'];raw=json.loads((ROOT/old['prior_raw_manifest']).read_text())
    assert sha(ROOT/old['prior_raw_manifest'])==old['prior_raw_manifest_sha256'] and raw['source_rows']==raw['unique_complete_boards']==old['pool_size']
    LEDGER.mkdir();np.savez_compressed(LEDGER/'excluded_source_ids.npz',excluded=excluded)
    save(LEDGER/'ledger.json',dict(status='prior_id_registry_sealed_without_new_draws',scope='Archived Ripper completion-source ledger only; not a global repository exposure ledger.',
         excluded_unique=1928477,eligible_rows=old['pool_size']-1928477,manifests=manifests,completion_sha256=files,
         excluded_array_sha256=sha(LEDGER/'excluded_source_ids.npz'),excluded_id_bytes_sha256=hashlib.sha256(excluded.tobytes()).hexdigest()))
    sources=['fresh_rate_study.py','fresh_rate_common.py','fresh_rate_acquire.py','audit_fresh_rate_acquisition.py','audit_fresh_rate.py','preflight_fresh_rate.py','run_fresh_rate.py',
             'permuted_noise_policy.py','calibration_estimator.py','calibrated_policy.py','noisy_policy.py','balanced_filter_policy.py','online_policy.py','fast_reference_mask.py',
             'audit_correlated_noise_acquisition.py','correlated_noise_common.py','run.py','laplace_rate_math.py','audit_laplace_rate.py','degree_prior_math.py','degree_prior_solver.py','audit_degree_prior.py']
    prior=json.loads((HERE/'protocol-laplace-rate.json').read_text())
    p=dict(id='H1az-fresh-ledger-disjoint-N900-confirmation',frozen_utc=datetime.now(timezone.utc).isoformat(),seeds=list(range(32)),rows=900,calibration_pairs=1000,
           arms=ARMS,inputs=str((OUT/'input_manifest.json').relative_to(ROOT)),source=str(SOURCE.relative_to(ROOT)),source_sha256=old['source_sha256'],pool_size=old['pool_size'],
           prior_raw_manifest=old['prior_raw_manifest'],prior_raw_manifest_sha256=old['prior_raw_manifest_sha256'],
           ledger_path=str((LEDGER/'ledger.json').relative_to(ROOT)),ledger_sha256=sha(LEDGER/'ledger.json'),excluded_ids_path=str((LEDGER/'excluded_source_ids.npz').relative_to(ROOT)),
           excluded_ids_sha256=sha(LEDGER/'excluded_source_ids.npz'),excluded_unique=1928477,eligible_rows=old['pool_size']-1928477,
           scope='Fresh source identities disjoint from the archived Ripper completion-source ledger; this is not a globally untouched identity claim. New calibration, predictable acquisition, corruption and tie streams. No held-out puzzle or solver inputs.',
           draw_rule='RNG202611710000 permutes sorted eligible source IDs. First floor half calibration, rest training. Perbank sample1000 calibration and900 training IDs without replacement using separate seeded streams; crossbank repeats allowed and reported. Globally disjoint calibration/training pools. Seal exact identity manifest before reading any selected labels.',
           streams=dict(pool_split=202611710000,calibration_source=202611720000,training_source=202611730000,ties=202611740000,calibration_cell=202611750000,calibration_noise=202611760000,training_noise=202611770000),
           acquisition='Unchanged H1ap exponential_e predictable25cell mask, misspecified repeated-label calibration, first crossing exclusion, histogram balance acceptance. Board position permutation noise probability9/160. Masks fixed before current corruption; repeated labels share a board. Run900 rows only; no prefix tuning or extra observations.',
           recovery='All3240 pairs, unchanged Laplace rate integer costs and degree20 LP. Top810 same cost control; original acquisition support descriptive. No source geometry enters masks or recovery.',
           supplied_prior=prior['supplied_prior'],cost=prior['cost'],certificate=prior['certificate'],solver_options=prior['solver_options'],
           primary='At least29/32 independently certified integral exact graphs at N900. Supported if successes>=29; falsified if successes+unresolved<29; otherwise inconclusive. All32 retained; no replacement or extra-prefix rescue.',
           fractional='Certified fractional returned optima are resolved procedure failures, not proof that no integral co-optimum exists. No primal snapping; exact proof certifies optimality, not uniqueness.',
           query_accounting=dict(training_per_bank=22500,calibration_per_bank=2000,total_per_bank=24500,new_training_queries=720000,new_calibration_queries=64000,new_queries=784000),
           budget=dict(aggregate_cpu_seconds=900,wall_seconds=180,max_cpu_workers=8,max_solver_calls=32,gpu_seconds=0,
                       scope='Identity selection/seal, source materialization, independent CSV/source audit, all32 acquisitions and independent mask/noise/calibration replays, LPs/proofs and summary. Prior-ledger registration, synthetic preflight and packaging separate.'),
           code_sha256={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in sources},preflight_sha256=sha(HERE/'fresh-rate-preflight.json'))
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),excluded_unique=len(excluded),eligible_rows=p['eligible_rows'])))


def select():
    p=verify();assert not OUT.exists();assert sha(ROOT/p['ledger_path'])==p['ledger_sha256'] and sha(ROOT/p['excluded_ids_path'])==p['excluded_ids_sha256']
    excluded=load_npz(ROOT/p['excluded_ids_path'])['excluded'];mask=np.ones(p['pool_size'],bool);mask[excluded]=False
    eligible=np.flatnonzero(mask);assert len(eligible)==p['eligible_rows']
    pool=np.random.default_rng(p['streams']['pool_split']).permutation(eligible);half=len(pool)//2
    cal=np.array([np.random.default_rng(p['streams']['calibration_source']+s).choice(pool[:half],1000,replace=False) for s in p['seeds']])
    train=np.array([np.random.default_rng(p['streams']['training_source']+s).choice(pool[half:],900,replace=False) for s in p['seeds']])
    assert not np.isin(cal,train).any() and mask[cal].all() and mask[train].all()
    OUT.mkdir();np.savez_compressed(OUT/'selected_sources.npz',calibration_source_row=cal,training_source_row=train,all_source_row=np.unique(np.r_[cal.ravel(),train.ravel()]))
    sealed=dict(status='identities_sealed_before_selected_labels',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PROTOCOL),selector_sha256=sha(__file__),
                selected_sources_sha256=sha(OUT/'selected_sources.npz'),ledger_sha256=p['ledger_sha256'],prior_overlap=0,calibration_training_overlap=0,
                calibration_unique=int(len(np.unique(cal))),training_unique=int(len(np.unique(train))),calibration_instances=32000,training_instances=28800,
                calibration_pool_rows=half,training_pool_rows=len(pool)-half,scope=p['scope'])
    save(OUT/'identity-seal.json',sealed);print(json.dumps(sealed),flush=True)


def materialize():
    p=verify();seal=json.loads((OUT/'identity-seal.json').read_text());assert seal['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==sha(OUT/'selected_sources.npz')
    assert sha(SOURCE)==p['source_sha256'];selected=load_npz(OUT/'selected_sources.npz');wanted={};arrays={}
    for role,n in [('calibration',1000),('training',900)]:
        arrays[role]=np.zeros((32,n,81),np.int8)
        for seed in range(32):
            for offset,row in enumerate(selected[role+'_source_row'][seed]):wanted.setdefault(int(row),[]).append((role,seed,offset))
    remaining=set(wanted)
    with SOURCE.open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if index not in remaining:continue
            board=np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1');assert board.shape==(81,)
            for role,seed,offset in wanted[index]:arrays[role][seed,offset]=board
            remaining.remove(index)
            if not remaining:break
    assert not remaining;entries=[];complete=[]
    for role in arrays:
        boards=arrays[role].reshape(-1,9,9);boxes=boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert ((boards==digit).sum(1)==1).all() and ((boards==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
        for seed in range(32):
            path=OUT/f'{role}-s{seed}.npz';np.savez_compressed(path,solutions=arrays[role][seed],source_row=selected[role+'_source_row'][seed])
            complete.append(dict(seed=seed,role=role,completion_path=str(path.relative_to(ROOT)),completion_sha256=sha(path)))
    for seed in range(32):
        entry=dict(seed=seed)
        for role in arrays:
            record=next(e for e in complete if e['seed']==seed and e['role']==role)
            entry[role+'_path']=record['completion_path'];entry[role+'_sha256']=record['completion_sha256']
        entries.append(entry)
    save(OUT/'all_source_manifest.json',dict(status='prepared_without_acquisition_outcomes',protocol_sha256=sha(PROTOCOL),source_sha256=p['source_sha256'],inputs=complete,purpose='Future ledger exclusions must include both calibration and training IDs.'))
    save(OUT/'input_manifest.json',dict(status='prepared_without_acquisition_outcomes',protocol_sha256=sha(PROTOCOL),identity_seal_sha256=sha(OUT/'identity-seal.json'),
         selected_sources_sha256=sha(OUT/'selected_sources.npz'),source_sha256=p['source_sha256'],all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),inputs=entries))


def bank(seed):
    from fresh_rate_acquire import main as acquire
    from audit_fresh_rate_acquisition import main as audit_acquisition
    p=load_protocol();sys.argv=['fresh-rate-bank','--seed',str(seed)];acquire();audit_acquisition()
    from laplace_rate_math import rate_costs
    from degree_prior_math import top_count_support,attempt_certificate
    from degree_prior_solver import solve
    d=OUT/f's{seed}';counts=load_npz(d/'fit.npz');ii,jj=np.triu_indices(81,1);cost=rate_costs(counts['joint_count'],counts['equal_count'])
    top=top_count_support(np.ones(3240,bool),cost);x,y,solver=solve(81,20,ii,jj,cost,seed,d/'recovery-highs.log')
    if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,witness=attempt_certificate(81,20,ii,jj,cost,x,y)
    else:
        cert=dict(certified=False,reason='solver_status_or_solution_validity');witness={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    if cert['certified'] and cert['integral']:
        truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));category='success' if np.array_equal(x==1,truth) else 'resolved_failure_nonexact'
    elif cert['certified']:category='resolved_failure_fractional'
    else:category='unresolved'
    np.savez_compressed(d/'recovery.npz',joint_count=counts['joint_count'],equal_count=counts['equal_count'],original_support=counts['support'],cost=cost,top810_support=top,primal_values=x,row_dual_values=y,**witness)
    save(d/'recovery.json',dict(status='complete',seed=seed,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),solver=solver,certificate=cert,classification=category,
         acquisition_audit_sha256=sha(d/'audit.json'),acquisition_receipt_sha256=sha(d/'receipt.json'),recovery_sha256=sha(d/'recovery.npz')))
    from audit_fresh_rate import audit_bank
    audit_bank(seed)


def summary():
    p=verify();audits=[];rows=[];arrays={};hashes={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';a=json.loads((d/'recovery-audit.json').read_text())
        assert a['status']=='validated' and a['seed']==seed and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_fresh_rate.py')
        assert a['recovery_sha256']==sha(d/'recovery.npz') and a['recovery_receipt_sha256']==sha(d/'recovery.json')
        audits.append(a);rows+=a['rows']
        arrays.update({f's{seed}_'+k:v for k,v in load_npz(d/'recovery.npz').items()})
        for f in d.iterdir():
            if f.is_file():hashes[str(f.relative_to(ROOT))]=sha(f)
    categories=[a['classification'] for a in audits];success=categories.count('success');unresolved=categories.count('unresolved')
    primary=dict(decision='supported' if success>=29 else ('falsified' if success+unresolved<29 else 'inconclusive'),certified_integral_exact_graphs=success,
                 resolved_failures=32-success-unresolved,unresolved=unresolved,success_lower_bound=success,success_upper_bound=success+unresolved,required=29)
    grouped=[]
    for arm in ARMS[:2]:
        part=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,exact_graphs=sum(r['exact_graph'] for r in part),missing_true_total=sum(r['missing_true_edges'] for r in part),extra_total=sum(r['extra_edges'] for r in part),banks=32))
    grouped.append(dict(arm='degree20',certified_integral_exact=success,certified_integral_nonexact=categories.count('resolved_failure_nonexact'),certified_fractional=categories.count('resolved_failure_fractional'),unresolved=unresolved,banks=32))
    arrays['banks']=np.arange(32);arrays['arms']=np.array(ARMS);np.savez_compressed(PUBLIC/'fresh_rate_certificates.npz',**arrays)
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),summarizer_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=grouped,rows=rows,bank_audits=audits,
                source_audit=json.loads((OUT/'source-audit.json').read_text()),identity_seal=json.loads((OUT/'identity-seal.json').read_text()),query_accounting=p['query_accounting'],
                artifact_sha256=hashes,certificate_export_sha256=sha(PUBLIC/'fresh_rate_certificates.npz'),all_banks_retained=True,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'fresh_rate.json',result);print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','select','materialize','source-audit','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action=='bank':bank(a.seed)
    elif a.action=='source-audit':
        from audit_fresh_rate import sources
        sources()
    else:globals()[a.action]()
