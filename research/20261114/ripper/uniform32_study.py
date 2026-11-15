"""H1bk: fresh paired donor-free uniform32 versus donor25+7 acquisition."""
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,shutil
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz,zero_witness
from transfer_feedback_policy import masks as donor_masks
from extra_coverage_study import extend,observe
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate,top_count_support
from laplace_rate_math import rate_costs
OUT=ROOT/'artifacts/research/20261114/ripper/uniform32'
REGISTRY=OUT.parent/'uniform32_registration'
PUBLIC=ROOT/'results/research/ripper/uniform32'
PROTOCOL=HERE/'protocol-uniform32.json'
H4=ROOT/'results/research/foureyes/fresh_curvature/controls'
DONORS=ROOT/'results/research/ripper/transfer_feedback_certificates.npz'
DONOR_RAW=ROOT/'artifacts/research/20261114/ripper/fresh_soft_feedback'
ARMS=['uniform','donor']


def make_masks(feedback,tie_seed,rank_seed,rows=900):
    old,_=donor_masks(feedback,tie_seed,rows);donor=extend(old,rank_seed);rng=np.random.default_rng(rank_seed);uniform=np.zeros_like(old)
    for row in range(rows):uniform[row,rng.permutation(81)[:32]]=True
    return dict(donor_old25=old,donor=donor,uniform=uniform)


def verify(raw=False):
    p=json.loads(PROTOCOL.read_text())
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    if raw:
        for rel,digest in p['raw_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists() and not PUBLIC.exists()
    gate=HERE/'uniform32-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    old=json.loads((HERE/'protocol-fresh-transfer.json').read_text());lp=json.loads((HERE/'protocol-extra-coverage.json').read_text());transfer=json.loads((HERE/'protocol-transfer-feedback.json').read_text());h4=json.loads((H4/'summary.json').read_text())
    ids=npz(H4/'source-identities.npz');rawdraws=ROOT/'artifacts/research/20261114/foureyes/fresh_curvature/source-draws.npz';assert sha(rawdraws)==h4['source_audit']['source_draws_sha256'] and np.array_equal(npz(rawdraws)['source_rows'],ids['source_rows'])
    excluded=np.union1d(ids['excluded_ids'],ids['source_rows']);assert len(ids['excluded_ids'])==2134884 and len(np.unique(ids['source_rows']))==147012 and len(excluded)==2281896
    ripper=np.union1d(npz(ROOT/'results/research/ripper/fresh_transfer_excluded_ids.npz')['excluded'],npz(ROOT/'results/research/ripper/fresh_transfer_selected_sources.npz')['all_source_row']);assert len(ripper)==2045471 and np.isin(ripper,excluded).all()
    REGISTRY.mkdir();np.savez_compressed(REGISTRY/'excluded_ids.npz',excluded=excluded,ripper_prior=ripper)
    ledger=dict(status='sealed_before_new_draws',scope='Declared H4ak prior exclusion union and H4ak draws, including latest declared Ripper ledger; not global repository untouchedness.',excluded_unique=len(excluded),ripper_prior_unique=len(ripper),eligible_rows=old['pool_size']-len(excluded),
                h4_protocol_sha256=sha(ROOT/'research/20261114/foureyes/fresh_curvature.json'),h4_summary_sha256=sha(H4/'summary.json'),h4_public_id_sha256=sha(H4/'source-identities.npz'),h4_raw_draw_sha256=sha(rawdraws),excluded_file_sha256=sha(REGISTRY/'excluded_ids.npz'))
    save(REGISTRY/'ledger.json',ledger)
    inputs=[gate,REGISTRY/'excluded_ids.npz',REGISTRY/'ledger.json',H4/'source-identities.npz',H4/'summary.json',ROOT/'research/20261114/foureyes/fresh_curvature.json',DONORS,HERE/'protocol-transfer-feedback.json',HERE/'protocol-fresh-transfer.json',HERE/'protocol-extra-coverage.json',ROOT/'results/research/ripper/extra_coverage/summary.json',ROOT/'results/research/ripper/fresh_transfer_excluded_ids.npz',ROOT/'results/research/ripper/fresh_transfer_selected_sources.npz']
    raw=[ROOT/old['source'],rawdraws,DONOR_RAW/'selected_sources.npz']+[DONOR_RAW/f's{seed}'/'aligned-observations.npz' for seed in range(32)]
    p=dict(id='H1bk-fresh-uniform32-versus-donor',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,rows=900,arms=ARMS,source=old['source'],source_csv_sha256=old['source_sha256'],pool_size=old['pool_size'],
           source_seed_base=202612010000,noise_seed_base=202612020000,tie_seed_base=202612030000,rank_seed_base=202612040000,donor_map=transfer['donor_map'],
           scope='Prospective32fresh900row paired banks disjoint from declared H4ak/Ripper exclusion union only. Same official corpus, not externally distributed or globally untouched. Uniform32 donor-free versus unchanged fixed donor25+7 recipe.',
           sampling='Independent uniform900withoutreplacement draw perbank from sortedeligibleIDs; retain/report crossbank repeats. Seal allIDs, then all64maskstreams, then materialize selectedCSVlabels.',
           acquisition='Uniform: first32 of fresh random81-cell order. Donor: inherited fixedfeedback25-cell sequential policy, then first7 absent cells of same row order. Shared ranks couple arms without changing either marginal recipe; no recipient-label feedback.',
           noise='Fresh mask-independent U and uniform81position permutation perrow; permute entireboard iffU<9/160. Paired arms share the same corruptedboard. Algorithm receives no noise probability.',
           degrees=lp['degrees'],candidate_degrees=lp['candidate_degrees'],costs=lp['costs'],objective=lp['objective'],selector=lp['selector'],prior=lp['prior'],proof=lp['proof'],
           uniform_endpoint=dict(required=31,success='Uniform selected graph certified integral exact after all grid degrees certify.'),donor_benefit_endpoint=dict(required=2,success='Donor exact count minus uniform exact count is at least2.'),
           decision='Separate finite targets. Uniform lower=sU,upper=sU+uU. Benefit lower=sD-sU-uU,upper=sD+uD-sU. Supported iff lower>=target; falsified iff upper<target; otherwise inconclusive. Any uncertified grid degree makes bank unresolved; selected fractional/nonexact is graph failure. No equivalence, significance or universal zero-donor-value claim.',
           diagnostics='Degree20 selections, all paired wins/losses, fixeddegree20 grid diagnostic and top810. BP first. All64arms retained; no tuning or rescue.',
           solver=dict(lp['solver'],random_seed_base=2026171502),seed_rule='Index=2*bank+arm_index; seed=base+index*41+d; fresh solver every degree including20.',
           budget=dict(max_cpu_workers=8,new_LPs=2560,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0,new_physical_labels_cap=1843200),
           budget_scope='IDseal,64maskstreams/materialization/source-and-donor audit,64acquisitions/grids,independent mask/value/cost/proof/substitution gates,unions and summary. Registration/synthetic gate/packaging separate.',
           query_accounting=dict(uniform_recipient=921600,uniform_donor=0,uniform_complete_deployment=921600,donor_recipient=921600,donor_historical_warmup=240000,donor_complete_deployment=1161600,new_donor_queries=0,prior_h1az_through_h1bj_sequence_union=5302606,prior_ripper_ledger_union=2045471,prior_combined_exclusion_union=2281896),
           physical_query_rule='Freshrecipient IDs outside declared exclusions/donors: actualnew=sum(uniform OR donor) overbank,row,cell; shared chargeonce. Historical donor240k reused for research but charged to donor deployment only. Crossbank source repeats have separate latents and chargedseparately.',
           source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in raw})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),excluded_ids=len(excluded),eligible_rows=ledger['eligible_rows'])))


def select():
    p=verify();assert not OUT.exists();excluded=npz(REGISTRY/'excluded_ids.npz')['excluded'];eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True);assert len(eligible)==1550098
    draws=np.stack([np.random.default_rng(p['source_seed_base']+seed).choice(eligible,900,replace=False) for seed in range(32)]);OUT.mkdir();np.savez_compressed(OUT/'selected_sources.npz',source_rows=draws,all_source_row=np.unique(draws))
    save(OUT/'identity-seal.json',dict(status='sealed_before_selected_labels',protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),selected_sources_sha256=sha(OUT/'selected_sources.npz'),new_unique_source_ids=len(np.unique(draws)),crossbank_repeated_instances=draws.size-len(np.unique(draws))))


def predraw(seed):
    p=verify();assert (OUT/'identity-seal.json').exists() and not (OUT/'input_manifest.json').exists();d=OUT/f's{seed}';d.mkdir(exist_ok=False)
    with np.load(DONORS,allow_pickle=False) as z:feedback=z[f's{seed}_donor_feedback'].copy()
    arrays=make_masks(feedback,p['tie_seed_base']+seed,p['rank_seed_base']+seed);arrays['feedback']=feedback;np.savez_compressed(d/'masks.npz',**arrays)
    save(d/'predraw-receipt.json',dict(status='sealed_before_selected_labels',seed=seed,protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),identity_seal_sha256=sha(OUT/'identity-seal.json'),masks_sha256=sha(d/'masks.npz')))


def materialize():
    p=verify();predraws={}
    for seed in range(32):
        d=OUT/f's{seed}';r=json.loads((d/'predraw-receipt.json').read_text());assert r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['masks_sha256']==sha(d/'masks.npz');predraws[str(seed)]=sha(d/'predraw-receipt.json')
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];draws=npz(OUT/'selected_sources.npz')['source_rows'];wanted={};solutions=np.zeros((32,900,81),np.int8)
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
    assert not remaining;entries=[]
    for seed in range(32):
        path=OUT/f'training-s{seed}.npz';np.savez_compressed(path,solutions=solutions[seed],source_row=draws[seed]);entries.append(dict(seed=seed,role='training',completion_path=str(path.relative_to(ROOT)),completion_sha256=sha(path)))
    manifest=dict(protocol_sha256=sha(PROTOCOL),all64_masks_sealed_before_materialization=True,predraw_receipts_sha256=predraws,identity_seal_sha256=sha(OUT/'identity-seal.json'),selected_sources_sha256=sha(OUT/'selected_sources.npz'),inputs=entries)
    save(OUT/'input_manifest.json',manifest);save(OUT/'all_source_manifest.json',dict(**manifest,purpose='Future exclusions must include these32new recipient banks.'))


def bank(seed):
    p=verify();source=json.loads((OUT/'source-audit.json').read_text());assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['input_manifest_sha256']==sha(OUT/'input_manifest.json')
    d=OUT/f's{seed}';m=npz(d/'masks.npz');entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert sha(ROOT/entry['completion_path'])==entry['completion_sha256'];clean=npz(ROOT/entry['completion_path'])['solutions']
    rng=np.random.default_rng(p['noise_seed_base']+seed);uniforms=np.zeros(900);permutations=np.zeros((900,81),np.uint8)
    for row in range(900):uniforms[row]=rng.random();permutations[row]=rng.permutation(81)
    noise=dict(uniforms=uniforms,permutations=permutations);np.savez_compressed(d/'potential_noise.npz',**noise);ii,jj=np.triu_indices(81,1)
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;observed=observe(m[arm],clean,noise);np.savez_compressed(d/f'{arm}-observations.npz',**observed)
        cost=rate_costs(observed['joint_count'],observed['equal_count']);a=dict(cost=cost,joint_count=observed['joint_count'],equal_count=observed['equal_count'],top810_support=top_count_support(np.ones(3240,bool),cost));records=[];a.update({'d00_'+k:v for k,v in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,60000+index*41+degree,d/f'{arm}-d{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            a.update({f'd{degree:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**w).items()});records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'{arm}-d{degree:02d}.log')))
        np.savez_compressed(d/f'{arm}-models.npz',**a);save(d/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/f'{arm}-models.npz'),observations_sha256=sha(d/f'{arm}-observations.npz'),noise_sha256=sha(d/'potential_noise.npz'),predraw_receipt_sha256=sha(d/'predraw-receipt.json')))
    from audit_uniform32 import audit_bank
    a=audit_bank(seed,False);save(d/'audit.json',a);print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def summary():
    from audit_uniform32 import aggregate
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);audits=[];files={}
    for seed in range(32):
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_uniform32.py')
        for name,digest in a['artifact_sha256'].items():assert sha(d/name)==digest
        audits.append(a);dest=PUBLIC/f's{seed}';dest.mkdir()
        for name in ['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]:shutil.copyfile(d/name,dest/name);files[f's{seed}/{name}']=sha(d/name)
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert q['uniform_queries']==q['donor_queries']==921600 and q['new_union_queries']<=1843200
    q.update(historical_donor_warmup=240000,new_donor_queries=0,h1az_through_present_sequence_union=5302606+q['new_union_queries'])
    for name in ['selected_sources.npz','identity-seal.json','input_manifest.json','all_source_manifest.json','source-audit.json']:shutil.copyfile(OUT/name,PUBLIC/name)
    shutil.copyfile(REGISTRY/'excluded_ids.npz',PUBLIC/'excluded_ids.npz');shutil.copyfile(REGISTRY/'ledger.json',PUBLIC/'ledger.json')
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],grouped=grouped,endpoints=endpoints,paired_changes=paired,rows=rows,raw_audits=audits,query_accounting=q,alternative_query_accounting=p['query_accounting'],source_audit=json.loads((OUT/'source-audit.json').read_text()),
                certified_LPs=sum(r['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),analytic_zero_proofs=64,complete_label_gates=64,artifact_sha256=files,all_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(grouped=grouped,endpoints=endpoints,query_accounting=q)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','select','predraw','materialize','source-audit','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action in ['predraw','bank']:globals()[a.action](a.seed)
    elif a.action=='source-audit':
        from audit_uniform32 import sources
        sources()
    else:globals()[a.action]()
