"""H1bm: equal pair coquery totals with more cells per board versus more boards."""
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,shutil
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz,zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate,top_count_support
from laplace_rate_math import rate_costs
OUT=ROOT/'artifacts/research/20261114/ripper/matched_coquery'
REGISTRY=OUT.parent/'matched_coquery_registration'
PUBLIC=ROOT/'results/research/ripper/matched_coquery'
PROTOCOL=HERE/'protocol-matched-coquery.json'
PARENT=ROOT/'results/research/ripper/coverage_tradeoff'
ARMS=['sparse','dense']


def make_masks(rank_seed,rows=639):
    assert rows>=124
    rng=np.random.default_rng(rank_seed);sparse=np.zeros((rows,81),bool);dense=np.zeros((124,81),bool)
    for row in range(rows):
        order=rng.permutation(81);sparse[row,order[:32]]=True
        if row<124:dense[row,order[:72]]=True
    return dict(sparse=sparse,dense=dense)


def observe(given,clean,noise):
    assert given.shape==clean.shape and given.ndim==2 and given.shape[1]==81
    values=np.full(given.shape,-1,np.int8);joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint)
    for row in range(len(given)):
        board=clean[row,noise['permutations'][row]] if noise['uniforms'][row]<9/160 else clean[row]
        cells=np.flatnonzero(given[row]);values[row,cells]=board[cells];block=np.ix_(cells,cells)
        joint[block]+=1;equal[block]+=board[cells,None]==board[None,cells]
    ii,jj=np.triu_indices(81,1)
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj])


def verify(raw=False):
    p=json.loads(PROTOCOL.read_text())
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    if raw:
        for rel,digest in p['raw_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists() and not PUBLIC.exists()
    gate=HERE/'matched-coquery-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    old=json.loads((HERE/'protocol-coverage-tradeoff.json').read_text());oldids=npz(PARENT/'excluded_ids.npz');selected=npz(PARENT/'selected_sources.npz')['all_source_row']
    excluded=np.union1d(oldids['excluded'],selected);ripper=np.union1d(oldids['ripper_prior'],selected)
    assert len(excluded)==2324738 and len(ripper)==2088313 and np.isin(ripper,excluded).all()
    previous=json.loads((PARENT/'source-audit.json').read_text());assert previous['updated_ripper_ledger_union']==len(ripper) and previous['updated_combined_exclusion_union']==len(excluded)
    REGISTRY.mkdir();np.savez_compressed(REGISTRY/'excluded_ids.npz',excluded=excluded,ripper_prior=ripper)
    ledger=dict(status='sealed_before_new_draws',scope='Declared combined Ripper/H4ak exclusions through H1bl only; not global repository untouchedness.',excluded_unique=len(excluded),ripper_prior_unique=len(ripper),eligible_rows=old['pool_size']-len(excluded),
                parent_protocol_sha256=sha(HERE/'protocol-coverage-tradeoff.json'),parent_source_audit_sha256=sha(PARENT/'source-audit.json'),parent_selected_ids_sha256=sha(PARENT/'selected_sources.npz'),parent_exclusions_sha256=sha(PARENT/'excluded_ids.npz'),excluded_file_sha256=sha(REGISTRY/'excluded_ids.npz'))
    save(REGISTRY/'ledger.json',ledger)
    inputs=[gate,REGISTRY/'excluded_ids.npz',REGISTRY/'ledger.json',PARENT/'excluded_ids.npz',PARENT/'selected_sources.npz',PARENT/'source-audit.json',PARENT/'summary.json',HERE/'protocol-coverage-tradeoff.json']
    p=dict(id='H1bm-matched-pair-coquery',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,rows=639,arms=ARMS,source=old['source'],source_csv_sha256=old['source_csv_sha256'],pool_size=old['pool_size'],
           source_seed_base=202612170000,noise_seed_base=202612180000,rank_seed_base=202612190000,
           scope='Prospective32fresh639row paired banks outside declared combined2,324,738ID ledger only. Same official corpus. Sparse32cells on639rows versus dense72cells on first124rows; exactly316,944paircoqueries perarm/perbank; no donors.',
           sampling='Independent uniform639withoutreplacement draw perbank from sortedeligibleIDs; all639identities sealed before all64masks, then selectedCSVlabels. Keep/report crossbank repeats. Dense arm uses fixed first124 prefix, no row selection.',
           acquisition='One fresh uniform81cell order perrow. Sparse uses first32 on all639rows; dense first72 on first124rows. Thus32subset72 on shared rows; no dense queries on suffix515. No label feedback.',
           noise='Fresh mask-independent U and uniform81position permutation perrow; permute entireboard iffU<9/160. Shared first124latentboards acrossarms; dense uses literal prefix. Algorithm receives no corruption probability.',
           degrees=old['degrees'],candidate_degrees=old['candidate_degrees'],costs=old['costs'],objective=old['objective'],selector=old['selector'],proof=old['proof'],
           prior='Common degree and bounded grid0..40 on81nodes; no supplied20,810,geometry,donorinformation or preferreddegree tie rule.',
           dense_endpoint=dict(required=31,success='Dense selected graph certified integral exact after all grid degrees certify.'),dense_advantage_endpoint=dict(required=2,success='Dense exact count minus sparse exact count is at least2.'),
           decision='Separate finite targets. Dense lower=sD,upper=sD+uD. Advantage lower=sD-sS-uS,upper=sD+uD-sS. Supported iff lower>=target; falsified iff upper<target; otherwise inconclusive. Any uncertified grid degree makes bank unresolved; selected fractional/nonexact is graph failure. Resource failure incomplete.',
           diagnostics='Degree20 selections,all paired wins/losses,fixeddegree20 grid diagnostic,top810,coquery count minimum/maximum/sum/sum-of-squares and exact variance numerator. BP first. Matched totalcoqueries do not match perpair countvectors or dependencies; different row counts, label budgets and correlated pair observations remain explicit. Prior H1bl equal-label result unpaired context only. No unique causal-mediation claim.',
           solver=dict(old['solver'],random_seed_base=2026191502),seed_rule='Index=2*bank+arm_index; seed=base+index*41+d; fresh solver every degree including20.',
           budget=dict(max_cpu_workers=8,new_LPs=2560,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0,new_physical_labels_cap=813056),
           budget_scope='IDseal,64maskstreams/materialization/sourceaudit,64acquisitions/grids,independent masks/values/costs/proofs/substitution gates,unions and summary. Registration/synthetic gate/packaging separate.',
           query_accounting=dict(sparse_complete_deployment=654336,dense_complete_deployment=285696,sparse_perbank=20448,dense_perbank=8928,each_arm_pair_coqueries_perbank=316944,paired_shared=126976,new_paired_union=813056,new_donor_queries=0,prior_h1az_through_h1bl_sequence_union=7376953,prior_ripper_ledger_union=2088313,prior_combined_exclusion_union=2324738),
           physical_query_rule='Actual union perbank=639*32+124*72-124*32=25,408; all32banks813,056. Sparse deployment654,336; dense285,696. Equalcoqueries639*C(32,2)=124*C(72,2)=316,944 perbank. No donor/calibration cost. Crossbank repeatedsources have separate latentinstances and chargedseparately.',
           source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},raw_input_sha256={old['source']:old['source_csv_sha256']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),excluded_ids=len(excluded),eligible_rows=ledger['eligible_rows'])))


def select():
    p=verify();assert not OUT.exists();excluded=npz(REGISTRY/'excluded_ids.npz')['excluded'];eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True);assert len(eligible)==1507256
    draws=np.stack([np.random.default_rng(p['source_seed_base']+seed).choice(eligible,639,replace=False) for seed in range(32)]);OUT.mkdir();np.savez_compressed(OUT/'selected_sources.npz',source_rows=draws,all_source_row=np.unique(draws))
    save(OUT/'identity-seal.json',dict(status='sealed_before_selected_labels',protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),selected_sources_sha256=sha(OUT/'selected_sources.npz'),new_unique_source_ids=len(np.unique(draws)),crossbank_repeated_instances=draws.size-len(np.unique(draws))))


def predraw(seed):
    p=verify();assert (OUT/'identity-seal.json').exists() and not (OUT/'input_manifest.json').exists();d=OUT/f's{seed}';d.mkdir(exist_ok=False)
    arrays=make_masks(p['rank_seed_base']+seed);np.savez_compressed(d/'masks.npz',**arrays)
    save(d/'predraw-receipt.json',dict(status='sealed_before_selected_labels',seed=seed,protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),identity_seal_sha256=sha(OUT/'identity-seal.json'),masks_sha256=sha(d/'masks.npz')))


def selected_answers(source,wanted,row_limit=None):
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for index,fields in enumerate(reader):
            if row_limit is not None and index>=row_limit:break
            if index in wanted:
                yield index,np.frombuffer(fields[column].encode('ascii'),np.uint8).astype(np.int8)-ord('1')


def materialize():
    p=verify();predraws={}
    for seed in range(32):
        d=OUT/f's{seed}';r=json.loads((d/'predraw-receipt.json').read_text());assert r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['masks_sha256']==sha(d/'masks.npz');predraws[str(seed)]=sha(d/'predraw-receipt.json')
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];draws=npz(OUT/'selected_sources.npz')['source_rows'];wanted={};solutions=np.zeros((32,639,81),np.int8)
    for seed in range(32):
        for offset,row in enumerate(draws[seed]):wanted.setdefault(int(row),[]).append((seed,offset))
    remaining=set(wanted)
    for index,board in selected_answers(source,wanted):
        assert board.shape==(81,)
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
    rng=np.random.default_rng(p['noise_seed_base']+seed);uniforms=np.zeros(639);permutations=np.zeros((639,81),np.uint8)
    for row in range(639):uniforms[row]=rng.random();permutations[row]=rng.permutation(81)
    noise=dict(uniforms=uniforms,permutations=permutations);np.savez_compressed(d/'potential_noise.npz',**noise);ii,jj=np.triu_indices(81,1)
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;n=len(m[arm]);observed=observe(m[arm],clean[:n],{k:v[:n] for k,v in noise.items()});np.savez_compressed(d/f'{arm}-observations.npz',**observed)
        cost=rate_costs(observed['joint_count'],observed['equal_count']);a=dict(cost=cost,joint_count=observed['joint_count'],equal_count=observed['equal_count'],top810_support=top_count_support(np.ones(3240,bool),cost));records=[];a.update({'d00_'+k:v for k,v in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,80000+index*41+degree,d/f'{arm}-d{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            a.update({f'd{degree:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**w).items()});records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'{arm}-d{degree:02d}.log')))
        np.savez_compressed(d/f'{arm}-models.npz',**a);save(d/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/f'{arm}-models.npz'),observations_sha256=sha(d/f'{arm}-observations.npz'),noise_sha256=sha(d/'potential_noise.npz'),predraw_receipt_sha256=sha(d/'predraw-receipt.json')))
    from audit_matched_coquery import audit_bank
    a=audit_bank(seed,False);save(d/'audit.json',a);print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def summary():
    from audit_matched_coquery import aggregate
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);audits=[];files={}
    for seed in range(32):
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_matched_coquery.py')
        for name,digest in a['artifact_sha256'].items():assert sha(d/name)==digest
        audits.append(a);dest=PUBLIC/f's{seed}';dest.mkdir()
        for name in ['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]:shutil.copyfile(d/name,dest/name);files[f's{seed}/{name}']=sha(d/name)
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert q['sparse_queries']==654336 and q['dense_queries']==285696 and q['shared_queries']==126976 and q['new_union_queries']==813056
    q.update(new_donor_queries=0,h1az_through_present_sequence_union=7376953+q['new_union_queries'])
    for name in ['selected_sources.npz','identity-seal.json','input_manifest.json','all_source_manifest.json','source-audit.json']:shutil.copyfile(OUT/name,PUBLIC/name)
    shutil.copyfile(REGISTRY/'excluded_ids.npz',PUBLIC/'excluded_ids.npz');shutil.copyfile(REGISTRY/'ledger.json',PUBLIC/'ledger.json')
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],grouped=grouped,endpoints=endpoints,paired_changes=paired,rows=rows,raw_audits=audits,query_accounting=q,alternative_query_accounting=p['query_accounting'],source_audit=json.loads((OUT/'source-audit.json').read_text()),
                certified_LPs=sum(r['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),analytic_zero_proofs=64,complete_label_gates=64,artifact_sha256=files,all_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(grouped=grouped,endpoints=endpoints,query_accounting=q)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','select','predraw','materialize','source-audit','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action in ['predraw','bank']:globals()[a.action](a.seed)
    elif a.action=='source-audit':
        from audit_matched_coquery import sources
        sources()
    else:globals()[a.action]()
