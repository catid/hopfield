"""H1bj: seven additional charged cells per fixed recipient row."""
from datetime import datetime,timezone
import argparse,json,os,shutil,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz,zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate,top_count_support
from laplace_rate_math import rate_costs
OUT=ROOT/'artifacts/research/20261114/ripper/extra_coverage'
PUBLIC=ROOT/'results/research/ripper/extra_coverage'
PARENT=ROOT/'artifacts/research/20261114/ripper/fresh_transfer'
OLD_PUBLIC=ROOT/'results/research/ripper/fresh_transfer_certificates.npz'
BASELINE=ROOT/'results/research/ripper/partial_degree/summary.json'
PROTOCOL=HERE/'protocol-extra-coverage.json'
ARMS=['aligned','permuted']


def extend(old,seed):
    assert old.dtype==bool and old.ndim==2 and old.shape[1]==81 and (old.sum(1)==25).all()
    result=old.copy();rng=np.random.default_rng(seed)
    for row in range(len(old)):
        order=rng.permutation(81);result[row,order[~old[row,order]][:7]]=True
    assert (result.sum(1)==32).all() and not (old&~result).any()
    return result


def acquire(old,seed,clean,noise):return observe(extend(old,seed),clean,noise)


def observe(given,clean,noise):
    assert given.shape==clean.shape and (given.sum(1)==32).all()
    values=np.full(given.shape,-1,np.int8);joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint)
    for row in range(len(given)):
        board=clean[row,noise['permutations'][row]] if noise['uniforms'][row]<9/160 else clean[row]
        cells=np.flatnonzero(given[row]);values[row,cells]=board[cells];block=np.ix_(cells,cells)
        joint[block]+=1;equal[block]+=board[cells,None]==board[None,cells]
    ii,jj=np.triu_indices(81,1)
    return dict(given=given,values=values,joint_count=joint[ii,jj],equal_count=equal[ii,jj])


def accounting(olda,oldp,newa,newp):
    previous=olda|oldp;both=newa|newp
    result=dict(old_aligned=int(olda.sum()),old_permuted=int(oldp.sum()),old_shared=int((olda&oldp).sum()),old_union=int(previous.sum()),
                aligned_total=int(newa.sum()),permuted_total=int(newp.sum()),new_shared_total=int((newa&newp).sum()),new_union_total=int(both.sum()),
                aligned_added_to_own=int((newa&~olda).sum()),permuted_added_to_own=int((newp&~oldp).sum()),
                aligned_new_to_prior_union=int((newa&~previous).sum()),permuted_new_to_prior_union=int((newp&~previous).sum()),
                newly_shared=int((newa&newp&~previous).sum()),actual_increment=int((both&~previous).sum()))
    return result


def verify(raw=False):
    p=json.loads(PROTOCOL.read_text())
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    if raw:
        for rel,digest in p['raw_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=HERE/'extra-coverage-preflight.json';g=json.loads(gate.read_text());assert g['status']=='passed'
    for rel,digest in g['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    prior=json.loads((HERE/'protocol-partial-degree.json').read_text());manifest=json.loads((PARENT/'input_manifest.json').read_text())
    raw=[PARENT/'input_manifest.json',PARENT/'source-audit.json',PARENT/'all_source_manifest.json']+[ROOT/e['completion_path'] for e in manifest['inputs']]
    raw+=[PARENT/f's{seed}'/name for seed in range(32) for name in ['potential_noise.npz','aligned-observations.npz','permuted-observations.npz']]
    inputs=[gate,OLD_PUBLIC,BASELINE,HERE/'protocol-fresh-transfer.json',HERE/'protocol-partial-degree.json',ROOT/'results/research/ripper/fresh_transfer.json',ROOT/'results/research/ripper/partial_degree_public_manifest.json']
    p=dict(id='H1bj-extra-coverage-on-fixed-recipients',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,rows=900,arms=ARMS,extension_seed_base=202611920000,
           scope='Prospective additional charged observations on examined H1bh identities and unchanged latent instances. Seven added cells per row from existing hidden source values; no new identity/source or corruption draw. Larger-budget test cannot reverse H1bh original-budget failures.',
           acquisition='Retain each entire old25-cell mask. Perbank,row draw one new uniform81-cell order, shared by paired arms. Append first7 cells absent from that arm old mask. All64 resulting32-cell streams persisted and globally sealed before extra labels are read. No recipient-label feedback.',
           labels='Reuse original clean boards and same mask-independent full-board corruption. Previously queried coordinates preserve literal value. This is simulated observation accounting on archived complete boards, not globally unseen data.',
           degrees=list(range(41)),candidate_degrees=list(range(1,40)),costs=prior['costs'],objective=prior['objective'],selector=prior['selector'],prior=prior['prior'],proof=prior['proof'],
           degree_endpoint=dict(arm='aligned',required=31,success='Select degree20 with all grid proofs certified.'),graph_endpoint=dict(arm='aligned',required=31,success='Selected graph certified integral and exactly equals evaluation truth.'),
           decision=prior['decision'],diagnostics='Permuted descriptive, no alignment endpoint. Report all paired gains/losses versus archivedH1bi; no unregistered superiority or significance claim. Fixeddegree20 from freshgrid and top810 controls; BP first.',
           solver=dict(prior['solver'],random_seed_base=2026161502),seed_rule='Index=2*bank+arm_index; seed=base+index*41+d; fresh solver for every degree including20.',
           budget=dict(max_cpu_workers=8,new_LPs=2560,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0,new_physical_labels_cap=403200),
           budget_scope='All32paired mask predraws/global seal,64acquisitions/grids,independent masks/labels/costs/full-label gates/proofs/mutations,query accounting and export. Synthetic gate/registration/packaging separate.',
           query_accounting=dict(each_arm_added=201600,each_arm_recipient_total=921600,each_arm_historical_donor=240000,each_arm_complete_deployment=1161600,new_sources=0,new_donor_queries=0,unchanged_ripper_ledger_union=2045471,prior_h1az_through_h1bh_sequence_union=5148181,prior_paired_recipient_union=1217792),
           physical_query_rule='Count bank,row,cell instances. Increment = union(extended masks) minus union(old masks), preserving all cross-arm reuse; each own-arm extension201600 is not actual paired increment. Crossbank repeated sourceIDs carry separate fixed latents and remain distinct instances. Historical donor warmups reused and charged in standalone deployment.',
           source_sha256=g['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs},raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in raw})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def predraw(seed):
    p=verify();assert not (OUT/'mask-seal.json').exists();d=OUT/f's{seed}';d.mkdir(exist_ok=False);arrays={}
    with np.load(OLD_PUBLIC,allow_pickle=False) as z:
        for arm in ARMS:
            old=z[f's{seed}_{arm}_mask'].copy();arrays[arm+'_old']=old;arrays[arm+'_new']=extend(old,p['extension_seed_base']+seed)
    np.savez_compressed(d/'masks.npz',**arrays)
    save(d/'predraw-receipt.json',dict(status='sealed_before_additional_labels',seed=seed,protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),masks_sha256=sha(d/'masks.npz')))


def seal():
    p=verify();paths={}
    for seed in range(32):
        d=OUT/f's{seed}';r=json.loads((d/'predraw-receipt.json').read_text());assert r['seed']==seed and r['status']=='sealed_before_additional_labels' and r['protocol_sha256']==sha(PROTOCOL) and r['masks_sha256']==sha(d/'masks.npz')
        assert not any(d.glob('*-models.npz'));paths[str(seed)]=sha(d/'predraw-receipt.json')
    save(OUT/'mask-seal.json',dict(status='all64_streams_sealed_before_additional_labels',protocol_sha256=sha(PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),predraw_receipts_sha256=paths))


def bank(seed):
    p=verify(True);d=OUT/f's{seed}';sealed=json.loads((OUT/'mask-seal.json').read_text());assert sealed['status']=='all64_streams_sealed_before_additional_labels' and sealed['protocol_sha256']==sha(PROTOCOL) and len(sealed['predraw_receipts_sha256'])==32
    assert sealed['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json');r=json.loads((d/'predraw-receipt.json').read_text());assert r['masks_sha256']==sha(d/'masks.npz');masks=npz(d/'masks.npz')
    manifest=json.loads((PARENT/'input_manifest.json').read_text());entry=manifest['inputs'][seed];assert entry['seed']==seed;clean=npz(ROOT/entry['completion_path'])['solutions'];noise=npz(PARENT/f's{seed}'/'potential_noise.npz');ii,jj=np.triu_indices(81,1)
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;observed=observe(masks[arm+'_new'],clean,noise);np.savez_compressed(d/f'{arm}-observations.npz',**observed)
        cost=rate_costs(observed['joint_count'],observed['equal_count']);arrays=dict(cost=cost,joint_count=observed['joint_count'],equal_count=observed['equal_count'],top810_support=top_count_support(np.ones(3240,bool),cost));records=[]
        arrays.update({'d00_'+k:v for k,v in zero_witness(81,cost).items()})
        for degree in range(1,41):
            x,y,solver=solve(81,degree,ii,jj,cost,50000+index*41+degree,d/f'{arm}-d{degree:02d}.log')
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:cert,w=attempt_certificate(81,degree,ii,jj,cost,x,y)
            else:cert=dict(certified=False,reason='solver_status_or_validity');w={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            arrays.update({f'd{degree:02d}_'+k:v for k,v in dict(primal_values=x,node_dual_values=y,**w).items()});records.append(dict(degree=degree,solver=solver,certificate=cert,log_sha256=sha(d/f'{arm}-d{degree:02d}.log')))
            save(d/'progress.json',dict(seed=seed,arm=arm,complete_degrees=40*arm_index+degree,pid=os.getpid(),phase='fit'))
        np.savez_compressed(d/f'{arm}-models.npz',**arrays)
        save(d/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),records=records,models_sha256=sha(d/f'{arm}-models.npz'),observations_sha256=sha(d/f'{arm}-observations.npz'),mask_seal_sha256=sha(OUT/'mask-seal.json'),predraw_receipt_sha256=sha(d/'predraw-receipt.json')))
    save(d/'accounting.json',accounting(*[masks[k] for k in ['aligned_old','permuted_old','aligned_new','permuted_new']]))
    from audit_extra_coverage import audit_bank
    a=audit_bank(seed,False);save(d/'audit.json',a);save(d/'progress.json',dict(seed=seed,complete_degrees=80,pid=os.getpid(),phase='audited'));print(json.dumps(dict(seed=seed,rows=a['rows'])),flush=True)


def summary():
    from audit_extra_coverage import aggregate
    p=verify(True);assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);audits=[];files={}
    shutil.copyfile(OUT/'mask-seal.json',PUBLIC/'mask-seal.json')
    for seed in range(32):
        d=OUT/f's{seed}';a=json.loads((d/'audit.json').read_text());assert a['status']=='validated' and a['seed']==seed and a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_extra_coverage.py')
        for name,digest in a['artifact_sha256'].items():assert sha(d/name)==digest
        audits.append(a);dest=PUBLIC/f's{seed}';dest.mkdir()
        for name in ['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]:
            shutil.copyfile(d/name,dest/name);files[f's{seed}/{name}']=sha(d/name)
    rows=[row for a in audits for row in a['rows']];grouped,endpoints,paired=aggregate(rows);queries={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']}
    assert queries['old_union']==1217792 and queries['actual_increment']<=p['budget']['new_physical_labels_cap']
    for arm in ARMS:assert queries[arm+'_added_to_own']==201600 and queries[arm+'_total']==921600
    queries['h1az_through_present_sequence_union']=5148181+queries['actual_increment'];queries['new_sources']=queries['new_donor_queries']=0
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],grouped=grouped,endpoints=endpoints,paired_changes=paired,rows=rows,query_accounting=queries,alternative_query_accounting=p['query_accounting'],
                certified_LPs=sum(r['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),analytic_zero_proofs=64,complete_label_gates=64,raw_audits=audits,
                artifact_sha256=files,mask_seal_sha256=sha(PUBLIC/'mask-seal.json'),all_banks_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(grouped=grouped,endpoints=endpoints,paired_changes=paired,query_accounting=queries)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','predraw','seal','bank','summary']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.action in ['predraw','bank']:globals()[a.action](a.seed)
    else:globals()[a.action]()
