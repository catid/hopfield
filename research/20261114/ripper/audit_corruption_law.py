"""Independent H1bn corruption laws, common masks, sources and LP proofs."""
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json,time
from fractions import Fraction
import numpy as np
from corruption_law_study import ROOT,HERE,OUT,PUBLIC,REGISTRY,PROTOCOL,PARENT,ARMS,sha,save,npz,verify
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs


def reference_masks(rank_seed,rows=124):
    rng=np.random.default_rng(rank_seed);given=np.zeros((rows,81),bool)
    for row in range(rows):
        order=list(map(int,rng.permutation(81)))
        for cell in order[:72]:given[row,cell]=True
    return dict(given=given)


def reference_corrupt(clean,noise,arm):
    result=clean.copy()
    if arm=='bursty':
        for row in range(len(clean)):
            if Fraction(float(noise['bursty_uniforms'][row]))<Fraction(9,160):
                for col,source in enumerate(noise['bursty_permutations'][row]):result[row,col]=clean[row,int(source)]
    else:
        p=Fraction(.05) if arm=='iid5' else Fraction(float.fromhex('0x1.756d7561abf76p-6'))
        for row in range(len(clean)):
            for col in range(81):
                offset=int(noise[arm+'_offsets'][row,col]);assert 1<=offset<=8
                if Fraction(float(noise[arm+'_uniforms'][row,col]))<p:result[row,col]=(int(clean[row,col])+offset)%9
    return result


def reference_observe(given,clean,noise,arm):
    corrupted=reference_corrupt(clean,noise,arm);values=np.full(clean.shape,-1,np.int8);values[given]=corrupted[given];ii,jj=np.triu_indices(81,1);joint=given[:,ii]&given[:,jj]
    return dict(given=given,values=values,joint_count=joint.sum(0),equal_count=(joint&(values[:,ii]==values[:,jj])).sum(0))


def noise_audit(noise,bases,bank,rows=124):
    rng=np.random.default_rng(bases['bursty']+bank)
    for row in range(rows):assert rng.random()==noise['bursty_uniforms'][row] and np.array_equal(rng.permutation(81),noise['bursty_permutations'][row])
    for arm in ['iid5','iid_pair']:
        rng=np.random.default_rng(bases[arm]+bank)
        assert np.array_equal(rng.random((rows,81)),noise[arm+'_uniforms']) and np.array_equal(rng.integers(1,9,(rows,81),dtype=np.int8),noise[arm+'_offsets'])


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2345051 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2108626 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-matched-coquery.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1486943 and draws.shape==(32,124) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all32_common_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==32
    assert [e['seed'] for e in manifest['inputs']]==list(range(32))
    for seed in range(32):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,124,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==124
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2108626+len(unique),updated_combined_exclusion_union=2345051+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=32*124*81),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(124,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
        grids=boards.reshape(-1,9,9);boxes=grids.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert ((grids==digit).sum(1)==1).all() and ((grids==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
        for index,board in zip(draws[seed],boards,strict=True):
            index=int(index)
            if index in wanted:assert np.array_equal(wanted[index],board)
            else:wanted[index]=board
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for index,fields in enumerate(reader):
            if index not in remaining:continue
            board=np.array([int(c)-1 for c in fields[column]],np.int8);assert np.array_equal(board,wanted[index]);remaining.remove(index)
            if not remaining:break
    assert not remaining
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),identity_seal_sha256=sha(OUT/'identity-seal.json'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all32_common_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def complete_gate(rank_seed,clean,noise,expected):
    from corruption_law_study import make_masks,observe
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=clean).all();masks=make_masks(rank_seed,len(clean));gates=[]
    for arm in ARMS:
        n=len(clean);actual=observe(masks['given'],changed,noise,arm);old=expected[arm];given=old['given']
        assert np.array_equal(actual['given'],given) and np.array_equal(actual['joint_count'],old['joint_count']) and (actual['values'][given]!=old['values'][given]).all() and not np.array_equal(actual['equal_count'],old['equal_count'])
        gates.append(dict(arm=arm,status='passed',source_rows=n,all_source_and_queried_values_changed=True,every_mask_and_joint_count_unchanged=True,equality_counts_changed=True,original_source_sha256=hashlib.sha256(clean[:n].tobytes()).hexdigest(),changed_source_sha256=hashlib.sha256(changed[:n].tobytes()).hexdigest()))
    return gates


def literal_account(given):
    sets={arm:{(arm,row,int(cell)) for row,mask in enumerate(given) for cell in np.flatnonzero(mask)} for arm in ARMS}
    union=set().union(*sets.values());total={arm+'_queries':len(values) for arm,values in sets.items()}
    assert sum(total.values())==len(union)
    total.update(shared_noisy_queries=0,new_union_queries=len(union));return total


def audit_bank(seed,public=False):
    p=verify();base=PUBLIC if public else OUT;d=base/f's{seed}';masked=npz(d/'masks.npz');predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['masks_sha256']==sha(d/'masks.npz')
    masks=reference_masks(p['rank_seed_base']+seed)
    for key,value in masks.items():assert np.array_equal(value,masked[key])
    assert masks['given'].shape==(124,81) and (masks['given'].sum(1)==72).all()
    clean=noise=None;expected={}
    if not public:
        entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert sha(ROOT/entry['completion_path'])==entry['completion_sha256'];clean=npz(ROOT/entry['completion_path'])['solutions'];noise=npz(d/'potential_noise.npz');noise_audit(noise,p['noise_seed_bases'],seed)
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[]
    for arm_index,arm in enumerate(ARMS):
        index=3*seed+arm_index;given=masks['given'];n=len(given);a=npz(d/f'{arm}-models.npz');r=json.loads((d/f'{arm}-receipt.json').read_text());assert r['index']==index and r['seed']==seed and r['arm']==arm and r['protocol_sha256']==sha(PROTOCOL) and len(r['records'])==40
        assert r['models_sha256']==sha(d/f'{arm}-models.npz') and r['predraw_receipt_sha256']==sha(d/'predraw-receipt.json');joint=(given[:,ii]&given[:,jj]).sum(0);assert np.array_equal(joint,a['joint_count']) and int(joint.sum())==316944
        if not public:
            assert r['observations_sha256']==sha(d/f'{arm}-observations.npz') and r['noise_sha256']==sha(d/'potential_noise.npz');observed=npz(d/f'{arm}-observations.npz');expected[arm]=reference_observe(given,clean,noise,arm)
            for key,value in expected[arm].items():assert np.array_equal(value,observed[key])
            assert np.array_equal(expected[arm]['equal_count'],a['equal_count'])
            for record in r['records']:assert record['log_sha256']==sha(d/f'{arm}-d{record["degree"]:02d}.log')
        cost=independent_rate_costs(joint,a['equal_count']);assert np.array_equal(cost,a['cost']);row=grid(a,r['records'],cost,p['solver'],index);selected=row['selected_degree']
        outcome=statistics(np.empty(0,np.int64) if selected is None else a[f'd{selected:02d}_primal_twice'],truth)
        if selected is not None and outcome['retained'] is not None:assert outcome['retained']==81*selected//2
        fixed=statistics(a['d20_primal_twice'],truth) if r['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,a['top810_support'])
        total=sum(map(int,joint));squares=sum(int(v)**2 for v in joint)
        exposure=dict(minimum=int(joint.min()),maximum=int(joint.max()),sum=total,sum_squares=squares,variance_numerator=3240*squares-total**2,variance_denominator=3240**2)
        rows.append(dict(index=index,seed=seed,arm=arm,**row,**outcome,coquery_summary=exposure,true_pair_equalities=int(a['equal_count'][truth].sum()),nonpeer_pair_equalities=int(a['equal_count'][~truth].sum()),fixed_degree20=fixed,top810=statistics(2*top.astype(np.int64),truth)))
    gates=[]
    if not public:
        gates=complete_gate(p['rank_seed_base']+seed,clean,noise,expected)
    files=['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]
    if not public:files+=['potential_noise.npz']+[arm+'-observations.npz' for arm in ARMS]
    realized={} if public else {arm:dict(full_cell_errors=int((reference_corrupt(clean,noise,arm)!=clean).sum()),queried_cell_errors=int((expected[arm]['values'][masks['given']]!=clean[masks['given']]).sum())) for arm in ARMS}
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,query_accounting=literal_account(masks['given']),realized_noise=realized,complete_label_gates=gates,artifact_sha256={name:sha(d/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(96));groups=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},selected_degree20=sum(r['selected_degree']==20 for r in part),
            selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part),fixed_degree20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part)))
    bounds={g['arm']:(g['success'],g['success']+g['unresolved'],31) for g in groups if g['arm']!='bursty'}
    endpoints={name:dict(decision='supported' if low>=required else 'falsified' if high<required else 'inconclusive',lower=low,upper=high,required=required) for name,(low,high,required) in bounds.items()}
    paired={}
    for ai,arm in enumerate(ARMS[1:],1):
        result=dict(candidate_wins=[],bursty_wins=[],both_success=[],both_resolved_failure=[],any_unresolved=[])
        for seed in range(32):
            b,c=[rows[3*seed+i]['classification'] for i in [0,ai]]
            key='any_unresolved' if 'unresolved' in [b,c] else 'both_success' if b==c=='success' else 'candidate_wins' if c=='success' else 'bursty_wins' if b=='success' else 'both_resolved_failure';result[key].append(seed)
        paired[arm]=result
    return groups,endpoints,paired


def public_job(seed):return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL);identity,_,_=identity_audit(True)
    for key,value in identity.items():assert saved['source_audit'][key]==value
    for path,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/path)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(32)))
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);assert rows==saved['rows'] and grouped==saved['grouped'] and endpoints==saved['endpoints'] and paired==saved['paired_changes']
    q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};q.update(new_donor_queries=0,h1az_through_present_sequence_union=8190009+q['new_union_queries'],clean_unique_oracle_cells=identity['clean_unique_oracle_cells'],clean_materialized_cell_instances=321408)
    assert q==saved['query_accounting'] and q['new_union_queries']==857088 and q['shared_noisy_queries']==0 and all(q[arm+'_queries']==285696 for arm in ARMS)
    assert saved['certified_LPs']==sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected']==sum(r['mutations_rejected'] for r in rows)
    print(json.dumps(dict(status='passed',endpoints=endpoints,grouped=grouped,paired_changes=paired,source_identity=identity,query_accounting=q,certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
        scope='Fresh-ID draws/exclusions, common72cell masks, separately charged corruption laws, costs/proofs and separate finite endpoints. OriginalCSV/recipient labels/noise replay and full-label gates require raw audit.')))


if __name__=='__main__':public_audit()
