"""H1bm independent fresh IDs, prefix masks, matched-coquery unions and LP proofs."""
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json,time
import numpy as np
from matched_coquery_study import ROOT,HERE,OUT,PUBLIC,REGISTRY,PROTOCOL,PARENT,ARMS,sha,save,npz,verify
from audit_extra_coverage import reference_observe,statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs


def reference_masks(rank_seed,rows=639):
    rng=np.random.default_rng(rank_seed);sparse=np.zeros((rows,81),bool);dense=np.zeros((124,81),bool)
    for row in range(rows):
        order=list(map(int,rng.permutation(81)))
        for cell in order[:32]:sparse[row,cell]=True
        if row<124:
            for cell in order[:72]:dense[row,cell]=True
    return dict(sparse=sparse,dense=dense)


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2324738 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2088313 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-coverage-tradeoff.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1507256 and draws.shape==(32,639) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all64_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==32
    assert [e['seed'] for e in manifest['inputs']]==list(range(32))
    for seed in range(32):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,639,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==639
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2088313+len(unique),updated_combined_exclusion_union=2324738+len(unique)),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(639,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
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
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all64_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def complete_gate(rank_seed,clean,noise,expected):
    from matched_coquery_study import make_masks,observe
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=clean).all();masks=make_masks(rank_seed,len(clean));gates=[]
    for arm in ARMS:
        n=len(masks[arm]);actual=observe(masks[arm],changed[:n],{k:v[:n] for k,v in noise.items()});old=expected[arm];given=old['given']
        assert np.array_equal(actual['given'],given) and np.array_equal(actual['joint_count'],old['joint_count']) and (actual['values'][given]!=old['values'][given]).all() and not np.array_equal(actual['equal_count'],old['equal_count'])
        gates.append(dict(arm=arm,status='passed',source_rows=n,all_source_and_queried_values_changed=True,every_mask_and_joint_count_unchanged=True,equality_counts_changed=True,original_source_sha256=hashlib.sha256(clean[:n].tobytes()).hexdigest(),changed_source_sha256=hashlib.sha256(changed[:n].tobytes()).hexdigest()))
    return gates


def literal_account(sparse,dense):
    total=dict(sparse_queries=0,dense_queries=0,shared_queries=0,new_union_queries=0)
    for row,s in enumerate(sparse):
        ss=set(map(int,np.flatnonzero(s)));ds=set(map(int,np.flatnonzero(dense[row]))) if row<len(dense) else set()
        total['sparse_queries']+=len(ss);total['dense_queries']+=len(ds);total['shared_queries']+=len(ss&ds);total['new_union_queries']+=len(ss|ds)
    assert total['new_union_queries']==total['sparse_queries']+total['dense_queries']-total['shared_queries'];return total


def audit_bank(seed,public=False):
    p=verify();base=PUBLIC if public else OUT;d=base/f's{seed}';masked=npz(d/'masks.npz');predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['masks_sha256']==sha(d/'masks.npz')
    masks=reference_masks(p['rank_seed_base']+seed)
    for key,value in masks.items():assert np.array_equal(value,masked[key])
    assert masks['sparse'].shape==(639,81) and masks['dense'].shape==(124,81)
    assert (masks['sparse'].sum(1)==32).all() and (masks['dense'].sum(1)==72).all() and not (masks['sparse'][:124]&~masks['dense']).any()
    clean=noise=None;expected={}
    if not public:
        entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert sha(ROOT/entry['completion_path'])==entry['completion_sha256'];clean=npz(ROOT/entry['completion_path'])['solutions'];noise=npz(d/'potential_noise.npz');rng=np.random.default_rng(p['noise_seed_base']+seed)
        for row in range(639):assert rng.random()==noise['uniforms'][row] and np.array_equal(rng.permutation(81),noise['permutations'][row])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[]
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;given=masks[arm];n=len(given);a=npz(d/f'{arm}-models.npz');r=json.loads((d/f'{arm}-receipt.json').read_text());assert r['index']==index and r['seed']==seed and r['arm']==arm and r['protocol_sha256']==sha(PROTOCOL) and len(r['records'])==40
        assert r['models_sha256']==sha(d/f'{arm}-models.npz') and r['predraw_receipt_sha256']==sha(d/'predraw-receipt.json');joint=(given[:,ii]&given[:,jj]).sum(0);assert np.array_equal(joint,a['joint_count']) and int(joint.sum())==316944
        if not public:
            assert r['observations_sha256']==sha(d/f'{arm}-observations.npz') and r['noise_sha256']==sha(d/'potential_noise.npz');observed=npz(d/f'{arm}-observations.npz');expected[arm]=reference_observe(given,clean[:n],{k:v[:n] for k,v in noise.items()})
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
        rows.append(dict(index=index,seed=seed,arm=arm,**row,**outcome,coquery_summary=exposure,fixed_degree20=fixed,top810=statistics(2*top.astype(np.int64),truth)))
    gates=[]
    if not public:
        shared=masks['sparse'][:124]&masks['dense'];assert np.array_equal(expected['sparse']['values'][:124][shared],expected['dense']['values'][shared]);gates=complete_gate(p['rank_seed_base']+seed,clean,noise,expected)
    files=['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]
    if not public:files+=['potential_noise.npz']+[arm+'-observations.npz' for arm in ARMS]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,query_accounting=literal_account(masks['sparse'],masks['dense']),complete_label_gates=gates,artifact_sha256={name:sha(d/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(64));groups=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},selected_degree20=sum(r['selected_degree']==20 for r in part),
            selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part),fixed_degree20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part)))
    sparse,dense=groups;bounds={'dense_procedure':(dense['success'],dense['success']+dense['unresolved'],31),'dense_advantage':(dense['success']-sparse['success']-sparse['unresolved'],dense['success']+dense['unresolved']-sparse['success'],2)}
    endpoints={name:dict(decision='supported' if low>=required else 'falsified' if high<required else 'inconclusive',lower=low,upper=high,required=required) for name,(low,high,required) in bounds.items()}
    paired=dict(dense_wins=[],sparse_wins=[],both_success=[],both_resolved_failure=[],any_unresolved=[])
    for seed in range(32):
        s,d=[rows[2*seed+i]['classification'] for i in range(2)]
        key='any_unresolved' if 'unresolved' in [s,d] else 'both_success' if s==d=='success' else 'dense_wins' if d=='success' else 'sparse_wins' if s=='success' else 'both_resolved_failure';paired[key].append(seed)
    return groups,endpoints,paired


def public_job(seed):return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL);identity,_,_=identity_audit(True)
    for key,value in identity.items():assert saved['source_audit'][key]==value
    for path,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/path)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(32)))
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);assert rows==saved['rows'] and grouped==saved['grouped'] and endpoints==saved['endpoints'] and paired==saved['paired_changes']
    q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};q.update(new_donor_queries=0,h1az_through_present_sequence_union=7376953+q['new_union_queries']);assert q==saved['query_accounting'] and q['new_union_queries']==813056 and q['shared_queries']==126976 and q['sparse_queries']==654336 and q['dense_queries']==285696
    assert saved['certified_LPs']==sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected']==sum(r['mutations_rejected'] for r in rows)
    print(json.dumps(dict(status='passed',endpoints=endpoints,grouped=grouped,paired_changes=paired,source_identity=identity,query_accounting=q,certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
        scope='Fresh-ID draws/exclusions, prefix-nested32/72 masks, matched-coquery unions, costs/proofs and separate finite endpoints. OriginalCSV/recipient labels and full-label gates require raw audit.')))


if __name__=='__main__':public_audit()
