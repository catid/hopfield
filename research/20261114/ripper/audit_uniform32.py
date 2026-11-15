"""H1bk independent fresh IDs, donor lineage, masks and LP proof audit."""
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json,time
import numpy as np
from uniform32_study import ROOT,HERE,OUT,PUBLIC,REGISTRY,PROTOCOL,H4,DONORS,DONOR_RAW,ARMS,sha,save,npz,verify
from audit_transfer_feedback import reference_masks as donor_reference_masks
from audit_extra_coverage import reference_observe,statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs


def reference_masks(feedback,tie_seed,rank_seed,rows=900):
    old=donor_reference_masks(feedback,tie_seed,rows);rng=np.random.default_rng(rank_seed);uniform=np.zeros_like(old);donor=old.copy()
    for row in range(rows):
        order=list(map(int,rng.permutation(81)));uniform[row,order[:32]]=True;extra=[v for v in order if not old[row,v]][:7];donor[row,extra]=True
    return dict(donor_old25=old,uniform=uniform,donor=donor)


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY;ids=npz(H4/'source-identities.npz');excluded=np.unique(np.concatenate([ids['excluded_ids'],ids['source_rows'].ravel()]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2281896 and len(data['ripper_prior'])==2045471 and np.isin(data['ripper_prior'],excluded).all()
    ripper=np.union1d(npz(ROOT/'results/research/ripper/fresh_transfer_excluded_ids.npz')['excluded'],npz(ROOT/'results/research/ripper/fresh_transfer_selected_sources.npz')['all_source_row']);assert np.array_equal(ripper,data['ripper_prior'])
    ledger=json.loads((register/'ledger.json').read_text());h4=json.loads((H4/'summary.json').read_text());assert ledger['h4_raw_draw_sha256']==h4['source_audit']['source_draws_sha256'] and ledger['h4_public_id_sha256']==sha(H4/'source-identities.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    assert ledger['h4_protocol_sha256']==sha(ROOT/'research/20261114/foureyes/fresh_curvature.json') and ledger['h4_summary_sha256']==sha(H4/'summary.json')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1550098 and draws.shape==(32,900) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all64_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==32
    assert [e['seed'] for e in manifest['inputs']]==list(range(32))
    for seed in range(32):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,900,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==900
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2045471+len(unique),updated_combined_exclusion_union=2281896+len(unique)),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(900,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
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
    ii,jj=np.triu_indices(81,1);donorids=npz(DONOR_RAW/'selected_sources.npz')['source_rows'];excluded=npz(REGISTRY/'excluded_ids.npz')['excluded'];assert np.isin(donorids,excluded).all()
    with np.load(DONORS,allow_pickle=False) as z:feedbacks=[z[f's{seed}_donor_feedback'].copy() for seed in range(32)]
    assert sorted(p['donor_map'])==list(range(32));warmup=0
    for seed,donor in enumerate(p['donor_map']):
        z=npz(DONOR_RAW/f's{donor}'/'aligned-observations.npz');given=z['given'][:300];values=z['values'][:300];assert (given.sum(1)==25).all();warmup+=int(given.sum())
        equal=((given[:,ii]&given[:,jj])&(values[:,ii]==values[:,jj])).sum(0);feedback=np.zeros((81,81),np.int32);feedback[ii,jj]=equal;feedback[jj,ii]=equal
        assert np.array_equal(feedback,feedbacks[seed]) and not np.intersect1d(draws[seed],donorids[donor,:300]).size
    assert warmup==240000
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),identity_seal_sha256=sha(OUT/'identity-seal.json'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                **identity,independent_csv_answers_verified=len(wanted),historical_donor_warmup_queries=warmup,donor_recipient_overlaps=0,all64_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def complete_gate(feedback,tie_seed,rank_seed,clean,noise,expected):
    from uniform32_study import make_masks,observe
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=clean).all();masks=make_masks(feedback,tie_seed,rank_seed,len(clean));gates=[]
    for arm in ARMS:
        actual=observe(masks[arm],changed,noise);old=expected[arm];given=old['given']
        assert np.array_equal(actual['given'],given) and np.array_equal(actual['joint_count'],old['joint_count']) and (actual['values'][given]!=old['values'][given]).all() and not np.array_equal(actual['equal_count'],old['equal_count'])
        gates.append(dict(arm=arm,status='passed',all_source_and_queried_values_changed=True,every_mask_and_joint_count_unchanged=True,equality_counts_changed=True,original_source_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),changed_source_sha256=hashlib.sha256(changed.tobytes()).hexdigest()))
    return gates


def literal_account(uniform,donor):
    total=dict(uniform_queries=0,donor_queries=0,shared_queries=0,new_union_queries=0)
    for u,d in zip(uniform,donor,strict=True):
        us=set(map(int,np.flatnonzero(u)));ds=set(map(int,np.flatnonzero(d)));total['uniform_queries']+=len(us);total['donor_queries']+=len(ds);total['shared_queries']+=len(us&ds);total['new_union_queries']+=len(us|ds)
    assert total['new_union_queries']==total['uniform_queries']+total['donor_queries']-total['shared_queries'];return total


def audit_bank(seed,public=False):
    p=verify();base=PUBLIC if public else OUT;d=base/f's{seed}';masked=npz(d/'masks.npz');predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['masks_sha256']==sha(d/'masks.npz')
    with np.load(DONORS,allow_pickle=False) as z:feedback=z[f's{seed}_donor_feedback'].copy()
    assert np.array_equal(feedback,masked['feedback']);masks=reference_masks(feedback,p['tie_seed_base']+seed,p['rank_seed_base']+seed)
    for key,value in masks.items():assert np.array_equal(value,masked[key])
    for arm in ARMS:assert (masks[arm].sum(1)==32).all()
    assert not (masks['donor_old25']&~masks['donor']).any() and ((masks['donor']&~masks['donor_old25']).sum(1)==7).all()
    clean=noise=None;expected={}
    if not public:
        entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert sha(ROOT/entry['completion_path'])==entry['completion_sha256'];clean=npz(ROOT/entry['completion_path'])['solutions'];noise=npz(d/'potential_noise.npz');rng=np.random.default_rng(p['noise_seed_base']+seed)
        for row in range(900):assert rng.random()==noise['uniforms'][row] and np.array_equal(rng.permutation(81),noise['permutations'][row])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[]
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;given=masks[arm];a=npz(d/f'{arm}-models.npz');r=json.loads((d/f'{arm}-receipt.json').read_text());assert r['index']==index and r['seed']==seed and r['arm']==arm and r['protocol_sha256']==sha(PROTOCOL) and len(r['records'])==40
        assert r['models_sha256']==sha(d/f'{arm}-models.npz') and r['predraw_receipt_sha256']==sha(d/'predraw-receipt.json');joint=(given[:,ii]&given[:,jj]).sum(0);assert np.array_equal(joint,a['joint_count']) and joint.sum()==446400
        if not public:
            assert r['observations_sha256']==sha(d/f'{arm}-observations.npz') and r['noise_sha256']==sha(d/'potential_noise.npz');observed=npz(d/f'{arm}-observations.npz');expected[arm]=reference_observe(given,clean,noise)
            for key,value in expected[arm].items():assert np.array_equal(value,observed[key])
            assert np.array_equal(expected[arm]['equal_count'],a['equal_count'])
            for record in r['records']:assert record['log_sha256']==sha(d/f'{arm}-d{record["degree"]:02d}.log')
        cost=independent_rate_costs(joint,a['equal_count']);assert np.array_equal(cost,a['cost']);row=grid(a,r['records'],cost,p['solver'],index);selected=row['selected_degree']
        outcome=statistics(np.empty(0,np.int64) if selected is None else a[f'd{selected:02d}_primal_twice'],truth)
        if selected is not None and outcome['retained'] is not None:assert outcome['retained']==81*selected//2
        fixed=statistics(a['d20_primal_twice'],truth) if r['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,a['top810_support'])
        rows.append(dict(index=index,seed=seed,arm=arm,**row,**outcome,fixed_degree20=fixed,top810=statistics(2*top.astype(np.int64),truth)))
    gates=[]
    if not public:
        shared=masks['uniform']&masks['donor'];assert np.array_equal(expected['uniform']['values'][shared],expected['donor']['values'][shared]);gates=complete_gate(feedback,p['tie_seed_base']+seed,p['rank_seed_base']+seed,clean,noise,expected)
    files=['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]
    if not public:files+=['potential_noise.npz']+[arm+'-observations.npz' for arm in ARMS]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,query_accounting=literal_account(masks['uniform'],masks['donor']),complete_label_gates=gates,artifact_sha256={name:sha(d/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(64));groups=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},selected_degree20=sum(r['selected_degree']==20 for r in part),
            selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part),fixed_degree20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part)))
    u,d=groups;bounds={'uniform_procedure':(u['success'],u['success']+u['unresolved'],31),'donor_benefit':(d['success']-u['success']-u['unresolved'],d['success']+d['unresolved']-u['success'],2)}
    endpoints={name:dict(decision='supported' if low>=required else 'falsified' if high<required else 'inconclusive',lower=low,upper=high,required=required) for name,(low,high,required) in bounds.items()}
    paired=dict(donor_wins=[],uniform_wins=[],both_success=[],both_resolved_failure=[],any_unresolved=[])
    for seed in range(32):
        u,d=[rows[2*seed+i]['classification'] for i in range(2)]
        key='any_unresolved' if 'unresolved' in [u,d] else 'both_success' if u==d=='success' else 'donor_wins' if d=='success' else 'uniform_wins' if u=='success' else 'both_resolved_failure';paired[key].append(seed)
    return groups,endpoints,paired


def public_job(seed):return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL);identity,_,_=identity_audit(True)
    for key,value in identity.items():assert saved['source_audit'][key]==value
    for path,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/path)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(32)))
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);assert rows==saved['rows'] and grouped==saved['grouped'] and endpoints==saved['endpoints'] and paired==saved['paired_changes']
    q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};q.update(historical_donor_warmup=240000,new_donor_queries=0,h1az_through_present_sequence_union=5302606+q['new_union_queries']);assert q==saved['query_accounting'] and q['new_union_queries']<=1843200
    assert saved['certified_LPs']==sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected']==sum(r['mutations_rejected'] for r in rows)
    print(json.dumps(dict(status='passed',endpoints=endpoints,grouped=grouped,paired_changes=paired,source_identity=identity,query_accounting=q,certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
        scope='Fresh-ID draws/exclusions, exported fixed donorfeedback, all64 masks/unions, costs/proofs and separate finite endpoints. OriginalCSV/recipient/donor labels and full-label gates require raw audit.')))


if __name__=='__main__':public_audit()
