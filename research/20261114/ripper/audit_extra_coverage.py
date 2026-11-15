"""Independent mask extensions, literal set accounting and exact-grid audit."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib,json,time
import numpy as np
from extra_coverage_study import ROOT,HERE,OUT,PUBLIC,PARENT,OLD_PUBLIC,BASELINE,PROTOCOL,ARMS,sha,save,npz,verify
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs


def reference_masks(old,seed):
    rng=np.random.default_rng(seed);result=np.zeros_like(old)
    for row in range(len(old)):
        retained=set(map(int,np.flatnonzero(old[row])));order=list(map(int,rng.permutation(81)));extra=[v for v in order if v not in retained][:7]
        result[row,list(retained|set(extra))]=True
    return result


def literal_account(olda,oldp,newa,newp):
    keys=['old_aligned','old_permuted','old_shared','old_union','aligned_total','permuted_total','new_shared_total','new_union_total','aligned_added_to_own','permuted_added_to_own','aligned_new_to_prior_union','permuted_new_to_prior_union','newly_shared','actual_increment'];counts=dict.fromkeys(keys,0)
    for row in range(len(olda)):
        oa,op,na,nn=[set(map(int,np.flatnonzero(m[row]))) for m in [olda,oldp,newa,newp]];old=oa|op
        values=[len(oa),len(op),len(oa&op),len(old),len(na),len(nn),len(na&nn),len(na|nn),len(na-oa),len(nn-op),len(na-old),len(nn-old),len((na&nn)-old),len((na|nn)-old)]
        for key,value in zip(keys,values,strict=True):counts[key]+=value
    assert counts['new_union_total']==counts['old_union']+counts['actual_increment']
    assert counts['actual_increment']==counts['aligned_new_to_prior_union']+counts['permuted_new_to_prior_union']-counts['newly_shared']
    return counts


def reference_observe(given,clean,noise):
    values=np.full(given.shape,-1,np.int8);ii,jj=np.triu_indices(81,1)
    for row in range(len(clean)):
        board=clean[row,noise['permutations'][row]] if Fraction(float(noise['uniforms'][row]))<Fraction(9,160) else clean[row]
        values[row,given[row]]=board[given[row]]
    joint=given[:,ii]&given[:,jj]
    return dict(given=given,values=values,joint_count=joint.sum(0),equal_count=(joint&(values[:,ii]==values[:,jj])).sum(0))


def complete_gate(old,seed,clean,noise,expected):
    from extra_coverage_study import acquire
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=clean).all()
    actual=acquire(old,seed,changed,noise);given=expected['given']
    assert np.array_equal(actual['given'],given) and np.array_equal(actual['joint_count'],expected['joint_count'])
    assert (actual['values'][given]!=expected['values'][given]).all() and not np.array_equal(actual['equal_count'],expected['equal_count'])
    return dict(status='passed',every_source_and_queried_value_changed=True,all_masks_and_joint_counts_unchanged=True,equality_counts_changed=True,original_source_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),changed_source_sha256=hashlib.sha256(changed.tobytes()).hexdigest())


def statistics(t,truth):
    if t.shape!=truth.shape:return dict(classification='unresolved',missing_true=None,extra=None,retained=None,support_sha256=None)
    if not np.isin(t,[0,2]).all():return dict(classification='resolved_failure_fractional',missing_true=None,extra=None,retained=None,support_sha256=None)
    support=t==2;missing=int((truth&~support).sum());extra=int((~truth&support).sum())
    return dict(classification='success' if missing==extra==0 else 'resolved_failure_nonexact',missing_true=missing,extra=extra,retained=int(support.sum()),support_sha256=hashlib.sha256(support.tobytes()).hexdigest())


def audit_bank(seed,public=False):
    p=verify(not public);d=(PUBLIC if public else OUT)/f's{seed}';sealed=json.loads(((PUBLIC if public else OUT)/'mask-seal.json').read_text());predraw=json.loads((d/'predraw-receipt.json').read_text());masks=npz(d/'masks.npz')
    assert sealed['status']=='all64_streams_sealed_before_additional_labels' and sealed['protocol_sha256']==sha(PROTOCOL) and len(sealed['predraw_receipts_sha256'])==32
    assert sealed['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json') and predraw['masks_sha256']==sha(d/'masks.npz') and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['seed']==seed
    assert predraw['status']=='sealed_before_additional_labels'
    with np.load(OLD_PUBLIC,allow_pickle=False) as z:old={arm:z[f's{seed}_{arm}_mask'].copy() for arm in ARMS}
    baseline=json.loads(BASELINE.read_text());oldrows={r['arm']:r for r in baseline['rows'] if r['seed']==seed};assert len(oldrows)==2
    clean=noise=None
    if not public:
        entry=json.loads((PARENT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed;clean=npz(ROOT/entry['completion_path'])['solutions'];noise=npz(PARENT/f's{seed}'/'potential_noise.npz')
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[];gates=[];values={};files={}
    for arm_index,arm in enumerate(ARMS):
        index=2*seed+arm_index;assert np.array_equal(old[arm],masks[arm+'_old']);given=reference_masks(old[arm],p['extension_seed_base']+seed)
        assert np.array_equal(given,masks[arm+'_new']) and not (old[arm]&~given).any() and (given.sum(1)==32).all() and ((given&~old[arm]).sum(1)==7).all()
        a=npz(d/f'{arm}-models.npz');r=json.loads((d/f'{arm}-receipt.json').read_text());assert r['index']==index and r['seed']==seed and r['arm']==arm and r['protocol_sha256']==sha(PROTOCOL)
        assert r['models_sha256']==sha(d/f'{arm}-models.npz') and r['mask_seal_sha256']==sha((PUBLIC if public else OUT)/'mask-seal.json') and r['predraw_receipt_sha256']==sha(d/'predraw-receipt.json') and len(r['records'])==40
        joint=(given[:,ii]&given[:,jj]).sum(0);assert np.array_equal(joint,a['joint_count']) and int(joint.sum())==900*496
        if not public:
            observed=npz(d/f'{arm}-observations.npz');assert r['observations_sha256']==sha(d/f'{arm}-observations.npz');expected=reference_observe(given,clean,noise)
            for key,v in expected.items():assert np.array_equal(v,observed[key])
            for key in ['joint_count','equal_count']:assert np.array_equal(expected[key],a[key])
            previous=npz(PARENT/f's{seed}'/f'{arm}-observations.npz');assert np.array_equal(previous['given'],old[arm]) and np.array_equal(previous['values'][old[arm]],expected['values'][old[arm]])
            gates.append(dict(arm=arm,**complete_gate(old[arm],p['extension_seed_base']+seed,clean,noise,expected)));values[arm]=expected['values']
            for record in r['records']:assert record['log_sha256']==sha(d/f'{arm}-d{record["degree"]:02d}.log')
        cost=independent_rate_costs(joint,a['equal_count']);assert np.array_equal(cost,a['cost']);row=grid(a,r['records'],cost,p['solver'],index);selected=row['selected_degree']
        outcome=statistics(np.empty(0,np.int64) if selected is None else a[f'd{selected:02d}_primal_twice'],truth)
        if selected is not None and outcome['retained'] is not None:assert outcome['retained']==81*selected//2
        fixed=statistics(a['d20_primal_twice'],truth) if r['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,a['top810_support'])
        topstats=statistics(2*top.astype(np.int64),truth)
        rows.append(dict(index=index,seed=seed,arm=arm,**row,**outcome,fixed_degree20=fixed,top810=topstats,archived_classification=oldrows[arm]['classification'],archived_selected_degree=oldrows[arm]['selected_degree']))
    queries=literal_account(*[masks[k] for k in ['aligned_old','permuted_old','aligned_new','permuted_new']])
    if not public:
        assert queries==json.loads((d/'accounting.json').read_text());shared=masks['aligned_new']&masks['permuted_new'];assert np.array_equal(values['aligned'][shared],values['permuted'][shared])
    names=['masks.npz','predraw-receipt.json']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]
    if not public:names+=['accounting.json']+[arm+'-observations.npz' for arm in ARMS]
    files={name:sha(d/name) for name in names}
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,query_accounting=queries,complete_label_gates=gates,artifact_sha256=files)


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(64));groups=[];paired={}
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        groups.append(dict(arm=arm,**{k:sum(r['classification']==k for r in part) for k in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
            selected_degree20=sum(r['selected_degree']==20 for r in part),selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},
            missing_integral_only=sum(r['missing_true'] or 0 for r in part),extra_integral_only=sum(r['extra'] or 0 for r in part),fixed_degree20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part)))
        paired[arm]=dict(gains=[r['seed'] for r in part if r['archived_classification']!='success' and r['classification']=='success'],losses_to_resolved_failure=[r['seed'] for r in part if r['archived_classification']=='success' and r['classification'] in ['resolved_failure_nonexact','resolved_failure_fractional']],
            prior_success_now_unresolved=[r['seed'] for r in part if r['archived_classification']=='success' and r['classification']=='unresolved'],retained_success=[r['seed'] for r in part if r['archived_classification']==r['classification']=='success'],remaining_prior_failure=[r['seed'] for r in part if r['archived_classification']!='success' and r['classification']!='success'])
    a=groups[0];endpoints={}
    for name,key in [('degree20','selected_degree20'),('exact_graph','success')]:
        lower=a[key];upper=lower+a['unresolved'];endpoints[name]=dict(decision='supported' if lower>=31 else 'falsified' if upper<31 else 'inconclusive',lower=lower,upper=upper,required=31)
    return groups,endpoints,paired


def public_job(seed):return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL) and saved['mask_seal_sha256']==sha(PUBLIC/'mask-seal.json')
    for path,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/path)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(32)))
    rows=[r for a in audits for r in a['rows']];grouped,endpoints,paired=aggregate(rows);assert rows==saved['rows'] and grouped==saved['grouped'] and endpoints==saved['endpoints'] and paired==saved['paired_changes']
    q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};q['h1az_through_present_sequence_union']=5148181+q['actual_increment'];q['new_sources']=q['new_donor_queries']=0;assert q==saved['query_accounting'] and q['actual_increment']<=403200
    for key in ['certified_LPs','mutations_rejected']:assert saved[key]==sum(r['certified_grid_LPs' if key=='certified_LPs' else key] for r in rows)
    print(json.dumps(dict(status='passed',endpoints=endpoints,grouped=grouped,paired_changes=paired,query_accounting=q,certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
        scope='Checks all masks, literal old/new query unions, rates from exported equalities, all proofs and paired outcomes. Original labels/equalities and full-label substitution gates require archived raw audit.')))


if __name__=='__main__':public_audit()
