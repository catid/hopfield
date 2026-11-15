"""Independent H1ca source, fresh records, three-arm score/proof and endpoint audit."""
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, time
import numpy as np
from posterior_fresh_confirm_study import ROOT, HERE, OUT, PUBLIC, REGISTRY, PROTOCOL, PARENT, ARMS, ARM_ROWS, sha, save, npz, verify
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs
from audit_balanced_mask_math import reference_lines
from audit_higher_noise import reference_corrupt


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2414270 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2177845 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-repeat-query-tradeoff.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1417724 and draws.shape==(64,270) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all64_full_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==64
    assert [e['seed'] for e in manifest['inputs']]==list(range(64))
    for seed in range(64):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,270,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==270
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2177845+len(unique),updated_combined_exclusion_union=2414270+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=1399680),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(270,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
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
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all64_full_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def reference_masks(seed):
    rng = np.random.default_rng(seed)
    lines = reference_lines()
    mask = np.ones((270, 81), bool)
    for cycle in range(3):
        permutation = list(map(int, rng.permutation(81)))
        order = list(map(int, rng.permutation(90)))
        for offset, line in enumerate(order):
            for cell in lines[line]:
                mask[90*cycle+offset, permutation[cell]] = False
    return mask


def literal_account(mask):
    single = {(row,int(cell),0) for row in range(270) for cell in np.flatnonzero(mask[row])}
    hard = {(row,int(cell),query) for row in range(90) for cell in np.flatnonzero(mask[row]) for query in range(3)}
    posterior = set(hard)
    assert len(single) == len(hard) == len(posterior) == 19440
    assert len(single & posterior) == 6480 and len(posterior-single) == 12960 and hard == posterior
    return dict(full_physical_noisy_union=len(single|hard|posterior),single270_deployment=len(single),
                hard_triple90_deployment=len(hard),posterior_triple90_deployment=len(posterior),
                shared_first_query90=len(single&posterior),extra_repeat_queries=len(posterior-single),
                shared_two_triple_methods=len(hard&posterior))



def reference_decode(first, second, third):
    answer = np.empty(first.shape, np.int8)
    for index in np.ndindex(first.shape):
        labels = [int(first[index]), int(second[index]), int(third[index])]
        frequency = {digit: labels.count(digit) for digit in set(labels)}
        most = max(frequency.values())
        answer[index] = labels[0] if most == 1 else next(digit for digit, count in frequency.items() if count == most)
    return answer


def reference_observations(mask, clean, noise):
    queries = {}
    for name, n in [('first', 270), ('repeat1', 90), ('repeat2', 90)]:
        queries[name] = reference_corrupt(clean[:n], {key: noise[name+'_'+key] for key in ['iid_uniforms', 'iid_offsets']})
    decoded = reference_decode(queries['first'][:90], queries['repeat1'], queries['repeat2'])
    result = {}
    for arm, labels in [('single270', queries['first']), ('triple90', decoded)]:
        n = len(labels)
        given = mask[:n]
        values = np.full(labels.shape, -1, np.int8)
        joint = np.zeros((81, 81), np.int32)
        equal = np.zeros_like(joint)
        for row in range(n):
            cells = np.flatnonzero(given[row])
            values[row, cells] = labels[row, cells]
            block = np.ix_(cells, cells)
            joint[block] += 1
            retained = labels[row, cells]
            equal[block] += retained[:, None] == retained[None, :]
        ii, jj = np.triu_indices(81, 1)
        result[arm] = dict(given=given, values=values, joint_count=joint[ii, jj], equal_count=equal[ii, jj])
    return result, queries, decoded


def legacy_complete_gate(mask, clean, noise, expected):
    from repeat_query_math import observations
    changed = ((clean+(np.arange(81) % 8+1)[None, :]) % 9).astype(np.int8)
    assert (changed != clean).all()
    altered = observations(mask, changed, noise)
    gates = []
    for arm in ['single270','triple90']:
        actual, old = altered[arm], expected[arm]
        given = old['given']
        n = len(given)
        assert np.array_equal(actual['given'], given) and np.array_equal(actual['joint_count'], old['joint_count'])
        assert (actual['values'][given] != old['values'][given]).all() and not np.array_equal(actual['equal_count'], old['equal_count'])
        gates.append(dict(arm=arm, rows=n, status='passed', all_source_and_queried_decoded_values_changed=True,
                          masks_and_joint_counts_unchanged=True, equality_counts_changed=True,
                          original_source_sha256=hashlib.sha256(clean[:n].tobytes()).hexdigest(),
                          changed_source_sha256=hashlib.sha256(changed[:n].tobytes()).hexdigest()))
    return gates


def noise_diagnostics(mask, clean, queries, decoded):
    result = {}
    for name, labels in list(queries.items())+[('decoded', decoded)]:
        given = mask[:len(labels)]
        truth = clean[:len(labels)]
        bad = labels != truth
        result[name] = dict(full_cells=labels.size, queried_cells=int(given.sum()),
                            full_cell_errors=int(bad.sum()), queried_cell_errors=int(bad[given].sum()),
                            queried_error_offsets=[int(((labels-truth) % 9 == offset)[given].sum()) for offset in range(1, 9)])
    first, second, third = queries['first'][:90], queries['repeat1'], queries['repeat2']
    ties = (first != second) & (first != third) & (second != third)
    result['ties'] = dict(all_cells=int(ties.sum()), queried_cells=int(ties[mask[:90]].sum()))
    return result



def read(path):
    return json.loads(path.read_text())


def audit_queries(seed,p):
    from posterior_fresh_confirm_study import reconstruct,arm_values
    directory=OUT/f's{seed}'
    entry=read(OUT/'input_manifest.json')['inputs'][seed]
    assert sha(ROOT/entry['completion_path'])==entry['completion_sha256']
    clean=npz(ROOT/entry['completion_path'])['solutions']
    mask=npz(directory/'masks.npz')['balanced'];noise=npz(directory/'potential_noise.npz')
    assert np.array_equal(mask,reference_masks(p['balanced_seed_base']+seed)) and len(noise)==6
    for name,key,n in [('first','noise_seed_base',270),('repeat1','repeat1_seed_base',90),('repeat2','repeat2_seed_base',90)]:
        rng=np.random.default_rng(p[key]+seed)
        assert np.array_equal(rng.random((n,81)),noise[name+'_iid_uniforms'])
        assert np.array_equal(rng.integers(1,9,(n,81),dtype=np.int8),noise[name+'_iid_offsets'])
    expected,queries,decoded=reference_observations(mask,clean,noise)
    records=npz(directory/'query-records.npz')
    assert set(records)=={'given','first','repeat1','repeat2'} and np.array_equal(records['given'],mask)
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        assert records[name].shape==(n,81) and records[name].dtype==np.int8
        assert np.array_equal(records[name][mask[:n]],queries[name][mask[:n]]) and (records[name][~mask[:n]]==-1).all()
    for arm,old in [('single270','single270'),('hard_triple90','triple90')]:
        values,given=arm_values(records,arm)
        assert np.array_equal(given,expected[old]['given']) and np.array_equal(values,expected[old]['values'])
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8)
    assert (clean!=changed).all()
    altered=reconstruct(mask,changed,noise)
    gates=[]
    for arm in ARMS:
        before,given=arm_values(records,arm);after,given2=arm_values(altered,arm)
        assert np.array_equal(given,given2)
        if before.ndim==3:
            assert (before[:,given]!=after[:,given]).all()
        else:
            assert (before[given]!=after[given]).all()
        gates.append(dict(arm=arm,status='passed',all_queried_values_changed=True,masks_unchanged=True))
    first,second,third=[records[name][:90] for name in ['first','repeat1','repeat2']]
    distinct=(first!=second)&(first!=third)&(second!=third)
    agree=(first==second)&(first==third)
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
                query_records_sha256=sha(directory/'query-records.npz'),completion_sha256=sha(ROOT/entry['completion_path']),
                masks_sha256=sha(directory/'masks.npz'),noise_sha256=sha(directory/'potential_noise.npz'),
                fresh_raw_correspondence_verified=True,new_queries=32400,scorer_receives_clean_values=False,
                complete_label_gates=gates,realized_noise=noise_diagnostics(mask,clean,queries,decoded),
                triple_query_patterns=dict(three_agree=int(agree[mask[:90]].sum()),all_distinct=int(distinct[mask[:90]].sum()),
                                           two_agree=int((~(agree|distinct))[mask[:90]].sum())))


def score_reference(records,arm):
    from posterior_query_math import independent_score
    if arm=='posterior_triple90':
        values=np.stack([records['first'][:90],records['repeat1'],records['repeat2']])
        cost,joint,totals=independent_score(values,records['given'][:90])
        return cost,joint,totals
    labels=records['first'] if arm=='single270' else reference_decode(records['first'][:90],records['repeat1'],records['repeat2'])
    given=records['given'][:len(labels)]
    ii,jj=np.triu_indices(81,1);joint=np.zeros(3240,np.int64);equal=np.zeros(3240,np.int64)
    for row in range(len(labels)):
        selected=given[row,ii]&given[row,jj]
        joint+=selected;equal+=selected&(labels[row,ii]==labels[row,jj])
    return independent_rate_costs(joint,equal),joint,equal


def audit_bank(seed,public=False,protocol=None):
    from fractions import Fraction
    from itertools import combinations_with_replacement
    from posterior_query_math import NORMALIZERS
    p=verify() if protocol is None else protocol
    base=PUBLIC if public else OUT;directory=base/f's{seed}'
    seal=read(base/'query-record-seal.json')
    assert seal['protocol_sha256']==sha(PROTOCOL) and seal['all64_query_records_validated_before_scoring']
    assert seal['record_sha256'][str(seed)]==sha(directory/'query-records.npz')
    assert seal['audit_sha256'][str(seed)]==sha(directory/'query-record-audit.json')
    assert seal['source_audit_sha256']==sha(base/'source-audit.json') and seal['physical_noisy_queries']==2073600
    mask=npz(directory/'masks.npz')['balanced']
    assert np.array_equal(mask,reference_masks(p['balanced_seed_base']+seed)) and mask.shape==(270,81) and mask.dtype==bool and (mask.sum(1)==72).all()
    predraw=read(directory/'predraw-receipt.json')
    assert predraw['masks_sha256']==sha(directory/'masks.npz') and predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL)
    query_audit=read(directory/'query-record-audit.json')
    assert query_audit['status']=='validated' and query_audit['seed']==seed and query_audit['protocol_sha256']==sha(PROTOCOL)
    assert query_audit['query_records_sha256']==sha(directory/'query-records.npz') and query_audit['masks_sha256']==sha(directory/'masks.npz')
    assert query_audit['fresh_raw_correspondence_verified'] and query_audit['new_queries']==32400 and not query_audit['scorer_receives_clean_values']
    if not public:
        assert query_audit['noise_sha256']==sha(directory/'potential_noise.npz')
        entry=read(OUT/'input_manifest.json')['inputs'][seed]
        assert entry['completion_sha256']==query_audit['completion_sha256']==sha(ROOT/entry['completion_path'])
    records_input=npz(directory/'query-records.npz')
    assert set(records_input)=={'given','first','repeat1','repeat2'} and np.array_equal(records_input['given'],mask)
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        values=records_input[name]
        assert values.shape==(n,81) and values.dtype==np.int8 and ((values[mask[:n]]>=0)&(values[mask[:n]]<=8)).all() and (values[~mask[:n]]==-1).all()
    ii,jj=np.triu_indices(81,1)
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    rows=[]
    for ai,arm in enumerate(ARMS):
        index=3*seed+ai;n=ARM_ROWS[arm]
        arrays=npz(directory/f'{arm}-models.npz');receipt=read(directory/f'{arm}-receipt.json')
        assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['rows']==n
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['models_sha256']==sha(directory/f'{arm}-models.npz') and len(receipt['records'])==40
        assert receipt['query_records_sha256']==sha(directory/'query-records.npz') and receipt['query_seal_sha256']==sha(base/'query-record-seal.json')
        assert receipt['predraw_receipt_sha256']==sha(directory/'predraw-receipt.json')
        cost,joint,extra=score_reference(records_input,arm)
        assert (joint==71*(n//90)).all() and (mask[:n].sum(0)==80*(n//90)).all()
        assert np.array_equal(cost,arrays['cost']) and np.array_equal(joint,arrays['joint_count'])
        if arm=='posterior_triple90':
            den=arrays['posterior_denominators'];num=arrays['posterior_numerators']
            assert den.dtype==num.dtype==np.int64 and den.tolist()==sorted({a*b for a,b in combinations_with_replacement(NORMALIZERS[3],2)})
            assert num.shape==(len(den),3240) and (num>=0).all()
            for edge in range(3240):
                actual=sum((Fraction(int(num[k,edge]),int(den[k])) for k in range(len(den))),Fraction(0))
                assert actual==extra[edge]
        else:
            assert np.array_equal(extra,arrays['equal_count'])
        if not public:
            for record in receipt['records']:
                assert record['log_sha256']==sha(directory/f'{arm}-d{record["degree"]:02d}.log')
        checked=grid(arrays,receipt['records'],cost,p['solver'],index);degree=checked['selected_degree']
        primal=np.empty(0,np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice']
        fixed=statistics(arrays['d20_primal_twice'],truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
        top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
        assert np.array_equal(top,arrays['top810_support'])
        rows.append(dict(index=index,seed=seed,arm=arm,rows=n,**checked,**statistics(primal,truth),
                         pair_coqueries=71*(n//90),queries_per_cell=240,distinct_rows_per_cell=80*(n//90),
                         fixed_degree20=fixed,top810=statistics(2*top.astype(np.int64),truth)))
    files=['masks.npz','predraw-receipt.json','query-records.npz','query-record-audit.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz','receipt.json']]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,
                query_accounting=literal_account(mask),complete_label_gates=query_audit['complete_label_gates'],
                realized_noise=query_audit['realized_noise'],triple_query_patterns=query_audit['triple_query_patterns'],
                artifact_sha256={name:sha(directory/name) for name in files})


def paired(rows,control):
    categories=dict(posterior_gains=[],posterior_losses=[],both_exact=[],both_resolved_failure=[],any_unresolved=[])
    low=high=0
    failures=['resolved_failure_nonexact','resolved_failure_fractional']
    for seed in range(64):
        left=rows[3*seed+control]['classification'];right=rows[3*seed+2]['classification']
        assert left in failures+['success','unresolved'] and right in failures+['success','unresolved']
        low+=left=='success' and right in failures
        high+=left in ['success','unresolved'] and right!='success'
        key='any_unresolved' if 'unresolved' in [left,right] else 'both_exact' if left==right=='success' else 'posterior_gains' if right=='success' else 'posterior_losses' if left=='success' else 'both_resolved_failure'
        categories[key].append(seed)
    return dict(**categories,loss_lower=low,loss_upper=high)


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(192))
    groups=[]
    for ai,arm in enumerate(ARMS):
        part=[r for r in rows if r['arm']==arm]
        assert len(part)==64 and [r['seed'] for r in part]==list(range(64)) and all(r['index']==3*r['seed']+ai for r in part)
        groups.append(dict(arm=arm,rows=ARM_ROWS[arm],**{kind:sum(r['classification']==kind for r in part) for kind in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
                           selected_degree20=sum(r['selected_degree']==20 for r in part),fixed_degree20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),
                           top810_exact=sum(r['top810']['classification']=='success' for r in part)))
    posterior=groups[2];lo=posterior['success'];hi=lo+posterior['unresolved']
    pairs={name:paired(rows,control) for name,control in [('versus_single270',0),('versus_hard_triple90',1)]}
    loss=pairs['versus_single270'];low,high=loss['loss_lower'],loss['loss_upper']
    return dict(grouped=groups,paired_changes=pairs,endpoints=dict(
        posterior_exact=dict(lower=lo,upper=hi,required=61,total=64,decision='supported' if lo>=61 else 'falsified' if hi<61 else 'inconclusive'),
        paired_loss=dict(lower=low,upper=high,maximum=3,total=64,decision='supported' if high<=3 else 'falsified' if low>3 else 'inconclusive')))


def accounting_total(audits,identity):
    names=['full_physical_noisy_union','single270_deployment','hard_triple90_deployment','posterior_triple90_deployment','shared_first_query90','extra_repeat_queries','shared_two_triple_methods']
    q={key:sum(a['query_accounting'][key] for a in audits) for key in names}
    assert q==dict(full_physical_noisy_union=2073600,single270_deployment=1244160,hard_triple90_deployment=1244160,posterior_triple90_deployment=1244160,
                   shared_first_query90=414720,extra_repeat_queries=829440,shared_two_triple_methods=1244160)
    q.update(clean_unique_oracle_cells=identity['clean_unique_oracle_cells'],clean_materialized_cell_instances=1399680,source_instances=17280,
             official_noisy_label_sequence_before=14687319,official_noisy_label_sequence_after=16760919)
    return q


def public_job(seed):
    return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    identity,_,_=identity_audit(True)
    for key,value in identity.items():assert saved['source_audit'][key]==value
    for rel,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(64)))
    for seed,audit in enumerate(audits):assert audit==read(PUBLIC/f's{seed}/audit.json')
    rows=[row for audit in audits for row in audit['rows']]
    assert rows==saved['rows'];outcomes=aggregate(rows)
    for key,value in outcomes.items():assert saved[key]==value
    assert accounting_total(audits,identity)==saved['query_accounting']
    assert saved['certified_LPs']==sum(row['certified_grid_LPs'] for row in rows) and saved['mutations_rejected']==sum(row['mutations_rejected'] for row in rows)
    assert saved['analytic_zero_proofs']==saved['complete_label_gates']==192 and saved['all192_arms_retained']
    print(json.dumps(dict(status='passed',**outcomes,query_accounting=saved['query_accounting'],source_identity=identity,
                         certified_LPs=saved['certified_LPs'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,
                         scope='All fresh IDs/masks/query records, three-arm scores/proofs/endpoints replayed. CSV and latent-noise correspondence relies on charged raw preparation audit.')))


if __name__=='__main__':
    public_audit()
