"""H1az independent source ledger, selected CSV answers, costs and LP proof audit."""
import csv,hashlib,json
import numpy as np
from audit_degree_prior import check_certificate
from audit_laplace_rate import independent_rate_costs


def sources():
    from fresh_rate_study import ROOT,HERE,OUT,PROTOCOL,SOURCE,sha,save,verify,load_npz
    p=verify();assert sha(SOURCE)==p['source_sha256']
    assert sha(ROOT/p['ledger_path'])==p['ledger_sha256'] and sha(ROOT/p['excluded_ids_path'])==p['excluded_ids_sha256']
    ledger=json.loads((ROOT/p['ledger_path']).read_text());parts=[];evidence={}
    for spec in ledger['manifests']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];evidence[spec['path']]=spec['sha256']
        for record in json.loads(path.read_text())['inputs']:
            path=ROOT/record['completion_path'];assert sha(path)==record['completion_sha256']==ledger['completion_sha256'][record['completion_path']]
            with np.load(path,allow_pickle=False) as z:parts.append(z['source_row'].copy())
            evidence[record['completion_path']]=record['completion_sha256']
    excluded=np.unique(np.concatenate(parts));assert len(excluded)==p['excluded_unique']==1928477
    assert np.array_equal(excluded,load_npz(ROOT/p['excluded_ids_path'])['excluded'])
    assert hashlib.sha256(excluded.tobytes()).hexdigest()==ledger['excluded_id_bytes_sha256']
    eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True);assert len(eligible)==p['eligible_rows']
    pool=np.random.default_rng(p['streams']['pool_split']).permutation(eligible);half=len(pool)//2
    seal=json.loads((OUT/'identity-seal.json').read_text());manifest=json.loads((OUT/'input_manifest.json').read_text());selected=load_npz(OUT/'selected_sources.npz')
    assert seal['status']=='identities_sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL)
    assert seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(OUT/'selected_sources.npz')
    assert seal['ledger_sha256']==p['ledger_sha256'] and manifest['identity_seal_sha256']==sha(OUT/'identity-seal.json')
    assert manifest['source_sha256']==p['source_sha256'] and manifest['all_source_manifest_sha256']==sha(OUT/'all_source_manifest.json')
    assert len(manifest['inputs'])==32 and [e['seed'] for e in manifest['inputs']]==list(range(32))
    wanted={};roles={'calibration':set(),'training':set()}
    for entry in manifest['inputs']:
        seed=entry['seed']
        for role,n,source_pool in [('calibration',1000,pool[:half]),('training',900,pool[half:])]:
            expected=np.random.default_rng(p['streams'][role+'_source']+seed).choice(source_pool,n,replace=False)
            path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256'];evidence[entry[role+'_path']]=entry[role+'_sha256'];z=load_npz(path)
            assert set(z)=={'solutions','source_row'} and z['source_row'].dtype==np.int64 and z['solutions'].dtype==np.int8
            ids=z['source_row'];boards=z['solutions'];assert np.array_equal(ids,expected) and np.array_equal(ids,selected[role+'_source_row'][seed])
            assert ids.shape==(n,) and len(set(map(int,ids)))==n and boards.shape==(n,81) and ((boards>=0)&(boards<9)).all()
            assert np.intersect1d(ids,excluded).size==0
            for index,board in zip(ids,boards,strict=True):
                index=int(index);roles[role].add(index)
                if index in wanted:assert np.array_equal(wanted[index],board)
                else:wanted[index]=board
                grid=board.reshape(9,9)
                assert all(len(set(map(int,grid[i,:])))==len(set(map(int,grid[:,i])))==9 for i in range(9))
                assert all(len(set(map(int,grid[r:r+3,c:c+3].ravel())))==9 for r in [0,3,6] for c in [0,3,6])
    assert not roles['calibration']&roles['training'];assert np.array_equal(selected['all_source_row'],sorted(wanted))
    assert len(roles['calibration'])==seal['calibration_unique'] and len(roles['training'])==seal['training_unique']
    remaining=set(wanted)
    with SOURCE.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for row,fields in enumerate(reader):
            if row not in remaining:continue
            board=np.array([int(c)-1 for c in fields[column]],np.int8);assert np.array_equal(board,wanted[row]);remaining.remove(row)
            if not remaining:break
    assert not remaining
    assert len({board.tobytes() for board in wanted.values()})==len(wanted)
    allmanifest=json.loads((OUT/'all_source_manifest.json').read_text());assert len(allmanifest['inputs'])==64
    for entry in allmanifest['inputs']:assert sha(ROOT/entry['completion_path'])==entry['completion_sha256']
    report=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),
                identity_seal_sha256=sha(OUT/'identity-seal.json'),selected_sources_sha256=sha(OUT/'selected_sources.npz'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                ledger_sha256=p['ledger_sha256'],prior_excluded_unique=len(excluded),eligible_rows=len(eligible),calibration_unique=len(roles['calibration']),training_unique=len(roles['training']),
                new_unique_source_ids=len(wanted),updated_ripper_ledger_union=len(excluded)+len(wanted),within_bank_repeats=0,calibration_training_overlap=0,declared_ledger_overlap=0,
                calibration_crossbank_repeats=32000-len(roles['calibration']),training_crossbank_repeats=28800-len(roles['training']),
                independent_csv_answers_verified=len(wanted),source_sha256=p['source_sha256'],source_rows=p['pool_size'],input_sha256=evidence,
                scope=p['scope'],new_solver_inputs=0)
    save(OUT/'source-audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)


def audit_bank(seed):
    from fresh_rate_study import HERE,OUT,PROTOCOL,sha,save,load_npz
    from fresh_rate_common import load_protocol
    p=load_protocol();d=OUT/f's{seed}';r=json.loads((d/'recovery.json').read_text());z=load_npz(d/'recovery.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['producer_sha256']==sha(HERE/'fresh_rate_study.py')
    assert r['acquisition_audit_sha256']==sha(d/'audit.json') and r['acquisition_receipt_sha256']==sha(d/'receipt.json') and r['recovery_sha256']==sha(d/'recovery.npz')
    audit=json.loads((d/'audit.json').read_text());assert audit['status']=='validated' and audit['auditor_sha256']==sha(HERE/'audit_fresh_rate_acquisition.py')
    assert audit['protocol_sha256']==sha(PROTOCOL) and audit['receipt_sha256']==sha(d/'receipt.json')
    assert audit['independent_masks_replayed']==900 and audit['training_queries']==22500 and audit['calibration_queries']==2000
    old=load_npz(d/'fit.npz');observed=load_npz(d/'observations.npz');ii,jj=np.triu_indices(81,1)
    m=np.zeros(3240,np.int64);c=np.zeros_like(m)
    for row in range(900):
        given=observed['given'][row];values=observed['values'][row];queried=given[ii]&given[jj]
        m+=queried;c+=queried&(values[ii]==values[jj])
    assert np.array_equal(z['joint_count'],m) and np.array_equal(z['equal_count'],c) and int(m.sum())==270000
    assert np.array_equal(z['original_support'],old['support'])
    for name in ['joint_count','equal_count']:assert np.array_equal(z[name],old[name])
    cost=independent_rate_costs(m,c);assert np.array_equal(cost,z['cost']);top=np.zeros(3240,bool)
    top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,z['top810_support'])
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    rows=[dict(seed=seed,arm=arm,**statistics(support)) for arm,support in [('original',old['support']),('top810',top)]]
    t,y,upper=[z[name] for name in ['primal_twice','node_dual_twice','upper_dual_twice']]
    certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==r['certificate']['certified'];raw=z['primal_values'];candidate=None
    if raw.shape==(3240,) and np.isfinite(raw).all() and np.isin(raw,[0.,1.]).all():
        support=raw==1;degree=np.bincount(np.r_[ii[support],jj[support]],minlength=81)
        if np.all(degree==20):candidate=statistics(support)
    if certified:
        assert r['solver']['solver_optimal'] and r['solver']['value_valid'] and r['solver']['dual_valid']
        assert np.array_equal(raw*2,t) and np.all(np.abs(z['row_dual_values']*2-y)<=1e-6)
        objective=sum(int(w)*int(x) for w,x in zip(cost,t,strict=True))
        assert objective==r['certificate']['primal_objective_twice']==r['certificate']['dual_objective_twice']
        assert abs(r['solver']['solver_objective']-objective/2)<1e-5
        integral=bool(np.isin(t,[0,2]).all());assert integral==r['certificate']['integral']
        if integral:
            assert candidate is not None;category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact'
            rows.append(dict(seed=seed,arm='degree20',certified=True,**candidate))
        else:category='resolved_failure_fractional'
    else:category='unresolved'
    assert category==r['classification'];solver=r['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
    for name,value in p['solver_options'].items():
        if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
        else:assert solver['solver_options'][name]==value
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),recovery_sha256=sha(d/'recovery.npz'),recovery_receipt_sha256=sha(d/'recovery.json'),
                acquisition_audit_sha256=sha(d/'audit.json'),acquisition_audit=audit,rows=rows,classification=category,certificate_verified=certified,integral_candidate_graph=candidate,
                fractional_variables=int((raw==.5).sum()) if raw.shape==(3240,) else None,solver=solver)
    save(d/'recovery-audit.json',result);return result
