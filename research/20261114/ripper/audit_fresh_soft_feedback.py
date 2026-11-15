"""Independent H1bd ledger/CSV/source and paired sequential policy/proof audit."""
import csv,hashlib,json
import numpy as np
from audit_soft_feedback import reference_replay,suffix_gate
from audit_laplace_rate import independent_rate_costs
from audit_degree_prior import check_certificate


def sources():
    from fresh_soft_feedback_study import ROOT,HERE,OUT,PROTOCOL,sha,save,verify,load_npz
    p=verify();source=ROOT/p['source'];assert sha(source)==p['source_sha256']
    ledger=json.loads((ROOT/p['ledger_path']).read_text());parts=[];evidence={}
    assert len(ledger['manifests'])==20
    for spec in ledger['manifests']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];evidence[spec['path']]=spec['sha256']
        for entry in json.loads(path.read_text())['inputs']:
            path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256']==ledger['completion_sha256'][entry['completion_path']]
            with np.load(path,allow_pickle=False) as z:parts.append(z['source_row'].copy())
            evidence[entry['completion_path']]=entry['completion_sha256']
    excluded=np.unique(np.concatenate(parts));assert len(excluded)==p['excluded_unique']==1988296
    assert np.array_equal(excluded,load_npz(ROOT/p['excluded_ids_path'])['excluded']) and hashlib.sha256(excluded.tobytes()).hexdigest()==ledger['excluded_id_bytes_sha256']
    eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True);assert len(eligible)==p['eligible_rows']==1843698
    seal=json.loads((OUT/'identity-seal.json').read_text());manifest=json.loads((OUT/'input_manifest.json').read_text());selected=load_npz(OUT/'selected_sources.npz')
    assert seal['status']=='identities_sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL)
    assert seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(OUT/'selected_sources.npz') and manifest['identity_seal_sha256']==sha(OUT/'identity-seal.json')
    assert seal['ledger_sha256']==p['ledger_sha256'] and manifest['source_sha256']==p['source_sha256']
    assert [e['seed'] for e in manifest['inputs']]==list(range(32)) and selected['source_rows'].shape==(32,900)
    wanted={}
    for entry in manifest['inputs']:
        seed=entry['seed'];assert entry['role']=='training';expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,900,replace=False)
        path=ROOT/entry['completion_path'];assert sha(path)==entry['completion_sha256'];evidence[entry['completion_path']]=entry['completion_sha256'];z=load_npz(path)
        assert set(z)=={'solutions','source_row'} and z['source_row'].dtype==np.int64 and z['solutions'].dtype==np.int8
        ids=z['source_row'];boards=z['solutions'];assert np.array_equal(ids,expected) and np.array_equal(ids,selected['source_rows'][seed])
        assert len(np.unique(ids))==900 and boards.shape==(900,81) and ((boards>=0)&(boards<9)).all() and np.intersect1d(ids,excluded).size==0
        grids=boards.reshape(-1,9,9);boxes=grids.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert ((grids==digit).sum(1)==1).all() and ((grids==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
        for index,board in zip(ids,boards,strict=True):
            index=int(index)
            if index in wanted:assert np.array_equal(wanted[index],board)
            else:wanted[index]=board
    assert np.array_equal(selected['all_source_row'],sorted(wanted)) and len(wanted)==seal['new_unique_source_ids'] and 28800-len(wanted)==seal['crossbank_repeated_instances']
    remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for row,fields in enumerate(reader):
            if row not in remaining:continue
            board=np.array([int(c)-1 for c in fields[column]],np.int8);assert np.array_equal(board,wanted[row]);remaining.remove(row)
            if not remaining:break
    assert not remaining and len({board.tobytes() for board in wanted.values()})==len(wanted)
    allmanifest=json.loads((OUT/'all_source_manifest.json').read_text());assert allmanifest['inputs']==manifest['inputs']
    report=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),
                identity_seal_sha256=sha(OUT/'identity-seal.json'),selected_sources_sha256=sha(OUT/'selected_sources.npz'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                ledger_sha256=p['ledger_sha256'],prior_excluded_unique=len(excluded),eligible_rows=len(eligible),new_unique_source_ids=len(wanted),updated_ripper_ledger_union=len(excluded)+len(wanted),
                within_bank_repeats=0,declared_ledger_overlap=0,crossbank_repeated_instances=28800-len(wanted),source_instances=28800,calibration_source_instances=0,
                independent_csv_answers_verified=len(wanted),source_sha256=p['source_sha256'],source_rows=p['pool_size'],input_sha256=evidence,scope=p['scope'],new_solver_inputs=0)
    save(OUT/'source-audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)

def audit_bank(seed,raw=None):
    from fresh_soft_feedback_study import ROOT,HERE,OUT,PROTOCOL,sha,save,load_npz,verify
    p=verify();d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());z=load_npz(d/'model.npz')
    assert r['status']=='complete' and r['seed']==seed and r['protocol_sha256']==sha(PROTOCOL) and r['producer_sha256']==sha(HERE/'fresh_soft_feedback_study.py')
    for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
    entry=json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed];assert entry['seed']==seed and r['source_sha256']==entry['completion_sha256']==sha(ROOT/entry['completion_path'])
    clean,noise=(load_npz(ROOT/entry['completion_path'])['solutions'],load_npz(d/'potential_noise.npz')) if raw is None else raw
    nrng=np.random.default_rng(p['noise_seed_base']+seed)
    for row in range(900):
        uniform=nrng.random();permutation=nrng.permutation(81).astype(np.uint8)
        assert uniform==noise['uniforms'][row] and np.array_equal(permutation,noise['permutations'][row])
    feedback_perm=np.random.default_rng(p['feedback_permutation_seed_base']+seed).permutation(81)
    assert np.array_equal(feedback_perm,p['feedback_permutations'][seed]) and set(map(int,feedback_perm))==set(range(81))
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    rows=[];arm_audits={};observations={};gates=[]
    def statistics(support):
        missing=int((truth&~support).sum());extra=int((~truth&support).sum())
        return dict(retained=int(support.sum()),missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,support_bytes_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    for arm in ['aligned','permuted']:
        prefix=arm+'_';record=None
        if arm in ['aligned','permuted']:
            observed=load_npz(d/f'{arm}-observations.npz');events=json.loads((d/f'{arm}-events.json').read_text());assert len(events)==900
            perm=feedback_perm if arm=='permuted' else None
            replay=reference_replay(clean,noise,p['tie_seed_base']+seed,perm,p['suffix_checkpoint'],observed,events)
            for name,value in replay.items():assert np.array_equal(value,observed[name])
            assert np.array_equal(observed['given'],z[arm+'_mask'])
            for key in ['joint_count','equal_count']:assert np.array_equal(observed[key],z[prefix+key])
            observations[arm]=observed;gates.append(dict(arm=arm,**suffix_gate(clean,noise,p['tie_seed_base']+seed,perm,p['suffix_checkpoint'],observed)))
            record=json.loads((d/f'{arm}-fit.json').read_text())
        m,c=z[prefix+'joint_count'],z[prefix+'equal_count'];assert int(m.sum())==270000
        cost=independent_rate_costs(m,c);assert np.array_equal(cost,z[prefix+'cost'])
        if arm in ['aligned','permuted']:
            top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
            assert np.array_equal(top,z[prefix+'top810_support']);rows.append(dict(seed=seed,arm=arm+'_top810',**statistics(top)))
        t,y,upper=[z[prefix+k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        certified=check_certificate(81,20,ii,jj,cost,t,y,upper);assert certified==record['certificate']['certified'];rawx=z[prefix+'primal_values'];candidate=None
        if rawx.shape==(3240,) and np.isfinite(rawx).all() and np.isin(rawx,[0.,1.]).all():
            graph=rawx==1;degree=np.bincount(np.r_[ii[graph],jj[graph]],minlength=81)
            if np.all(degree==20):candidate=statistics(graph)
        if certified:
            assert record['solver']['solver_optimal'] and record['solver']['value_valid'] and record['solver']['dual_valid']
            assert np.array_equal(rawx*2,t) and np.all(np.abs(z[prefix+'row_dual_values']*2-y)<=1e-6)
            objective=sum(int(w)*int(x) for w,x in zip(cost,t,strict=True));assert objective==record['certificate']['primal_objective_twice']==record['certificate']['dual_objective_twice']
            assert abs(record['solver']['solver_objective']-objective/2)<1e-5
            integral=bool(np.isin(t,[0,2]).all());assert integral==record['certificate']['integral']
            if integral:
                assert candidate is not None;category='success' if candidate['exact_graph'] else 'resolved_failure_nonexact';rows.append(dict(seed=seed,arm=arm,certified=True,**candidate))
            else:category='resolved_failure_fractional'
        else:category='unresolved'
        assert category==record['classification'];solver=record['solver'];assert solver['highs_version']=='1.12.0' and solver['numpy']=='2.3.4'
        for name,value in p['solver_options'].items():
            if name=='random_seed_base':assert solver['solver_options']['random_seed']==value+seed
            else:assert solver['solver_options'][name]==value
        arm_audits[arm]=dict(classification=category,certificate_verified=certified,integral_candidate_graph=candidate,fractional_variables=int((rawx==.5).sum()) if rawx.shape==(3240,) else None,solver=solver)
    aligned=observations['aligned']['given'];permuted=observations['permuted']['given'];shared=aligned&permuted
    assert np.array_equal(observations['aligned']['values'][shared],observations['permuted']['values'][shared])
    accounting=dict(aligned_queries=int(aligned.sum()),permuted_queries=int(permuted.sum()),shared_queries=int(shared.sum()),new_union_queries=int((aligned|permuted).sum()),new_calibration_queries=0)
    assert accounting==r['query_accounting'] and accounting['aligned_queries']==accounting['permuted_queries']==22500 and accounting['new_union_queries']==45000-accounting['shared_queries']
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),receipt_sha256=sha(d/'receipt.json'),model_sha256=sha(d/'model.npz'),
                rows=rows,arm_audits=arm_audits,query_accounting=accounting,suffix_substitution_gates=gates,independent_predictable_masks_replayed=1800,common_noise_rows_replayed=900)
    save(d/'audit.json',result);return result
