"""H1cf synthetic source/channel/endpoint/runner gates; no fresh cohort draws."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import csv,json,platform,tempfile,time,statistics
import numpy as np
import heterogeneous_channel_study as S
import heterogeneous_channel_data as D
from audit_heterogeneous_channel import aggregate,accounting_total
from pair_likelihood_math import UncertifiedCost
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn,*args,**kwargs):
    try:fn(*args,**kwargs)
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Invalid synthetic case accepted')


def seed_audit(design):
    values=set();files={}
    def scan(v,seed=False):
        if isinstance(v,dict):
            for key,item in v.items():scan(item,seed or 'seed' in key.lower())
        elif isinstance(v,list):
            for item in v:scan(item,seed)
        elif seed and type(v) is int:values.add(v)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:d=S.read(path)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(d,dict) and 'frozen_utc' in d:files[str(path.relative_to(S.ROOT))]=S.sha(path);scan(d)
    grids={}
    for name,n in [('fixed-270-confirm',128),('repeat-query-tradeoff',128),('raw-count-quantization',128),('posterior-query-labels',128),('posterior-fresh-confirm',192),('pair-likelihood',128),('channel-misspecification',128),('estimated-channel',64)]:
        p=S.read(S.HERE/f'protocol-{name}.json');base=p['solver']['random_seed_base'];grids[name]={base+41*i+d for i in range(n) for d in range(1,41)}
    streams={}
    for name in ['fixed-270-confirm','repeat-query-tradeoff','posterior-fresh-confirm']:
        p=S.read(S.HERE/f'protocol-{name}.json')
        for key in ['source_seed_base','balanced_seed_base','noise_seed_base','repeat1_seed_base','repeat2_seed_base']:
            if key in p:streams[name+':'+key]=set(range(p[key],p[key]+64))
    base=design['seeds']['solver_base'];proposed={base+41*i+d for i in range(144) for d in range(1,41)}
    acquisition={name:set(range(base,base+72)) for name,base in design['seeds'].items() if name!='solver_base'}
    new=set().union(proposed,*acquisition.values());old=values|set().union(*grids.values(),*streams.values())
    assert len(proposed)==5760 and len(new)==6120 and not new&old and 0<min(proposed)<max(proposed)<2**31
    result=dict(status='passed',frozen_protocol_sha256=files,historical_explicit_seed_count=len(values),
        expanded_solver_ranges={name:[min(v),max(v)] for name,v in grids.items()},expanded_solver_counts={name:len(v) for name,v in grids.items()},
        expanded_acquisition_ranges={name:[min(v),max(v)] for name,v in streams.items()},new_solver_seed_range=[min(proposed),max(proposed)],new_solver_seed_count=5760,
        new_acquisition_ranges={name:[min(v),max(v)] for name,v in acquisition.items()},new_acquisition_seed_count=360,collisions=[],
        scope='Explicit frozen seed fields plus expandedH1bv/bw/ca acquisition streams andH1bv/bw/by/bz/ca/cc/cd/ce solvergrids. Other historical formulas not exhaustive; no PRNG independence proof.')
    S.save(S.HERE/'heterogeneous-channel-seed-audit.json',result);return result


def fixture(changes):
    rows=[]
    for seed in range(72):
        for ai,arm in enumerate(S.ARMS):
            kind=changes.get((seed,ai),'success')
            rows.append(dict(index=2*seed+ai,seed=seed,arm=arm,classification=kind,selected_degree=20,fixed_degree20=dict(classification=kind),top810=dict(classification=kind),estimator=None,numeric=dict(fit_allowed=kind!='unresolved')))
    return aggregate(rows)


def lifecycle():
    """Exercise all72 synthetic materializations/query audits without any cohort IDs."""
    attributes=['ROOT','HERE','OUT','PUBLIC','REGISTRY','PROTOCOL','verify']
    originals={module:{name:getattr(module,name) for name in attributes} for module in [S,D]}
    with tempfile.TemporaryDirectory(prefix='h1cf-lifecycle-') as tmp:
        try:
            root=Path(tmp);out=root/'synthetic';protocol=root/'protocol.json'
            for module in [S,D]:
                module.ROOT=root;module.OUT=out;module.PUBLIC=root/'public';module.REGISTRY=root/'registry';module.PROTOCOL=protocol
            source=root/'synthetic.csv';rr,cc=np.indices((9,9));board=((3*rr+rr//3+cc)%9).astype(np.int8).ravel()
            with source.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['answer'])
                for row in range(90):writer.writerow([''.join(str((int(v)+row)%9+1) for v in board)])
            p=dict(source='synthetic.csv',source_csv_sha256=S.sha(source),balanced_seed_base=992500,noise_seed_base=992600,repeat1_seed_base=992700,repeat2_seed_base=992800)
            S.save(protocol,p);S.verify=D.verify=lambda *args,**kwargs:p
            out.mkdir();ids=np.tile(np.arange(90,dtype=np.int64),(72,1));np.savez_compressed(out/'selected_sources.npz',source_rows=ids,all_source_row=np.arange(90,dtype=np.int64))
            S.save(out/'identity-seal.json',dict(status='synthetic_seal',protocol_sha256=S.sha(protocol),selected_sources_sha256=S.sha(out/'selected_sources.npz')))
            rejected(S.materialize);rejected(S.prepare);rejected(S.bank,0,p)
            S.predraw_all();S.materialize();manifest=S.read(out/'input_manifest.json');assert len(manifest['inputs'])==72 and len(manifest['predraw_receipts_sha256'])==72
            for entry in manifest['inputs']:
                data=S.npz(root/entry['completion_path']);D.validate_boards(data['solutions']);assert np.array_equal(data['source_row'],np.arange(90))
                for row in range(90):assert np.array_equal(data['solutions'][row],(board+row)%9)
            rejected(S.prepare)
            S.save(out/'source-audit.json',dict(status='validated',protocol_sha256=S.sha(protocol),input_manifest_sha256=S.sha(out/'input_manifest.json'),fixture_only=True))
            S.prepare();seal=S.read(out/'query-record-seal.json');assert len(seal['record_sha256'])==len(seal['audit_sha256'])==72 and seal['physical_noisy_queries']==1399680
            rejected(S.prepare)
            for bank in range(72):assert D.audit_queries(bank,p)==S.read(out/f's{bank}/query-record-audit.json')
            path=out/'s0/potential_noise.npz';original=path.read_bytes();noise=S.npz(path);noise['first_iid_offsets'][0,0]=0;np.savez_compressed(path,**noise);rejected(D.audit_queries,0,p);path.write_bytes(original)
            path=out/'s0/query-records.npz';original=path.read_bytes();records=S.npz(path);cell=tuple(np.argwhere(records['given'])[0]);records['first'][cell]=(records['first'][cell]+1)%9;np.savez_compressed(path,**records)
            rejected(D.audit_queries,0,p);rejected(S.bank,0,p);path.write_bytes(original)
            path=out/'s0/masks.npz';original=path.read_bytes();masks=S.npz(path);masks['balanced'][0,0]=~masks['balanced'][0,0];np.savez_compressed(path,**masks);rejected(S.materialize);path.write_bytes(original)
        finally:
            for module,values in originals.items():
                for name,value in values.items():setattr(module,name,value)
    return dict(synthetic_banks=72,cohort_ids_or_labels_read=False,materialization_and_all_query_audits_passed=True,
        missing_masks_source_audit_or_query_seal_rejected=True,repeated_acquisition_rejected=True,raw_noise_observation_and_mask_mutations_rejected=True)


def numerical_skips():
    import estimated_channel_study as E
    previous=E.solve;calls=[]
    def forbidden(*args,**kwargs):calls.append(1);raise AssertionError('Uncertified cost reached solver')
    E.solve=forbidden
    try:
        for production,reference in [(False,True),(True,False),(False,False)]:
            arrays=dict(cost=np.zeros(3240,np.int64),cost_certified=np.full(3240,production,bool),cost_precision=np.full(3240,80,np.int64))
            numeric=S.numeric_state(arrays,dict(certified=reference,error=None if reference else 'synthetic_uncertified_interval'))
            out,records=S.fit_models(arrays,numeric,0,{},Path('/nonexistent-synthetic'),'synthetic')
            assert out is arrays and records==[] and not numeric['fit_allowed'] and 'top810_support' not in arrays
    finally:E.solve=previous
    assert not calls
    return dict(uncertified_producer_reference_and_both_skipped=True,solver_calls=0,placeholder_costs_never_submitted=True)


def main():
    assert platform.node()=='ripper';target=S.HERE/'heterogeneous-channel-preflight.json'
    assert not target.exists() and not S.PROTOCOL.exists() and not S.REGISTRY.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    started=time.perf_counter();design=S.read(S.HERE/'heterogeneous-channel-proposal.json');seed=seed_audit(design)
    for module,protocol in [('channel_misspecification_math.py','protocol-channel-misspecification.json'),('estimated_channel_math.py','protocol-estimated-channel.json')]:
        assert S.sha(S.HERE/module)==S.read(S.HERE/protocol)['source_sha256']['research/20261114/ripper/'+module]
    for name in ['channel-misspecification-math-check.json','estimated-channel-math-check.json']:
        gate=S.read(S.HERE/name);assert gate['status']=='passed' and not gate['study_scores_read']
        for name,h in gate['source_sha256'].items():assert S.sha(S.HERE/name)==h
    import run_heterogeneous_channel as runner
    assert runner.OUT==S.OUT and runner.PUBLIC==S.PUBLIC and runner.PROTOCOL==S.PROTOCOL
    commands=[(a,None) for a in ['select','predraw-all','materialize','source-audit','prepare']]+[('worker',i) for i in range(8)]+[('summary',None)]
    for action,index in commands:
        cmd=runner.command(action,index);assert cmd[1]==str(S.HERE/'heterogeneous_channel_study.py') and Path(cmd[1]).exists() and cmd[2]==action and cmd[3:]==([] if index is None else ['--seed',str(index)])
    life=lifecycle();skips=numerical_skips()
    rng=np.random.default_rng(992401);mask=S.make_masks(992402)
    assert np.array_equal(mask,D.reference_masks(992402)) and mask.shape==(90,81) and (mask.sum(0)==80).all() and (mask.sum(1)==72).all()
    ii,jj=np.triu_indices(81,1);assert ((mask.astype(np.int64).T@mask.astype(np.int64))[ii,jj]==71).all()
    account=D.literal_account(mask);assert all(v==19440 for v in account.values())
    totals=accounting_total([dict(query_accounting=account)]*72,dict(clean_unique_oracle_cells=524880));assert totals['full_physical_noisy_union']==1399680 and totals['official_noisy_label_sequence_after']==18160599
    rr,cc=np.indices((9,9));board=((3*rr+rr//3+cc)%9).astype(np.int8).ravel();clean=np.tile(board,(90,1));D.validate_boards(clean)
    for kind in ['dtype','shape','range','structure']:
        bad=clean.astype(np.int64) if kind=='dtype' else clean[:89] if kind=='shape' else clean.copy()
        if kind=='range':bad[0,0]=9
        if kind=='structure':bad[0,0]=bad[0,1]
        rejected(D.validate_boards,bad)
    # Synthetic CSV serialization/parser parity; official CSV header only, no row draw.
    with tempfile.TemporaryDirectory(prefix='h1cf-csv-') as tmp:
        path=Path(tmp)/'toy.csv'
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['answer']);w.writerow([''.join(str(int(x)+1) for x in board)]);w.writerow([''.join(str((int(x)+1)%9+1) for x in board)])
        got=list(S.selected_answers(path,{0,1}));assert [i for i,_ in got]==[0,1] and np.array_equal(got[0][1],board)
        with path.open(newline='') as f:
            rows=list(csv.reader(f));assert all(np.array_equal(value,np.array([int(c)-1 for c in rows[i+1][0]],np.int8)) for i,value in got)
    source=S.ROOT/S.read(S.HERE/'protocol-posterior-fresh-confirm.json')['source']
    with source.open(newline='') as f:assert 'answer' in next(csv.reader(f))
    channels=[];checks=[];timings={}
    for ri,rate in enumerate(S.RATES):
        event=D.channel_event(rate);n,den=event['event_probability'];threshold=Fraction(float(Fraction(*rate)))
        assert Fraction(n-1,den)<threshold<=Fraction(n,den) and event['threshold_hex']==['0x1.3333333333333p-5','0x1.3333333333333p-4','0x1.3333333333333p-3'][ri]
        channels.append(event);noise={}
        for query,name in enumerate(['first','repeat1','repeat2']):
            gen=np.random.default_rng(992410+3*ri+query);noise[name+'_iid_uniforms']=gen.random((90,81));noise[name+'_iid_offsets']=gen.integers(1,9,(90,81),dtype=np.int8)
        records=S.reconstruct(mask,clean,noise,rate);values,given=S.arm_values(records)
        for name in ['first','repeat1','repeat2']:
            before=time.perf_counter();expected=D.reference_corrupt(clean,noise[name+'_iid_uniforms'],noise[name+'_iid_offsets'],rate);elapsed=time.perf_counter()-before
            assert np.array_equal(records[name][mask],expected[mask]) and (records[name][~mask]==-1).all()
            timings[f'rate{ri}_{name}_literal_corrupt_seconds']=elapsed
        changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);altered=S.reconstruct(mask,changed,noise,rate)
        assert all((records[name][mask]!=altered[name][mask]).all() for name in ['first','repeat1','repeat2'])
        for arm in S.ARMS:
            before=time.perf_counter();arrays,estimator,numeric,reference=S.method_costs(values,given,arm,rate if arm=='supplied_nominal' else None);elapsed=time.perf_counter()-before
            assert numeric['fit_allowed'] and arrays['cost_certified'].all() and reference['certified']
            if estimator is not None:assert estimator['total_comparisons']==19440
            checks.append(dict(acquisition=list(rate),arm=arm,synthetic_estimator=estimator,all_costs_certified=True,wall_seconds=elapsed))
        rejected(S.method_costs,values,given,'estimated_channel',rate)
        # Literal below/equal/above threshold behavior, with no fresh RNG.
        boundary={name:arr.copy() for name,arr in noise.items()};cells=list(zip(*np.where(mask)))[:3]
        for value,(row,cell) in zip([np.nextafter(float(threshold),-np.inf),float(threshold),np.nextafter(float(threshold),np.inf)],cells):boundary['first_iid_uniforms'][row,cell]=value
        out=S.reconstruct(mask,clean,boundary,rate)['first'];assert out[cells[0]]!=clean[cells[0]] and out[cells[1]]==clean[cells[1]] and out[cells[2]]==clean[cells[2]]
    # Invalid estimate never reaches likelihood scoring; supplied metadata is rejected.
    distinct=np.stack([np.full((90,81),k,np.int8) for k in range(3)]);distinct[:,~mask]=-1;old=(S.estimated_score,S.estimated_reference)
    def forbidden(*args,**kwargs):raise AssertionError('Invalid estimate reached cost calculation')
    S.estimated_score=S.estimated_reference=forbidden
    try:
        arrays,estimator,numeric,reference=S.method_costs(distinct,mask,'estimated_channel')
        assert not estimator['fit_allowed'] and numeric is reference is None and set(arrays)=={'estimator_unexecuted'}
    finally:S.estimated_score,S.estimated_reference=old
    good,bad,frac,unknown='success','resolved_failure_nonexact','resolved_failure_fractional','unresolved';endpoints=[]
    cases=[({},('supported','supported','supported')),
        ({(i,a):bad for i in range(60,72) for a in [0,1]},('supported','supported','supported')),
        ({(i,a):bad for i in range(59,72) for a in [0,1]},('falsified','falsified','supported')),
        ({**{(i,a):bad for i in range(60,72) for a in [0,1]},(59,0):unknown,(59,1):unknown},('inconclusive','inconclusive','supported')),
        ({(0,1):bad,(1,1):frac},('supported','supported','supported')),
        ({(0,1):bad,(1,1):bad,(2,1):bad},('supported','supported','falsified')),
        ({(0,1):bad,(1,1):bad,(2,1):unknown},('supported','supported','inconclusive')),
        ({(0,1):bad,(1,1):bad,(2,0):unknown,(2,1):bad},('supported','supported','inconclusive')),
        ({(i,a):unknown for i in range(72) for a in [0,1]},('inconclusive','inconclusive','inconclusive'))]
    for changes,want in cases:
        result=fixture(changes);assert tuple(result['endpoints'][k]['decision'] for k in ['supplied_exact','estimated_exact','paired_regressions'])==want;endpoints.append(result)
    combos=[]
    for left,right in product([good,bad,frac,unknown],repeat=2):
        result=fixture({(0,0):left,(0,1):right})['endpoints']['paired_regressions'];assert result['lower']==int(left==good and right in [bad,frac]) and result['upper']==int(left in [good,unknown] and right!=good);combos.append(dict(left=left,right=right,**result))
    folder=S.ROOT/'artifacts/research/20261114/ripper/heterogeneous_channel_preflight';folder.mkdir(parents=True,exist_ok=True)
    tiny=np.array([4,2,7,1,9,3],np.int64);i,j=np.triu_indices(4,1);x,y,meta=solve(4,1,i,j,tiny,992430-2026111502,folder/'synthetic.log');cert,w=attempt_certificate(4,1,i,j,tiny,x,y);proof=[w[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert meta['solver_optimal'] and cert['certified'] and check_certificate(4,1,i,j,tiny,*proof)
    for k in range(3):
        altered=[v.copy() for v in proof];altered[k][0]+=1;assert not check_certificate(4,1,i,j,tiny,*altered)
    partitions=[list(range(w,72,8)) for w in range(8)];assert sorted(i for part in partitions for i in part)==list(range(72))
    sources=set(S.read(S.HERE/'estimated-channel-preflight.json')['source_sha256'])|set(S.read(S.HERE/'posterior-fresh-confirm-preflight.json')['source_sha256'])
    sources|={'research/20261114/ripper/'+name for name in ['heterogeneous_channel_study.py','heterogeneous_channel_data.py','audit_heterogeneous_channel.py','preflight_heterogeneous_channel.py','run_heterogeneous_channel.py']}
    result=dict(status='passed',host=platform.node(),cohort_draws_or_estimates_read=False,proposal_sha256=S.sha(S.HERE/'heterogeneous-channel-proposal.json'),seed_audit_sha256=S.sha(S.HERE/'heterogeneous-channel-seed-audit.json'),
        immutable_method_sources_verified=True,inherited_numerical_gates_verified=True,synthetic_lifecycle=life,numerical_skips=skips,runner_all14commands=True,source_format_and_csv_parser_parity=True,official_csv_header_only_read=True,
        balanced_mask_and_literal_query_identity=True,query_accounting=totals,channels=channels,fullshape_method_checks=checks,corruption_timings=timings,invalid_estimate_score_calls=0,estimator_nominal_metadata_rejected=True,
        endpoint_cases=endpoints,paired_uncertainty_cases=combos,synthetic_certificate=cert,mutations_rejected=3,worker_partition=partitions,source_sha256={rel:S.sha(S.ROOT/rel) for rel in sorted(sources)},wall_seconds=time.perf_counter()-started)
    S.save(target,result);print(json.dumps(dict(status='passed',source_files=len(sources),new_solver_seed_range=seed['new_solver_seed_range'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
