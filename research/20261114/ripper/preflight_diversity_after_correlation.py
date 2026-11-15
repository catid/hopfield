"""H1ci synthetic source/channel/endpoint/runner gates; no fresh cohort draws."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import csv,json,platform,tempfile,time,statistics
import numpy as np
import diversity_after_correlation_study as S
import diversity_after_correlation_data as D
from audit_diversity_after_correlation import aggregate,accounting_total
from pair_likelihood_math import UncertifiedCost
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn,*args,**kwargs):
    try:fn(*args,**kwargs)
    except (AssertionError,FileNotFoundError,TypeError):return
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
    for name,n in [('fixed-270-confirm',128),('repeat-query-tradeoff',128),('raw-count-quantization',128),('posterior-query-labels',128),('posterior-fresh-confirm',192),('pair-likelihood',128),('channel-misspecification',128),('estimated-channel',64),('heterogeneous-channel',144),('cell-channel',144),('correlated-repeat',144)]:
        p=S.read(S.HERE/f'protocol-{name}.json');base=p['solver']['random_seed_base'];grids[name]={base+41*i+d for i in range(n) for d in range(1,41)}
    streams={}
    for name in ['fixed-270-confirm','repeat-query-tradeoff','posterior-fresh-confirm','heterogeneous-channel','cell-channel','correlated-repeat']:
        p=S.read(S.HERE/f'protocol-{name}.json')
        for key in ['source_seed_base','balanced_seed_base','noise_seed_base','repeat1_seed_base','repeat2_seed_base','cell_channel_seed_base','shared_branch_seed_base']:
            if key in p:streams[name+':'+key]=set(range(p[key],p[key]+(72 if name in ['heterogeneous-channel','cell-channel','correlated-repeat'] else 64)))
    root_protocol=S.ROOT/'research/20261114/foureyes/fresh-pooled-donor.json'
    root_seeds=S.read(root_protocol)['seeds']
    for name,base in root_seeds.items():
        if name!='solver':streams['H4bc:'+name]=set(range(base,base+256))
    grids['H4bc']={root_seeds['solver']+44*i+slot for i in range(256) for slot in range(1,44)}
    for name,base in [('repeat1',202613900000),('repeat2',202613910000)]:streams['H4be_declared:'+name]=set(range(base,base+256))
    grids['H4be_declared']={2112631502+i for i in range(256)}
    base=design['seeds']['solver_base'];proposed={base+41*i+d for i in range(144) for d in range(1,41)}
    acquisition={name:set(range(base,base+72)) for name,base in design['seeds'].items() if name!='solver_base'}
    new=set().union(proposed,*acquisition.values());old=values|set().union(*grids.values(),*streams.values())
    assert len(proposed)==5760 and len(new)==6480 and not new&old and 0<min(proposed)<max(proposed)<2**31
    result=dict(status='passed',frozen_protocol_sha256=files,historical_explicit_seed_count=len(values),
        expanded_solver_ranges={name:[min(v),max(v)] for name,v in grids.items()},expanded_solver_counts={name:len(v) for name,v in grids.items()},
        expanded_acquisition_ranges={name:[min(v),max(v)] for name,v in streams.items()},new_solver_seed_range=[min(proposed),max(proposed)],new_solver_seed_count=5760,
        new_acquisition_ranges={name:[min(v),max(v)] for name,v in acquisition.items()},new_acquisition_seed_count=720,collisions=[],
        scope='Explicit frozen seed fields plus expandedH1bv/bw/ca/cf/cg/ch acquisition streams, H1bv/bw/by/bz/ca/cc/cd/ce/cf/cg/ch solvergrids and all declaredH4bc acquisition/solver ranges plus prospectiveH4be reservations. Other historical formulas not exhaustive; no PRNG independence proof.')
    S.save(S.HERE/'diversity-after-correlation-seed-audit.json',result);return result


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
    with tempfile.TemporaryDirectory(prefix='h1ci-lifecycle-') as tmp:
        try:
            root=Path(tmp);out=root/'synthetic';protocol=root/'protocol.json'
            for module in [S,D]:
                module.ROOT=root;module.OUT=out;module.PUBLIC=root/'public';module.REGISTRY=root/'registry';module.PROTOCOL=protocol
            source=root/'synthetic.csv';rr,cc=np.indices((9,9));board=((3*rr+rr//3+cc)%9).astype(np.int8).ravel()
            with source.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['answer'])
                for row in range(270):writer.writerow([''.join(str((int(v)+row)%9+1) for v in board)])
            p=dict(source='synthetic.csv',source_csv_sha256=S.sha(source),balanced_seed_bases=[993500,994500,995500],shared_branch_seed_base=993900,noise_seed_base=993600,repeat1_seed_base=993700,repeat2_seed_base=993800,diversity1_seed_base=996500,diversity2_seed_base=997500)
            S.save(protocol,p);S.verify=D.verify=lambda *args,**kwargs:p
            out.mkdir();ids=np.tile(np.arange(270,dtype=np.int64),(72,1));np.savez_compressed(out/'selected_sources.npz',source_rows=ids,all_source_row=np.arange(270,dtype=np.int64))
            S.save(out/'identity-seal.json',dict(status='synthetic_seal',protocol_sha256=S.sha(protocol),selected_sources_sha256=S.sha(out/'selected_sources.npz')))
            rejected(S.materialize);rejected(S.prepare);rejected(S.bank,0,p)
            S.predraw_all();S.materialize();manifest=S.read(out/'input_manifest.json');assert len(manifest['inputs'])==72 and len(manifest['predraw_receipts_sha256'])==72
            for entry in manifest['inputs']:
                data=S.npz(root/entry['completion_path']);D.validate_boards(data['solutions']);assert np.array_equal(data['source_row'],np.arange(270))
                for row in range(270):assert np.array_equal(data['solutions'][row],(board.astype(np.int64)+row)%9)
            rejected(S.prepare)
            S.save(out/'source-audit.json',dict(status='validated',protocol_sha256=S.sha(protocol),input_manifest_sha256=S.sha(out/'input_manifest.json'),fixture_only=True))
            S.prepare();seal=S.read(out/'query-record-seal.json');assert len(seal['record_sha256'])==len(seal['audit_sha256'])==72 and seal['physical_noisy_queries']==2332800
            rejected(S.prepare)
            for bank in range(72):assert D.audit_queries(bank,p)==S.read(out/f's{bank}/query-record-audit.json')
            path=out/'s0/potential_noise.npz';original=path.read_bytes();noise=S.npz(path);noise['first_iid_offsets'][0,0]=0;np.savez_compressed(path,**noise);rejected(D.audit_queries,0,p);path.write_bytes(original)
            path=out/'s0/query-records.npz';original=path.read_bytes();records=S.npz(path);cell=tuple(np.argwhere(records['given'][:90])[0]);records['first'][cell]=(records['first'][cell]+1)%9;np.savez_compressed(path,**records)
            rejected(D.audit_queries,0,p);rejected(S.bank,0,p);path.write_bytes(original)
            path=out/'s0/potential_noise.npz';original=path.read_bytes();noise=S.npz(path);records=S.npz(out/'s0/query-records.npz');clean=S.npz(root/manifest['inputs'][0]['completion_path'])['solutions'];mask=records['given']
            cell=tuple(np.argwhere(mask[:90]&(noise['shared_uniforms']<.5))[0]);noise['repeat1_iid_uniforms'][cell]=0.;noise['repeat1_iid_offsets'][cell]=(int(noise['repeat1_iid_offsets'][cell])%8)+1
            unchanged=S.reconstruct(mask,clean,noise);assert all(np.array_equal(records[k],unchanged[k]) for k in records)
            np.savez_compressed(path,**noise);rejected(D.audit_queries,0,p);path.write_bytes(original)
            noise=S.npz(path);noise['shared_uniforms'][cell]=.75;np.savez_compressed(path,**noise);rejected(D.audit_queries,0,p);path.write_bytes(original)
            path=out/'s0/masks.npz';original=path.read_bytes();masks=S.npz(path);masks['balanced'][0,0]=~masks['balanced'][0,0];np.savez_compressed(path,**masks);rejected(S.materialize);path.write_bytes(original)
        finally:
            for module,values in originals.items():
                for name,value in values.items():setattr(module,name,value)
    return dict(synthetic_banks=72,cohort_ids_or_labels_read=False,materialization_and_all_query_audits_passed=True,
        missing_masks_source_audit_or_query_seal_rejected=True,repeated_acquisition_rejected=True,raw_noise_observation_mask_and_branch_mutations_rejected=True,unused_shared_branch_potential_does_not_change_observations_but_fails_raw_audit=True)


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
    assert platform.node()=='ripper';target=S.HERE/'diversity-after-correlation-preflight.json'
    assert not target.exists() and not S.PROTOCOL.exists() and not S.REGISTRY.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    started=time.perf_counter();design=S.read(S.HERE/'diversity-after-correlation-proposal.json');seed=seed_audit(design)
    for module,protocol in [('estimated_channel_math.py','protocol-estimated-channel.json')]:
        assert S.sha(S.HERE/module)==S.read(S.HERE/protocol)['source_sha256']['research/20261114/ripper/'+module]
    for name in ['diversity-after-correlation-math-check.json','correlated-repeat-math-check.json','channel-misspecification-math-check.json']:
        gate=S.read(S.HERE/name);assert gate['status']=='passed' and not gate['study_scores_read']
        for name,h in gate['source_sha256'].items():assert S.sha(S.HERE/name)==h
    import run_diversity_after_correlation as runner
    assert runner.OUT==S.OUT and runner.PUBLIC==S.PUBLIC and runner.PROTOCOL==S.PROTOCOL
    commands=[(a,None) for a in ['select','predraw-all','materialize','source-audit','prepare']]+[('worker',i) for i in range(8)]+[('summary',None)]
    for action,index in commands:
        cmd=runner.command(action,index);assert cmd[1]==str(S.HERE/'diversity_after_correlation_study.py') and Path(cmd[1]).exists() and cmd[2]==action and cmd[3:]==([] if index is None else ['--seed',str(index)])
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix='h1ci-late-failure-') as tmp:
        out=Path(tmp)/'out';public=Path(tmp)/'public';out.mkdir();public.mkdir()
        for path in [out/'audit.json',public/'summary.json',out/'bank-evidence.json']:path.write_text('{}\n')
        with patch.object(runner,'OUT',out),patch.object(runner,'PUBLIC',public):runner.discard_primary()
        assert not (out/'audit.json').exists() and not (public/'summary.json').exists() and (out/'bank-evidence.json').read_text()=='{}\n'
    life=lifecycle();skips=numerical_skips()
    rng=np.random.default_rng(993401);mask=np.concatenate([S.make_masks(993402+k) for k in range(3)])
    assert np.array_equal(mask,np.concatenate([D.reference_masks(993402+k) for k in range(3)])) and mask.shape==(270,81) and (mask.sum(0)==240).all() and (mask.sum(1)==72).all()
    ii,jj=np.triu_indices(81,1);assert ((mask.astype(np.int64).T@mask.astype(np.int64))[ii,jj]==213).all()
    account=D.literal_account(mask);assert account==dict(full_physical_noisy_union=32400,supplied_correlated90_deployment=19440,supplied_single270_deployment=19440,shared_between_methods=6480)
    totals=accounting_total([dict(query_accounting=account)]*72,dict(clean_unique_oracle_cells=1574640));assert totals['full_physical_noisy_union']==2332800 and totals['official_noisy_label_sequence_after']==23292759
    rr,cc=np.indices((9,9));board=((3*rr+rr//3+cc)%9).astype(np.int8).ravel();clean=np.tile(board,(270,1));D.validate_boards(clean)
    for kind in ['dtype','shape','range','structure']:
        bad=clean.astype(np.int64) if kind=='dtype' else clean[:269] if kind=='shape' else clean.copy()
        if kind=='range':bad[0,0]=9
        if kind=='structure':bad[0,0]=bad[0,1]
        rejected(D.validate_boards,bad)
    # Synthetic CSV serialization/parser parity; official CSV header only, no row draw.
    with tempfile.TemporaryDirectory(prefix='h1ci-csv-') as tmp:
        path=Path(tmp)/'toy.csv'
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['answer']);w.writerow([''.join(str(int(x)+1) for x in board)]);w.writerow([''.join(str((int(x)+1)%9+1) for x in board)])
        got=list(S.selected_answers(path,{0,1}));assert [i for i,_ in got]==[0,1] and np.array_equal(got[0][1],board)
        with path.open(newline='') as f:
            rows=list(csv.reader(f));assert all(np.array_equal(value,np.array([int(c)-1 for c in rows[i+1][0]],np.int8)) for i,value in got)
    source=S.ROOT/S.read(S.HERE/'protocol-correlated-repeat.json')['source']
    with source.open(newline='') as f:assert 'answer' in next(csv.reader(f))
    channels=[];checks=[];timings={}
    noise=dict(shared_uniforms=np.random.default_rng(993410).random((90,81)))
    groups=[('first',0),('repeat1',0),('repeat2',0),('diversity1',1),('diversity2',2)]
    for query,(name,block) in enumerate(groups):
        gen=np.random.default_rng(993420+query);noise[name+'_iid_uniforms']=gen.random((90,81));noise[name+'_iid_offsets']=gen.integers(1,9,(90,81),dtype=np.int8)
    records=S.reconstruct(mask,clean,noise);shared=noise['shared_uniforms']<.5;potentials={}
    for name,block in groups:
        given=mask[90*block:90*(block+1)];block_clean=clean[90*block:90*(block+1)]
        before=time.perf_counter();potentials[name]=D.reference_corrupt(block_clean,noise[name+'_iid_uniforms'],noise[name+'_iid_offsets'],S.RATE);elapsed=time.perf_counter()-before
        expected=np.where(shared,potentials['first'],potentials[name]) if name in ['repeat1','repeat2'] else potentials[name]
        assert np.array_equal(records[name][given],expected[given]) and (records[name][~given]==-1).all();timings[name+'_literal_corrupt_seconds']=elapsed
    arm_data={arm:S.arm_values(records,arm) for arm in S.ARMS}
    repeated,rg=arm_data[S.ARMS[0]];diverse,dg=arm_data[S.ARMS[1]]
    assert np.array_equal(repeated[0],diverse[0,:90]) and np.array_equal(rg,dg[:90])
    for target_arm,other,field,block in [(S.ARMS[0],S.ARMS[1],'repeat1',0),(S.ARMS[1],S.ARMS[0],'diversity1',1)]:
        changed={k:v.copy() for k,v in records.items()};cell=tuple(np.argwhere(mask[90*block:90*(block+1)])[0]);changed[field][cell]=(changed[field][cell]+1)%9
        assert not np.array_equal(S.arm_values(changed,target_arm)[0],arm_data[target_arm][0]) and np.array_equal(S.arm_values(changed,other)[0],arm_data[other][0])
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);altered=S.reconstruct(mask,changed,noise)
    for arm in S.ARMS:
        before,given=arm_data[arm];after,given2=S.arm_values(altered,arm);assert np.array_equal(given,given2) and (before[:,given]!=after[:,given]).all()
    from unittest.mock import patch
    scorer_calls=[]
    def checked(fn,label,arm):
        def call(v,g,channel):
            assert v is arm_data[arm][0] and g is arm_data[arm][1]
            assert channel==(S.PARAMETERS if arm==S.ARMS[0] else S.RATE)
            scorer_calls.append(label);return fn(v,g,channel)
        return call
    with patch.object(S,'supplied_score',checked(S.supplied_score,'joint_producer',S.ARMS[0])),patch.object(S,'supplied_reference',checked(S.supplied_reference,'joint_reference',S.ARMS[0])),patch.object(S,'single_score',checked(S.single_score,'single_producer',S.ARMS[1])),patch.object(S,'single_reference',checked(S.single_reference,'single_reference',S.ARMS[1])):
        for arm in S.ARMS:
            values,given=arm_data[arm];before=time.perf_counter();arrays,estimator,numeric,reference=S.method_costs(values,given,arm);elapsed=time.perf_counter()-before
            assert estimator is None and numeric['fit_allowed'] and arrays['cost_certified'].all() and reference['certified']
            checks.append(dict(arm=arm,shape=list(values.shape),all_costs_certified=True,wall_seconds=elapsed))
    assert scorer_calls==['joint_producer','joint_reference','single_producer','single_reference']
    for arm in S.ARMS:
        values,given=arm_data[arm]
        rejected(S.method_costs,values,given,arm,noise)
        rejected(S.method_costs,values.astype(np.int64),given,arm)
        rejected(S.method_costs,values,given.astype(np.int8),arm)
        other=S.ARMS[1-S.ARMS.index(arm)];rejected(S.method_costs,*arm_data[other],arm)
        for hidden in [False,True]:
            invalid=values.copy();cell=tuple(np.argwhere(~given if hidden else given)[0]);invalid[(0,)+cell]=0 if hidden else 9;rejected(S.method_costs,invalid,given,arm)
    for rate,hexvalue in [(S.RATE,'0x1.3333333333333p-4'),(S.RHO,'0x1.0000000000000p-1')]:
        event=D.channel_event(rate);n,den=event['event_probability'];threshold=Fraction(float(Fraction(*rate)))
        assert Fraction(n-1,den)<threshold<=Fraction(n,den) and event['threshold_hex']==hexvalue;channels.append(event)
    assert channels[1]['event_probability']==[2**52,2**53]
    for name,block in groups:
        cells=[tuple(v) for v in np.argwhere(mask[90*block:90*(block+1)])[:3]];boundary={k:arr.copy() for k,arr in noise.items()};threshold=float(Fraction(*S.RATE))
        for value,cell in zip([np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf)],cells):boundary[name+'_iid_uniforms'][cell]=value;boundary['shared_uniforms'][cell]=.75
        out=S.reconstruct(mask,clean,boundary)[name];block_clean=clean[90*block:90*(block+1)];assert out[cells[0]]!=block_clean[cells[0]] and out[cells[1]]==block_clean[cells[1]] and out[cells[2]]==block_clean[cells[2]]
    cells=[tuple(v) for v in np.argwhere(mask[:90])[:3]];boundary={name:arr.copy() for name,arr in noise.items()}
    for value,cell in zip([np.nextafter(.5,-np.inf),.5,np.nextafter(.5,np.inf)],cells):
        boundary['shared_uniforms'][cell]=value;boundary['first_iid_uniforms'][cell]=.9
        for name in ['repeat1','repeat2']:boundary[name+'_iid_uniforms'][cell]=0.;boundary[name+'_iid_offsets'][cell]=1
    out=S.reconstruct(mask,clean,boundary)
    for name in ['repeat1','repeat2']:assert out[name][cells[0]]==out['first'][cells[0]] and all(out[name][cell]!=out['first'][cell] for cell in cells[1:])
    good,bad,frac,unknown='success','resolved_failure_nonexact','resolved_failure_fractional','unresolved';endpoints=[]
    cases=[({},('supported','falsified')),
        ({(i,0):bad for i in range(48)},('supported','supported')),
        ({(i,0):bad for i in range(47)},('supported','falsified')),
        ({**{(i,0):bad for i in range(47)},(47,0):unknown},('supported','inconclusive')),
        ({(i,1):bad for i in range(60,72)},('supported','falsified')),
        ({(i,1):bad for i in range(59,72)},('falsified','falsified')),
        ({**{(i,1):bad for i in range(60,72)},(59,1):unknown},('inconclusive','falsified')),
        ({(i,a):bad for i in range(72) for a in [0,1]},('falsified','falsified')),
        ({(i,a):unknown for i in range(72) for a in [0,1]},('inconclusive','inconclusive'))]
    for changes,want in cases:
        result=fixture(changes);assert tuple(result['endpoints'][k]['decision'] for k in ['diverse_exact','net_recovery_gain'])==want;endpoints.append(result)
    combos=[]
    for left,right in product([good,bad,frac,unknown],repeat=2):
        result=fixture({(0,0):left,(0,1):right})['endpoints']['net_recovery_gain']
        possible=lambda value: [0,1] if value==unknown else [int(value==good)]
        differences=[d-r for r in possible(left) for d in possible(right)]
        assert result['lower']==min(differences) and result['upper']==max(differences);combos.append(dict(left=left,right=right,**result))
    folder=S.ROOT/'artifacts/research/20261114/ripper/diversity_after_correlation_preflight';folder.mkdir(parents=True,exist_ok=True)
    tiny=np.array([4,2,7,1,9,3],np.int64);i,j=np.triu_indices(4,1);x,y,meta=solve(4,1,i,j,tiny,993430-2026111502,folder/'synthetic.log');cert,w=attempt_certificate(4,1,i,j,tiny,x,y);proof=[w[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert meta['solver_optimal'] and cert['certified'] and check_certificate(4,1,i,j,tiny,*proof)
    for k in range(3):
        altered=[v.copy() for v in proof];altered[k][0]+=1;assert not check_certificate(4,1,i,j,tiny,*altered)
    partitions=[list(range(w,72,8)) for w in range(8)];assert sorted(i for part in partitions for i in part)==list(range(72))
    sources=set(S.read(S.HERE/'correlated-repeat-preflight.json')['source_sha256'])
    sources|={'research/20261114/ripper/'+name for name in ['diversity_after_correlation_study.py','diversity_after_correlation_data.py','audit_diversity_after_correlation.py','preflight_diversity_after_correlation.py','run_diversity_after_correlation.py','preflight_diversity_after_correlation_math.py']}
    result=dict(status='passed',host=platform.node(),cohort_draws_or_estimates_read=False,proposal_sha256=S.sha(S.HERE/'diversity-after-correlation-proposal.json'),seed_audit_sha256=S.sha(S.HERE/'diversity-after-correlation-seed-audit.json'),
        immutable_method_sources_verified=True,inherited_numerical_gates_verified=True,synthetic_lifecycle=life,numerical_skips=skips,runner_all14commands=True,late_failure_primary_cleanup_preserves_bank_evidence=True,source_format_and_csv_parser_parity=True,official_csv_header_only_read=True,
        balanced_mask_and_literal_query_identity=True,branch_stream_and_threshold_parity=True,branch_blind_scorer_calls=scorer_calls,paired_math_gate_sha256=S.sha(S.HERE/'diversity-after-correlation-math-check.json'),query_accounting=totals,channels=channels,fullshape_method_checks=checks,corruption_timings=timings,no_estimation_or_counterarm_input=True,dtype_range_shape_hidden_label_guards=True,literal_shared_first90_response_and_mask=True,
        endpoint_cases=endpoints,paired_uncertainty_cases=combos,synthetic_certificate=cert,mutations_rejected=3,worker_partition=partitions,source_sha256={rel:S.sha(S.ROOT/rel) for rel in sorted(sources)},wall_seconds=time.perf_counter()-started)
    S.save(target,result);print(json.dumps(dict(status='passed',source_files=len(sources),new_solver_seed_range=seed['new_solver_seed_range'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
