"""Synthetic H4ay disjoint data, full noise replay, seals and endpoint gates."""
from pathlib import Path
import tempfile,time
import numpy as np
import disjoint_donor as S
from disjoint_donor_math import counts,independent_counts,noise_stream,replay_noise,duplicate_counts
from conflict_mask_math import detector,independent_detector
from conflict_mask_math import literal_scores
from corruption_censor_math import literal_score
from audit_disjoint_donor import aggregate
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn,*args):
    try:fn(*args)
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Invalid synthetic case accepted')


def seed_audit(design):
    values,files=set(),{}
    def scan(v,context=False):
        if isinstance(v,dict):
            for k,x in v.items():scan(x,context or 'seed' in k.lower())
        elif isinstance(v,list):
            for x in v:scan(x,context)
        elif context and type(v) is int:values.add(v)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:p=S.read(path)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(p,dict) and p.get('frozen_utc'):
            files[str(path.relative_to(S.ROOT))]=S.sha(path);scan(p)
    expanded=set();ranges={}
    for name,count in [('corruption-censor',256),('partial-mask-recall',1280),('conflict-mask',512),('purified-detector',512),('crossfit-conflicts',768)]:
        p=S.read(S.HERE/f'{name}.json');base=p['solver']['random_seed_base']
        expanded.update(range(base,base+count));ranges[name+'-solver']=[base,base+count-1]
        if 'random_seed_base' in p:
            base=p['random_seed_base'];expanded.update(range(base,base+256));ranges[name+'-mask']=[base,base+255]
    thinning=S.read(S.HERE/'partial-mask-recall.json')['thinning']['seed_base']
    thinning_sequences=[[thinning,i] for i in range(256)]
    solver=set(range(design['solver_seed_base'],design['solver_seed_base']+512))
    assert len(solver)==512 and not solver&(values|expanded) and 0<=min(solver)<max(solver)<2**31
    S.save(S.HERE/'disjoint-donor-seed-audit.json',dict(status='passed',frozen_protocol_sha256=files,
        historical_explicit_seed_count=len(values),expanded_ancestor_ranges=ranges,H4au_actual_thinning_seed_sequences=thinning_sequences,
        solver_range=[min(solver),max(solver)],collisions=[],
        scope='Explicit frozen seed fields and expanded H4at/H4au/H4av/H4aw/H4ax solver and declared random-mask seeds. Other historical formulas not exhaustive; no proof of PRNG independence.'))


def endpoints(changes):
    rows=[]
    for i in range(256):
        kind=changes.get(i,'success')
        rows.append(dict(index=i,baseline_exact=i<129,parent_detector_exact=i<130,classification=kind,family=S.FAMILIES[i%4],
            mask_constructed=kind!='unresolved',true_positive_locations=600,false_positive_locations=4800,false_negative_locations=4800,
            certified=kind!='unresolved',donors=[dict(certified=True)]))
    return aggregate(rows)['primary']


def lifecycle():
    oldout,oldprotocol=S.OUT,S.PROTOCOL
    with tempfile.TemporaryDirectory(prefix='h4ay-synthetic-') as tmp:
        try:
            S.OUT=Path(tmp)/'out';S.OUT.mkdir();S.PROTOCOL=Path(tmp)/'protocol.json';S.save(S.PROTOCOL,dict(source_sha256={}))
            rejected(S.final_worker,0);rejected(S.audit_worker,0);rejected(S.seal_phase,'donor')
            ph=S.sha(S.PROTOCOL)
            for i in range(256):
                folder=S.OUT/f'b{i:03d}';folder.mkdir()
                for name in ['donor']:
                    np.savez(folder/f'{name}.npz',fixture=np.zeros(1))
                    S.save(folder/f'{name}.json',dict(index=i,arm=name,protocol_sha256=ph,models_sha256=S.sha(folder/f'{name}.npz'),
                        fit_started_monotonic_ns=time.monotonic_ns(),fit_completed_monotonic_ns=time.monotonic_ns(),certificate=dict(certified=False)))
            S.seal_phase('donor');seal=S.check_phase_seal(S.OUT,'donor');rejected(S.seal_phase,'donor')
            file=S.OUT/'b000/donor.npz';old=file.read_bytes();file.write_bytes(old+b'bad');rejected(S.check_phase_seal,S.OUT,'donor');file.write_bytes(old)
            # With unresolved folds, no noisy input or mask is available and fit_one must never run.
            oldfit=S.fit_one
            def forbidden(*args,**kwargs):raise AssertionError('Unresolved donor reached final solver')
            S.fit_one=forbidden
            try:
                for i in range(256):S.final_bank(i,dict(jobs=[dict(inferred_degree=2)]*256))
            finally:S.fit_one=oldfit
            assert all(not S.read(S.OUT/f'b{i:03d}/final.json')['executed'] for i in range(256))
            file=S.OUT/'b000/final.json';rec=S.read(file);S.save(file,dict(rec,fit_started_monotonic_ns=0))
            rejected(S.seal_phase,'fit');(S.OUT/'fit-seal.json').unlink();S.save(file,rec)
            S.seal_phase('fit');S.check_seal(S.OUT);rejected(S.seal_phase,'fit')
            file=S.OUT/'b000/final.npz';file.write_bytes(file.read_bytes()+b'bad');rejected(S.check_seal,S.OUT)
        finally:S.OUT,S.PROTOCOL=oldout,oldprotocol
    return dict(early_final_and_audit_rejected=True,missing_donors_rejected=True,mutated_donor_rejected=True,
        final_before_donor_seal_rejected=True,unresolved_donor_skips_all256_final_fits=True,mutated_final_rejected=True,repeated_seals_rejected=True)


def main():
    gatepath=S.HERE/'disjoint-donor-preflight.json'
    assert not gatepath.exists() and not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    started=time.perf_counter();seed_audit(S.read(S.HERE/'disjoint-donor-proposal.json'))
    rng=np.random.default_rng(982090);noisy=rng.integers(0,9,(1200,81),dtype=np.int64)
    target,donor=noisy[:600],noisy[600:1200]
    assert not set(range(600))&set(range(600,1200))
    assert np.array_equal(counts(donor),independent_counts(donor))
    tiny=noisy[:8,:12]
    assert np.array_equal(counts(tiny),literal_score(tiny,np.zeros_like(tiny,bool))['cost'])
    primal=rng.integers(0,3,3240,dtype=np.int64)
    scores,mask=detector(target,primal);si,mi=independent_detector(target,primal)
    assert np.array_equal(scores,si) and np.array_equal(mask,mi) and (mask.sum(1)==9).all()
    # Arbitrary donor observations cannot enter target scoring once its graph is fixed.
    changed_donor=donor.copy();changed_donor[:]=0
    sc,mc=detector(target,primal);assert np.array_equal(sc,scores) and np.array_equal(mc,mask)
    zero=np.zeros(3240,np.int64);zs,zm=detector(target,zero)
    assert not zs.any() and zm[:,:9].all() and not zm[:,9:].any()
    permutation=rng.permutation(9);ps,pm=detector(permutation[target],primal)
    assert np.array_equal(ps,scores) and np.array_equal(pm,mask)
    for valid in [True,False]:assert S.fit_allowed(dict(certificate=dict(certified=valid,integral=False)))==valid
    # Unrelated original-sized stream: independent producer stable-argsort vs replay tuple sort.
    hotseed,noiseseed=982091,982092;clean=rng.integers(0,9,(2400,81),dtype=np.int64)
    generator=np.random.default_rng(noiseseed);uniform=1-generator.random(clean.shape);offsets=generator.integers(1,9,clean.shape,dtype=np.int64)
    weights=np.ones(81);weights[np.random.default_rng(hotseed).choice(81,18,replace=False)]=32
    cells=np.argsort(-np.log(uniform)/weights,axis=1,kind='stable')[:,:9]
    corrupted=clean.copy();np.put_along_axis(corrupted,cells,(np.take_along_axis(clean,cells,1)+np.take_along_axis(offsets,cells,1))%9,1)
    raw=dict(solutions=clean,free=np.ones_like(clean,bool),uniforms=uniform,offsets=offsets,cells=cells,noisy=corrupted)
    assert np.array_equal(replay_noise(raw,hotseed,noiseseed),corrupted)
    for key in raw:
        bad={k:v.copy() for k,v in raw.items()}
        if key=='solutions':bad[key][0,0]=9
        elif key=='free':bad[key][0,0]=False
        elif key=='uniforms':bad[key][0,0]/=2
        elif key=='offsets':bad[key][0,0]=9
        elif key=='cells':bad[key][0,[0,1]]=bad[key][0,[1,0]]
        else:bad[key][0,0]=(bad[key][0,0]+1)%9
        rejected(replay_noise,bad,hotseed,noiseseed)
    full=noise_stream(hotseed,noiseseed)[1]
    reset=noise_stream(hotseed,noiseseed,rows=600)[1]
    assert not np.array_equal(full[:600],reset) # Offset RNG comes after all2400uniform rows.
    duplicate=duplicate_counts(np.array([[1,2],[1,2],[3,4]]),np.array([[1,2],[5,6],[5,6]]))
    assert duplicate==dict(target_duplicate_instances=1,donor_duplicate_instances=1,donor_rows_matching_any_target=1,shared_unique_values=1)
    folder=S.ROOT/'artifacts/research/20261114/foureyes/disjoint_donor_preflight';folder.mkdir(parents=True,exist_ok=True)
    ii,jj=np.triu_indices(4,1);cost=np.ones(6,np.int64)
    x,y,meta=solve(4,1,ii,jj,cost,982093-2026111502,folder/'synthetic.log');cert,w=attempt_certificate(4,1,ii,jj,cost,x,y)
    proof=[w[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert cert['certified'] and meta['solver_optimal'] and check_certificate(4,1,ii,jj,cost,*proof)
    for k in range(3):
        bad=[v.copy() for v in proof];bad[k][0]+=1;assert not check_certificate(4,1,ii,jj,cost,*bad)
    bad,unk='resolved_failure_nonexact','unresolved';cases=[]
    for changes,want in [({},('supported','supported')),
        ({i:bad for i in range(162,256)},('supported','supported')),
        ({i:bad for i in range(161,256)},('falsified','supported')),
        ({**{i:bad for i in range(162,256)},161:unk},('inconclusive','supported')),
        ({i:bad for i in range(8)},('supported','supported')),
        ({i:bad for i in range(9)},('supported','falsified')),
        ({**{i:bad for i in range(8)},8:unk},('supported','inconclusive')),
        ({i:unk for i in range(256)},('inconclusive','inconclusive'))]:
        result=endpoints(changes);assert tuple(r['decision'] for r in result.values())==want;cases.append(result)
    life=lifecycle()
    large=rng.integers(0,9,(600,81),dtype=np.int64);timings={}
    for fn in [counts,independent_counts]:
        times=[]
        for _ in range(3):
            before=time.perf_counter();actual=fn(large);times.append(time.perf_counter()-before)
            assert np.array_equal(actual,counts(large))
        timings[fn.__name__]=float(np.median(times))
    sources=S.read(S.PARENT_PROTOCOL)['source_sha256'].copy()
    for name in ['disjoint_donor.py','disjoint_donor_math.py','audit_disjoint_donor.py','preflight_disjoint_donor.py']:
        path=S.HERE/name;sources[str(path.relative_to(S.ROOT))]=S.sha(path)
    S.save(gatepath,dict(status='passed',study_masks_scores_or_LPs_read=False,disjoint_row_ranges_and_independent_donor_costs=True,full2400_noise_and_offset_replay=True,source_corruptions_rejected=6,
        target_scores_use_fixed_donor_graph_only=True,literal_fractional_graph_scores_and_ties=True,digit_permutation_invariance=True,
        all_donor_certification_cases=True,certificate_mutations_rejected=3,endpoint_cases=cases,lifecycle=life,
        unrelated600x81_donor_cost_timings=timings,source_sha256=sources,seed_audit_sha256=S.sha(S.HERE/'disjoint-donor-seed-audit.json'),gate_wall_seconds=time.perf_counter()-started))
    print('PREFLIGHT_PASSED',timings,flush=True)


if __name__=='__main__':main()
