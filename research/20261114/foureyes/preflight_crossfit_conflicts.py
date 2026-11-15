"""Synthetic H4ax dependence, certificate, unresolved, seal and endpoint gates."""
from pathlib import Path
import tempfile,time
import numpy as np
import crossfit_conflicts as S
from crossfit_conflict_math import fold_costs,independent_fold_costs,crossfit_mask,independent_crossfit_mask
from conflict_mask_math import literal_scores
from corruption_censor_math import literal_score
from audit_crossfit_conflicts import aggregate
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
    for name,count in [('corruption-censor',256),('partial-mask-recall',1280),('conflict-mask',512),('purified-detector',512)]:
        p=S.read(S.HERE/f'{name}.json');base=p['solver']['random_seed_base']
        expanded.update(range(base,base+count));ranges[name+'-solver']=[base,base+count-1]
        if 'random_seed_base' in p:
            base=p['random_seed_base'];expanded.update(range(base,base+256));ranges[name+'-mask']=[base,base+255]
    thinning=S.read(S.HERE/'partial-mask-recall.json')['thinning']['seed_base']
    thinning_sequences=[[thinning,i] for i in range(256)]
    solver=set(range(design['solver_seed_base'],design['solver_seed_base']+768))
    assert len(solver)==768 and not solver&(values|expanded) and 0<=min(solver)<max(solver)<2**31
    S.save(S.HERE/'crossfit-conflicts-seed-audit.json',dict(status='passed',frozen_protocol_sha256=files,
        historical_explicit_seed_count=len(values),expanded_ancestor_ranges=ranges,H4au_actual_thinning_seed_sequences=thinning_sequences,
        solver_range=[min(solver),max(solver)],collisions=[],
        scope='Explicit frozen seed fields and expanded H4at/H4au/H4av/H4aw solver and declared random-mask seeds. Other historical formulas not exhaustive; no proof of PRNG independence.'))


def endpoints(changes):
    rows=[]
    for i in range(256):
        kind=changes.get(i,'success')
        rows.append(dict(index=i,baseline_exact=i<129,parent_detector_exact=i<130,classification=kind,family=S.FAMILIES[i%4],
            mask_constructed=kind!='unresolved',true_positive_locations=600,false_positive_locations=4800,false_negative_locations=4800,
            certified=kind!='unresolved',folds=[dict(certified=True),dict(certified=True)]))
    return aggregate(rows)['primary']


def lifecycle():
    oldout,oldprotocol=S.OUT,S.PROTOCOL
    with tempfile.TemporaryDirectory(prefix='h4ax-synthetic-') as tmp:
        try:
            S.OUT=Path(tmp)/'out';S.OUT.mkdir();S.PROTOCOL=Path(tmp)/'protocol.json';S.save(S.PROTOCOL,dict(source_sha256={}))
            rejected(S.final_worker,0);rejected(S.audit_worker,0);rejected(S.seal_phase,'fold')
            ph=S.sha(S.PROTOCOL)
            for i in range(256):
                folder=S.OUT/f'b{i:03d}';folder.mkdir()
                for k in range(2):
                    name=f'fold{k}';np.savez(folder/f'{name}.npz',fixture=np.zeros(1))
                    S.save(folder/f'{name}.json',dict(index=i,arm=name,protocol_sha256=ph,models_sha256=S.sha(folder/f'{name}.npz'),
                        fit_started_monotonic_ns=time.monotonic_ns(),fit_completed_monotonic_ns=time.monotonic_ns(),certificate=dict(certified=False)))
            S.seal_phase('fold');seal=S.check_phase_seal(S.OUT,'fold');rejected(S.seal_phase,'fold')
            file=S.OUT/'b000/fold0.npz';old=file.read_bytes();file.write_bytes(old+b'bad');rejected(S.check_phase_seal,S.OUT,'fold');file.write_bytes(old)
            # With unresolved folds, no noisy input or mask is available and fit_one must never run.
            oldfit=S.fit_one
            def forbidden(*args,**kwargs):raise AssertionError('Unresolved fold reached final solver')
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
    return dict(early_final_and_audit_rejected=True,missing_folds_rejected=True,mutated_fold_rejected=True,
        final_before_fold_seal_rejected=True,unresolved_folds_skip_all256_final_fits=True,mutated_final_rejected=True,repeated_seals_rejected=True)


def main():
    target=S.HERE/'crossfit-conflicts-preflight.json'
    assert not target.exists() and not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    started=time.perf_counter();seed_audit(S.read(S.HERE/'crossfit-conflicts-proposal.json'))
    rng=np.random.default_rng(982080);noisy=rng.integers(0,9,(20,18),dtype=np.int8)
    a=rng.integers(0,3,153,dtype=np.int64);b=rng.integers(0,3,153,dtype=np.int64)
    costs=fold_costs(noisy);independent=independent_fold_costs(noisy)
    assert all(np.array_equal(x,y) for x,y in zip(costs,independent))
    assert np.array_equal(costs[0]+costs[1],literal_score(noisy,np.zeros_like(noisy,bool))['cost'])
    scores,mask=crossfit_mask(noisy,a,b);si,mi=independent_crossfit_mask(noisy,a,b)
    assert np.array_equal(scores,si) and np.array_equal(mask,mi) and (mask.sum(1)==9).all()
    assert np.array_equal(scores[:10],literal_scores(noisy[:10],b)) and np.array_equal(scores[10:],literal_scores(noisy[10:],a))
    changed=rng.integers(0,3,153,dtype=np.int64);sc,mc=crossfit_mask(noisy,changed,b)
    assert np.array_equal(sc[:10],scores[:10]) and np.array_equal(mc[:10],mask[:10])
    sc,mc=crossfit_mask(noisy,a,changed);assert np.array_equal(sc[10:],scores[10:]) and np.array_equal(mc[10:],mask[10:])
    zero=np.zeros(153,np.int64);zs,zm=crossfit_mask(noisy,zero,zero)
    assert not zs.any() and zm[:,:9].all() and not zm[:,9:].any()
    permutation=rng.permutation(9);ps,pm=crossfit_mask(permutation[noisy],a,b)
    assert np.array_equal(ps,scores) and np.array_equal(pm,mask)
    for pair,expected in [([True,True],True),([False,True],False),([True,False],False),([False,False],False)]:
        assert S.fit_allowed([dict(certificate=dict(certified=v,integral=False)) for v in pair])==expected
    folder=S.ROOT/'artifacts/research/20261114/foureyes/crossfit_conflicts_preflight';folder.mkdir(parents=True,exist_ok=True)
    ii,jj=np.triu_indices(4,1);cost=np.ones(6,np.int64)
    x,y,meta=solve(4,1,ii,jj,cost,982081-2026111502,folder/'synthetic.log');cert,w=attempt_certificate(4,1,ii,jj,cost,x,y)
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
    # Independent reference optimization: vectorized per-pair row equality retains literal semantics.
    large=rng.integers(0,9,(600,81),dtype=np.int8);timings={}
    for fn in [fold_costs,independent_fold_costs]:
        times=[]
        for _ in range(3):
            before=time.perf_counter();actual=fn(large);times.append(time.perf_counter()-before)
            assert all(np.array_equal(x,y) for x,y in zip(actual,fold_costs(large)))
        timings[fn.__name__]=float(np.median(times))
    sources=S.read(S.PARENT_PROTOCOL)['source_sha256'].copy()
    for name in ['crossfit_conflicts.py','crossfit_conflict_math.py','audit_crossfit_conflicts.py','preflight_crossfit_conflicts.py']:
        path=S.HERE/name;sources[str(path.relative_to(S.ROOT))]=S.sha(path)
    S.save(target,dict(status='passed',study_masks_scores_or_LPs_read=False,complementary_fold_cost_identity=True,
        opposite_graph_only_dependence=True,literal_fractional_graph_scores_and_ties=True,digit_permutation_invariance=True,
        all_four_fold_certification_cases=True,certificate_mutations_rejected=3,endpoint_cases=cases,lifecycle=life,
        unrelated600x81_fold_cost_timings=timings,source_sha256=sources,seed_audit_sha256=S.sha(S.HERE/'crossfit-conflicts-seed-audit.json'),gate_wall_seconds=time.perf_counter()-started))
    print('PREFLIGHT_PASSED',timings,flush=True)


if __name__=='__main__':main()
