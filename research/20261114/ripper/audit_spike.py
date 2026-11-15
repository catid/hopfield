"""Fixed-spike acquisition hypothesis and independent new-cohort search audit."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity

OUT=ROOT/'artifacts/research/20261114/ripper/spike'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-spike.json'


def main():
    p=json.loads(PROTOCOL.read_text());manifest=json.loads((ROOT/p['inputs']).read_text())
    assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    assert sha(HERE/'spike_policy.py')==p['frozen_policy_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256']
    preflight_path=ROOT/p['policy_preflight_path'];assert sha(preflight_path)==p['policy_preflight_sha256']
    preflight=json.loads(preflight_path.read_text())
    assert preflight['status']=='validated' and preflight['spike_policy_sha256']==p['frozen_policy_sha256']
    assert preflight['original_policy_sha256']==p['original_policy_sha256']
    assert preflight['wrapper_sha256']==sha(HERE/'test_spike_policy.py')
    assert preflight['exact_rational_score_checks']==301 and preflight['initial_tie_checks']==64
    assert preflight['unchanged_control_checks']==128 and preflight['read_only_purity_and_determinism_checks']==64
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['input_protocol_sha256']==p['input_protocol_sha256']
    assert input_audit['wrapper_sha256']==sha(HERE/'audit_spike_inputs.py')
    assert input_audit['unique_source_rows']==input_audit['independent_csv_answers_verified']==9600
    assert input_audit['cross_bank_overlap']==input_audit['prior_bank_overlap']==0
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'acquisition_audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'spike_acquire.py') and r['policy_sha256']==sha(HERE/'spike_policy.py')
        assert a['wrapper_sha256']==sha(HERE/'audit_spike_acquisition.py')
        assert a['acquisition_receipt_sha256']==sha(d/'receipt.json') and a['status']=='validated'
        assert a['independent_masks_replayed']==900 and a['prefix_supports_verified']==3 and a['hidden_value_recount_checks']==6
        assert r['current_row_permutation_invariance_checks']==r['premature_reveal_rejections']==900
        audits.append(a);timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'selection_s':r['selection_s'],
            'reveal_s':r['reveal_s'],'independent_replay_s':a['elapsed_s']})
        e=next(e for e in manifest['inputs'] if e['seed']==seed)
        assert sha(ROOT/e['completion_path'])==e['completion_sha256']==r['source_completion_sha256']
        completion[seed]=np.load(ROOT/e['completion_path'])['solutions']
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];m=np.load(ROOT/model['path'])
            support=m['support'];assert support.dtype==bool and len(support)==3240
            assert model['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            assert np.array_equal(support,m['equal_count']==0)
            missing=int((peer&~support).sum());extra=int((~peer&support).sum());assert missing==0
            arm,n=model['arm'],model['n'];supports[arm,n,seed]=support
            structure.append({'seed':seed,'n':n,'arm':arm,'queried_digits':25*n,'extra_edges':extra,
                'missing_true_edges':missing,'retained_edges':int(support.sum()),'exact_graph':extra==0,
                'joint_observation_min':int(m['jointly_given_count'].min()),
                'joint_observation_max':int(m['jointly_given_count'].max()),
                'joint_observation_sd':float(m['jointly_given_count'].std()),
                'support_bytes_sha256':model['support_bytes_sha256']})
        for name in ['receipt.json','acquisition_audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    extra={arm:np.array([next(r['extra_edges'] for r in structure if (r['arm'],r['n'],r['seed'])==(arm,300,s)) for s in p['seeds']]) for arm in p['arms']}
    rng=np.random.default_rng(20261203);boot=[];zero=0
    for _ in range(5000):
        ix=rng.integers(0,32,32);den=extra['candidate'][ix].sum()
        if den==0:zero+=1
        else:boot.append(1-extra['spike'][ix].sum()/den)
    effect=float(1-extra['spike'].sum()/extra['candidate'].sum()) if extra['candidate'].sum() else None
    ci=[float(x) for x in np.quantile(boot,[.025,.975])] if zero==0 else None
    decision='inconclusive' if ci is None else 'supported_at_least_10percent_reduction' if ci[0]>=.10 else 'falsified_10percent_target' if ci[1]<.10 else 'inconclusive'
    primary={'decision':decision,'comparison':'spike versus unchanged candidate','relative_reduction':effect,'paired_bank_bootstrap_ci95':ci,'target':.10,
        'zero_denominator_bootstrap_draws':zero,'extra_edges_per_seed':{k:v.tolist() for k,v in extra.items()},
        'total_extra_edges':{k:int(v.sum()) for k,v in extra.items()},'seeds':32,'bootstrap_draws':5000,
        'scope':'Thirty-two fresh disjoint banks and one frozen tie stream per bank; earlier acquisition results are not pooled.'}
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue) and len(queue)==mapping['unique_searches']
    assert len(mapping['rows'])==96 and mapping['wrapper_sha256']==sha(HERE/'spike_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'spike_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[180000,190000] and r['test_rows']==10000
        assert r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==125 and r['hidden_target_state_comparisons']==110
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        support=supports[r['arm'],300,r['seed']]
        assert r['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        model=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu')
        assert torch.equal(model['W'],torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9)) and not model['B'].any()
        z=np.load(d/'boards.npz');b=z['boards'];st=z['stopped'];moves=z['moves']
        assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
        exact=(b==target).all(1);valid=validity(b,target,free)
        assert np.array_equal(exact,valid) and np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid'])
        assert int(exact.sum())==r['solved'] and int((~st).sum())==r['budget_failures']
        assert int((st&~exact).sum())==r['wrong_stops']==0 and np.all((moves>0)&(moves<=262144))
        per=json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(per['solved'],exact) and np.array_equal(per['stopped'],st) and np.array_equal(per['flops'],moves*r['fl_move']*4)
        energy=(target[:,i[support]]==target[:,j[support]]).sum(1);assert np.all(~st|(energy==0))
        oracle.append({'seed':r['seed'],'arm':r['arm'],'zero_energy_targets':int((energy==0).sum()),
            'mean_target_energy':float(energy.mean()),'correct_certified_stops':int((st&exact).sum()),
            'correct_uncertified_returns':int((~st&exact).sum())})
        outputs[job['job_id']]=exact
        rows.append({k:r[k] for k in ['job_id','n','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','edges','D']})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    arrays={}
    for m in mapping['rows']:
        r=next(r for r in rows if r['job_id']==m['canonical_job']);assert r['seed']==m['seed']
        assert np.array_equal(supports[r['arm'],300,r['seed']],supports[m['arm'],300,m['seed']])
        arrays[m['arm'],m['seed']]=outputs[m['canonical_job']]
    grouped=[]
    for arm in p['arms']:
        solved=[int(arrays[arm,s].sum()) for s in p['seeds']]
        grouped.append({'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),
            'wrong_stops':0,'seeds':32,'n_per_seed':10000})
    contrasts=[]
    for arm,reference_arm in [('candidate','uniform'),('spike','uniform'),('spike','candidate')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        rng=np.random.default_rng(20261204)
        boot=[100*diff[rng.integers(0,32,32)][:,rng.integers(0,10000,10000)].mean() for _ in range(5000)]
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    report={'status':'validated','utc':now(),'primary':primary,'structure':structure,'grouped':grouped,
        'secondary_contrasts':contrasts,'rows':rows,'oracle_diagnostic':oracle,'saved_boards_verified':len(queue)*10000,
        'unique_searches':len(queue),'requested_searches':96,'search_reuse_mapping':mapping,
        'acquisition_audits':audits,'independent_masks_replayed':28800,'prefix_supports_verified':96,
        'hidden_value_recount_checks':192,'timings':timings,'search_wall_sum_s':sum(r['wall_s'] for r in rows),
        'four_gpu_queue_wall_envelope_s':4*span,'training_test_completeboard_overlap':overlaps,
        'artifact_sha256':hashes,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,
        'prior_result_not_pooled':True,'frozen_policy_sha256':p['frozen_policy_sha256'],
        'independent_input_audit':input_audit,'input_audit_sha256':sha(OUT/'input_audit.json'),
        'standalone_policy_preflight':preflight,'policy_preflight_sha256':sha(preflight_path),
        'original_control_policy_sha256':p['original_policy_sha256'],
        'board_validity_helper_sha256':sha(HERE/'audit_clues.py'),'test_data_sha256':sha(datafile),
        'limitations':p['limitations']+[p['uncertainty_limit'],
            'The graph-coverage target is the registered primary. Secondary accuracy intervals are descriptive and unadjusted for multiple comparisons.',
            'Correct returned best boards may have positive energy; zero-energy oracle fractions bound correct certified stops only.']}
    save(OUT/'audit.json',report);save(PUBLIC/'spike.json',report)
    for name,rs in [('spike_structure.csv',structure),('spike_per_seed.csv',rows),('spike_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['spike_mixture_acquisition']=report;save(PUBLIC/'summary.json',summary)
    lines=['# H1u: fixed spike-mixture query scoring','',
        f'The frozen 10% improvement-over-candidate target is **{decision}**. Spike scoring reduces total N300 false extra edges relative to the unchanged candidate policy by {100*effect:.3f}%, with a paired-bank 95% bootstrap interval [{100*ci[0]:.3f}%, {100*ci[1]:.3f}%]. This uncertainty uses only 32 fresh disjoint banks and one frozen tie stream per bank.','',
        '| Policy | Total N300 extra edges (32 banks) | Mean test solved (%) | Wrong stops |',
        '|---|---|---:|---:|','| BP AdamW: outside this acquisition study; not run | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {int(extra[g["arm"]].sum())} | {g["mean_accuracy_pct"]:.4f} | 0 |')
    lines+=['','Every policy queries exactly 25 cells per incoming board: 7,500 revealed digits at N300 in each bank. The stored complete boards form a simulated label oracle. The policy receives only past joint/equal observation counts and a fixed tie order; it cannot access the current board, source identity, geometry or oracle handle.','',
        'Uniform and inverse-exposure candidate selection use the unchanged original policy. The new heuristic assigns prior probability one half to an impossible pair, otherwise a fixed equality probability one ninth. After m observed unequal co-queries with no equality, its predictive score is (1/9)*(8/9)^m/(1+(8/9)^m); an observed equality makes the score zero. Greedy selection uses 18 times this score, preserving real-arithmetic rankings while making initial weights exactly one for correct FP64 tie handling. Zero after an observed equality means there is no value in reobserving a pair already excluded from the count graph; it is not a prediction that future equality has probability zero. The prior and rate were never tuned. This heuristic is not a calibrated posterior guarantee for Sudoku.','',
        'All 28,800 mask decisions were independently reconstructed from past queried archives. Masks were fixed before current labels were revealed, premature reveals were rejected, and hidden current-row permutations left those masks unchanged. All 96 prefix supports and 192 hidden-value substitution recounts passed. Geometry appears only in the final audit.','',
        'Fresh evaluation uses full-all rows [180000,190000), 32 paired seeds, four chains and 262,144 moves per chain. All '+f'{len(queue)*10000:,}'+' returned boards were independently checked; there were no wrong stops. The count energy uses literal unit1, zero biases, tau0 and ebar0 without hidden-label calibration.','']
    for c in contrasts:lines.append(f'Secondary {c["arm"]} minus {c["reference"]} accuracy: {c["effect_pp"]:.4f} pp, descriptive 95% interval {c["descriptive_ci95_pp"]}.')
    lines+=['','The earlier positive candidate-policy confirmation remains separate and is not pooled into this improvement test. Secondary accuracy comparisons are descriptive. The acquisition policies adapt to prior observations, so the independent-uniform-mask theorem does not apply directly. This experiment does not establish a universally optimal acquisition policy.','',
        '[Protocol](protocol-spike.json) · [Restricted policy](spike_policy.py) · [Acquisition harness](spike_acquire.py) · [Independent replay](audit_spike_acquisition.py) · [Search](spike_search.py) · [Final audit](audit_spike.py) · [Public results](../../../results/research/ripper/spike.json)','']
    (HERE/'REPORT-spike.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'secondary':contrasts,'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
