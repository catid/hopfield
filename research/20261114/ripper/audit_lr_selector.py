"""Independent noisy acquisition, saved-board and paired accuracy audit."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity
from lr_selector_common import load_protocol
from bootstrap_two_way import resample_means

OUT=ROOT/'artifacts/research/20261114/ripper/lr_selector'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-lr-selector.json'


def main():
    p=load_protocol();manifest=json.loads((ROOT/p['inputs']).read_text())
    registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'lr_selector_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'lr_selector_acquire.py') and registration['search_sha256']==sha(HERE/'lr_selector_search.py')
    assert sha(HERE/'lr_selector_policy.py')==p['policy_sha256']==registration['policy_sha256']
    assert sha(HERE/'LR_EVIDENCE_SELECTOR.md')==p['proof_sha256']==registration['proof_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256'] and sha(HERE/'exponential_policy.py')==p['original_exponential_sha256']
    assert sha(ROOT/p['policy_preflight_path'])==p['policy_preflight_sha256'];preflight=json.loads((ROOT/p['policy_preflight_path']).read_text())
    assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    assert preflight['independent_mask_comparisons']==1536 and preflight['readonly_history_cases']==384
    assert sha(ROOT/p['replay_optimization_validation_path'])==p['replay_optimization_validation_sha256']
    assert sha(HERE/'fast_reference_mask.py')==p['fast_reference_mask_sha256']
    assert p['original_noisy_policy_sha256']==sha(HERE/'noisy_policy.py')
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['protocol_sha256']==sha(PROTOCOL) and input_audit['wrapper_sha256']==sha(HERE/'audit_lr_selector_inputs.py')
    assert input_audit['independent_csv_answers_verified']==100721 and input_audit['calibration_training_overlap']==0
    assert input_audit['within_bank_overlap']==input_audit['prior_bank_overlap']==0
    bootstrap_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json').read_text())
    assert bootstrap_validation['status']=='validated' and bootstrap_validation['helper_sha256']==sha(HERE/'bootstrap_two_way.py')
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={};calibrations=[];calibration_sources={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'lr_selector_acquire.py') and a['wrapper_sha256']==sha(HERE/'audit_lr_selector_acquisition.py')
        assert r['policy_sha256']==p['policy_sha256'] and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['status']=='validated' and a['independent_masks_replayed']==8880 and a['log_lr_values_verified']==28771200
        assert a['independent_noise_rows_verified']==2220 and a['supports_verified']==4 and a['first_crossing_paths_verified']==12960
        assert r['masks_fixed_before_all_reveals']==8880 and r['read_only_history_fields']==['joint','equal','excluded']
        assert a['lr_weight_matrices_verified']==4440 and a['potential_noise_cached_once'] and a['cached_noise_readonly_and_unchanged']
        assert a['fast_mask_helper_sha256']==p['fast_reference_mask_sha256'];audits.append(a)
        timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'independent_replay_s':a['elapsed_s']})
        e=next(e for e in manifest['inputs'] if e['seed']==seed);assert sha(ROOT/e['training_path'])==e['training_sha256']==r['source_sha256']
        completion[seed]=np.load(ROOT/e['training_path'])['solutions']
        calibration_sources[seed]=np.load(ROOT/e['calibration_path'])['solutions']
        cal=json.loads((d/'calibration.json').read_text());assert sha(d/'calibration.json')==r['calibration_sha256']
        assert a['decimal_threshold_checks']==8884 and a['calibration_audit']['shared_calibration_boards']==1000
        for arm,fit in cal['fits'].items():
            checked=next(x for x in a['calibration_audit']['fits'] if x['arm']==arm)
            assert checked['disagreements']==fit['disagreements'] and checked['pairs']==p['arm_calibration_pairs'][arm]
            calibrations.append(dict(seed=seed,arm=arm,**fit))
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];m=np.load(ROOT/model['path']);support=m['support']
            assert support.dtype==bool and len(support)==3240
            arm,n=model['arm'],model['n'];supports[arm,n,seed]=support
            assert np.array_equal(support,m['first_crossing']<0)
            missing=int((peer&~support).sum());extra=int((~peer&support).sum())
            checked=next(x for x in a['rows'] if x['arm']==arm and x['n']==n)
            assert checked['missing_true_edges']==missing and checked['extra_edges']==extra
            assert checked['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            structure.append({'seed':seed,'n':n,'arm':arm,'queried_noisy_digits':25*n,'calibration_queries':p['label_costs'][arm]['calibration'],'total_queries':p['label_costs'][arm]['total'],'extra_edges':extra,
                'missing_true_edges':missing,'retained_edges':int(support.sum()),'exact_graph':extra==missing==0,
                'support_bytes_sha256':checked['support_bytes_sha256']})
        for name in ['receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue) and len(queue)==mapping['unique_searches']
    assert len(mapping['rows'])==128 and mapping['wrapper_sha256']==sha(HERE/'lr_selector_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'lr_selector_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[300000,310000] and r['test_rows']==10000
        assert r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==125 and r['hidden_target_state_comparisons']==110
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        support=supports[r['arm'],r['n'],r['seed']]
        assert r['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        model=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu')
        assert torch.equal(model['W'],torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9)) and not model['B'].any()
        z=np.load(d/'boards.npz');b=z['boards'];st=z['stopped'];moves=z['moves']
        assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
        exact=(b==target).all(1);valid=validity(b,target,free)
        assert np.array_equal(exact,valid) and np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid'])
        assert int(exact.sum())==r['solved'] and int((~st).sum())==r['budget_failures']
        assert int((st&~exact).sum())==r['wrong_stops'] and np.all((moves>0)&(moves<=262144))
        per=json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(per['solved'],exact) and np.array_equal(per['stopped'],st) and np.array_equal(per['flops'],moves*r['fl_move']*4)
        energy=(target[:,i[support]]==target[:,j[support]]).sum(1)
        returned_energy=(b[:,i[support]]==b[:,j[support]]).sum(1)
        assert np.all(~st|(returned_energy==0)) and np.all(~(st&exact)|(energy==0))
        if np.all(support[peer]):assert not (st&~valid).any()
        oracle.append({'seed':r['seed'],'arm':r['arm'],'zero_energy_targets':int((energy==0).sum()),
            'mean_target_energy':float(energy.mean()),'correct_certified_stops':int((st&exact).sum()),
            'correct_uncertified_returns':int((~st&exact).sum())})
        outputs[job['job_id']]=exact
        rows.append({k:r[k] for k in ['job_id','n','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','edges','D']})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    arrays={};wrong={};invalid_capped={}
    for m in mapping['rows']:
        r=next(r for r in rows if r['job_id']==m['canonical_job']);assert r['seed']==m['seed']
        assert np.array_equal(supports[r['arm'],r['n'],r['seed']],supports[m['arm'],m['n'],m['seed']])
        arrays[m['arm'],m['seed']]=outputs[m['canonical_job']]
        wrong[m['arm'],m['seed']]=r['wrong_stops']
        diagnostic=next(o for o in oracle if o['arm']==r['arm'] and o['seed']==r['seed'])
        invalid_capped[m['arm'],m['seed']]=r['budget_failures']-diagnostic['correct_uncertified_returns']
    grouped=[]
    for arm in p['arms']:
        solved=[int(arrays[arm,s].sum()) for s in p['seeds']]
        grouped.append({'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),
            'invalid_capped_returns':sum(invalid_capped[arm,s] for s in p['seeds']),'invalid_capped_per_seed':[invalid_capped[arm,s] for s in p['seeds']],'wrong_stops':sum(wrong[arm,s] for s in p['seeds']),'wrong_stops_per_seed':[wrong[arm,s] for s in p['seeds']],'seeds':32,'n_per_seed':10000})
    contrasts=[]
    for arm,reference_arm in [('lr_learned','exposure_learned'),('lr_exact','exposure_exact'),('lr_learned','lr_exact'),('exposure_learned','exposure_exact')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        boot=resample_means(diff,20341000,5000)
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    c=contrasts[0];lo_ci,hi_ci=c['descriptive_ci95_pp']
    protected_wrong=sum(g['wrong_stops'] for g in grouped if g['arm'] in ('exposure_learned','lr_learned'))
    accuracy_decision='supported_10pp_gain' if lo_ci>=10 else 'falsified_10pp_gain' if hi_ci<10 else 'inconclusive'
    learned_matrix=np.array([arrays['lr_learned',seed] for seed in p['seeds']],dtype=float)
    absolute_boot=resample_means(learned_matrix,20341000,5000)
    absolute_lo,absolute_hi=[float(x) for x in np.quantile(absolute_boot,[.025,.975])]
    absolute_mean=float(100*learned_matrix.mean())
    floor_decision='supported_70percent_floor' if absolute_lo>=70 else 'falsified_70percent_floor' if absolute_hi<70 else 'inconclusive'
    if protected_wrong:decision='falsified_wrong_stop_gate'
    elif accuracy_decision.startswith('falsified') or floor_decision.startswith('falsified'):decision='falsified_conjunction'
    elif accuracy_decision.startswith('supported') and floor_decision.startswith('supported'):decision='supported_conjunction'
    else:decision='inconclusive_conjunction'
    primary={'decision':decision,'accuracy_decision':accuracy_decision,'gain_decision':accuracy_decision,'comparison':'lr_learned minus exposure_learned','effect_pp':c['effect_pp'],
        'paired_two_way_bootstrap_ci95_pp':[lo_ci,hi_ci],'minimum_gain_pp':10,'mean_learned_accuracy_pct':absolute_mean,
        'learned_accuracy_ci95_pct':[absolute_lo,absolute_hi],'usefulness_floor_pct':70,'usefulness_decision':floor_decision,
        'protected_wrong_stops':protected_wrong,'zero_protected_wrong_stops_gate':protected_wrong==0,
        'bank_seeds':32,'distinct_test_inputs':10000,'bootstrap_draws':5000,'bootstrap_seed':20341000,
        'interval_scope':'Each interval has marginal nominal95percentcoverage; no simultaneous-coverage claim.', 'scope':p['uncertainty_limit']}
    structural_groups=[]
    for arm in p['arms']:
        rs=[r for r in structure if r['arm']==arm]
        structural_groups.append({'arm':arm,'banks':32,'exact_graphs':sum(r['exact_graph'] for r in rs),
            'all_true_edges_retained':sum(r['missing_true_edges']==0 for r in rs),
            'total_missing_true_edges':sum(r['missing_true_edges'] for r in rs),'total_extra_edges':sum(r['extra_edges'] for r in rs)})
    report={'status':'validated','utc':now(),'primary':primary,'structure':structure,'structural_groups':structural_groups,
        'calibration_results':calibrations,'label_costs':p['label_costs'],'calibration_coverage_banks':{arm:sum(c['postfit_covers_actual_eta'] for c in calibrations if c['arm']==arm) for arm in ['exposure_learned','lr_learned']},'calibration_fallback_banks':{arm:sum(c['no_pruning_fallback'] for c in calibrations if c['arm']==arm) for arm in ['exposure_learned','lr_learned']},'grouped':grouped,'secondary_contrasts':contrasts[1:],'rows':rows,'oracle_diagnostic':oracle,
        'saved_boards_verified':len(queue)*10000,'unique_searches':len(queue),'requested_searches':128,'search_reuse_mapping':mapping,
        'acquisition_audits':audits,'independent_masks_replayed':284160,'log_lr_values_verified':920678400,
        'unique_calibration_fits':32,'lr_weight_matrices_verified':142080,'replay_optimization_validation':json.loads((ROOT/p['replay_optimization_validation_path']).read_text()),'prefix_supports_verified':128,'first_crossing_paths_verified':414720,'timings':timings,
        'search_wall_sum_s':sum(r['wall_s'] for r in rows),'four_gpu_queue_wall_envelope_s':4*span,
        'training_test_completeboard_overlap':overlaps,'calibration_test_completeboard_overlap':[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in calibration_sources.items()],'independent_input_audit':input_audit,
        'input_registration':registration,'standalone_policy_preflight':preflight,'audit_bootstrap_validation':bootstrap_validation,
        'artifact_sha256':hashes,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,
        'proof_sha256':p['proof_sha256'],'pilot_not_pooled':True,'test_data_sha256':sha(datafile),
        'limitations':p['limitations']+[p['uncertainty_limit'],'Correct zero-energy target fractions bound correct certified stops only,not all returned-best-board accuracy.']}
    save(OUT/'audit.json',report);save(PUBLIC/'lr_selector.json',report)
    for name,rs in [('lr_selector_structure.csv',structure),('lr_selector_per_seed.csv',rows),('lr_selector_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    lines=['# H1ae: LR evidence for acquisition at fixed cost','',f'The frozen conjunction is **{decision}**. Its endpoints are reported separately:',
        '',f'- Gain of at least 10 pp: **{accuracy_decision}**. LR/learned minus exposure/learned accuracy is {primary["effect_pp"]:.4f} pp; marginal paired two-way 95% interval [{lo_ci:.4f}, {hi_ci:.4f}] pp.',
        f'- 70% usefulness floor: **{floor_decision}**. LR/learned accuracy is {absolute_mean:.4f}%; marginal two-way 95% interval [{absolute_lo:.4f}, {absolute_hi:.4f}]%.',
        f'- Zero wrong certified stops in both learned arms: **{"passed" if protected_wrong==0 else "failed"}**, with {protected_wrong} wrong stops.',
        '', 'Both intervals have marginal nominal 95% coverage, without a simultaneous-coverage claim. Exact-bound controls and other contrasts are descriptive.', '',
        '| Selector / bound | Mean solved (%) | Calibration / training queries | Total queries | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing edges | Extra edges |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — | — | — | — | — |']
    labels={'exposure_exact':'Exposure / exact','lr_exact':'LR / exact','exposure_learned':'Exposure / learned','lr_learned':'LR / learned'}
    for g in grouped:
        t=next(t for t in structural_groups if t['arm']==g['arm']);cost=p['label_costs'][g['arm']]
        lines.append(f'| {labels[g["arm"]]} | {g["mean_accuracy_pct"]:.4f} | {cost["calibration"]:,} / {cost["training"]:,} | {cost["total"]:,} | {g["wrong_stops"]} | {g["invalid_capped_returns"]} | {t["exact_graphs"]} | {t["total_missing_true_edges"]} | {t["total_extra_edges"]} |')
    bounds=[c['eta_upper']*100 for c in calibrations if c['arm']=='lr_learned']
    lines+=['',f'The 32 unique learned bounds span {min(bounds):.4f}%–{max(bounds):.4f}%, covering the actual 5% rate in {report["calibration_coverage_banks"]["lr_learned"]}/32 banks, with {report["calibration_fallback_banks"]["lr_learned"]} no-pruning fallbacks. Each bank’s one calibration result is reused across selectors. Every outcome is retained; repeated reporting does not create 64 independent estimates.','',
        'Every arm uses actual independent replacement noise 5%, N2220 and 25 training queries per board. Learned arms pay 2,000 additional M1000 calibration queries, for 57,500 total; exact controls cost 55,500. The selector comparison within each calibration type has equal query cost. No training or calibration budget, q1, crossing law, temperature or search cap was tuned.','',
        'For unexcluded pairs, the candidate weights are exp(min(logLR, log648000)/2), with zero diagonal and excluded weights. Its greedy 25-cell choice and tie permutation are unchanged. The policy sees only past queried joint/equality counts, exclusions and frozen calibration multipliers; read-only state hashes and event order verify all current masks finalize before current noise is generated or revealed. Capping before exponentiation prevents overflow; underflow to zero is harmless.','',
        'The predictable-mask supermartingale proof applies despite using past LR evidence for selection. On valid calibration, edge-deletion risk is at most .005; calibration failure at most .005 gives at most .01 per learned model. Shared calibration does not create a joint 1% guarantee across selectors or banks. The proof requires the declared independent replacement-noise family and does not guarantee gain, usefulness, graph recovery or valid capped returns.','',
        'All settings, endpoints and streams froze before the old-board pilot and fresh outcomes. The 32 fresh banks contain 31,624 distinct calibration boards among 32,000 instances and 69,097 distinct training boards among 71,040 instances, excluding all 1,326,762 prior Ripper boards. Source roles are globally disjoint within this study. All 100,721 distinct source boards were independently matched against the CSV.','',
        'Before inference, independent cached read-only replay verified 284,160 masks and ties, 142,080 LR weight matrices, 920,678,400 log-LR values, all noise, 128 supports and 414,720 first-crossing paths. Exact rational crossings remain separate from floating selector weights. The six-bank pilot is separate and not pooled.','',
        f'All 32 bank/search seeds share 10,000 full-all inputs [300000,310000). Exact support/settings/seed/input deduplication reduces 128 requested cells to {len(queue)} searches; all {len(queue)*10000:,} returned boards passed independent audit. The four-GPU queue envelope is {4*span/60:.2f} GPU-min. Four chains, 262,144 moves per chain and unit1/B0/tau0/ebar0 remain fixed. Wrong certified stops and invalid capped returns are separate.','',
        'Earlier H1ac and H1ad results are unchanged and not pooled. No weight, gain threshold, usefulness floor or sample count was changed in response to outcomes.','',
        '[Protocol](protocol-lr-selector.json) · [Design and proof](LR_EVIDENCE_SELECTOR.md) · [Policy](lr_selector_policy.py) · [Acquisition](lr_selector_acquire.py) · [Replay](audit_lr_selector_acquisition.py) · [Results](../../../results/research/ripper/lr_selector.json)','']
    (HERE/'REPORT-lr-selector.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'structure':structural_groups,'coverage':report['calibration_coverage_banks'],'fallback':report['calibration_fallback_banks'],'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
