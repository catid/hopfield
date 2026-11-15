"""Independent noisy acquisition, saved-board and paired accuracy audit."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity
from calibrated_common import load_protocol
from bootstrap_two_way import resample_means

OUT=ROOT/'artifacts/research/20261114/ripper/calibrated'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-calibrated.json'


def main():
    p=load_protocol();manifest=json.loads((ROOT/p['inputs']).read_text())
    registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'calibrated_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'calibrated_acquire.py') and registration['search_sha256']==sha(HERE/'calibrated_search.py')
    assert sha(HERE/'calibrated_policy.py')==p['policy_sha256']==registration['policy_sha256']
    assert sha(HERE/'REPEATED_QUERY_CALIBRATION.md')==p['proof_sha256']==registration['proof_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256'] and sha(HERE/'exponential_policy.py')==p['original_exponential_sha256']
    assert sha(ROOT/p['policy_preflight_path'])==p['policy_preflight_sha256'];preflight=json.loads((ROOT/p['policy_preflight_path']).read_text())
    assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    assert preflight['unchanged_readonly_mask_checks']==192
    assert p['original_noisy_policy_sha256']==sha(HERE/'noisy_policy.py')
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['protocol_sha256']==sha(PROTOCOL) and input_audit['wrapper_sha256']==sha(HERE/'audit_calibrated_inputs.py')
    assert input_audit['independent_csv_answers_verified']==340537 and input_audit['calibration_training_overlap']==0
    assert input_audit['within_bank_overlap']==input_audit['prior_bank_overlap']==0
    bootstrap_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json').read_text())
    assert bootstrap_validation['status']=='validated' and bootstrap_validation['helper_sha256']==sha(HERE/'bootstrap_two_way.py')
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={};calibrations=[];calibration_sources={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'calibrated_acquire.py') and a['wrapper_sha256']==sha(HERE/'audit_calibrated_acquisition.py')
        assert r['policy_sha256']==p['policy_sha256'] and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['status']=='validated' and a['independent_masks_replayed']==4500 and a['log_lr_values_verified']==14580000
        assert a['independent_noise_rows_verified']==1500 and a['supports_verified']==3 and a['first_crossing_paths_verified']==9720
        assert r['masks_fixed_before_all_reveals']==4500;audits.append(a)
        timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'independent_replay_s':a['elapsed_s']})
        e=next(e for e in manifest['inputs'] if e['seed']==seed);assert sha(ROOT/e['training_path'])==e['training_sha256']==r['source_sha256']
        completion[seed]=np.load(ROOT/e['training_path'])['solutions']
        calibration_sources[seed]=np.load(ROOT/e['calibration_path'])['solutions']
        cal=json.loads((d/'calibration.json').read_text());assert sha(d/'calibration.json')==r['calibration_sha256']
        assert a['decimal_threshold_checks']==4503 and a['calibration_audit']['pairs']==10000
        assert a['calibration_audit']['disagreements']==cal['disagreements']
        calibrations.append(dict(seed=seed,**cal))
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];m=np.load(ROOT/model['path']);support=m['support']
            assert support.dtype==bool and len(support)==3240
            arm,n=model['arm'],model['n'];supports[arm,n,seed]=support
            assert np.array_equal(support,m['first_crossing']<0)
            missing=int((peer&~support).sum());extra=int((~peer&support).sum())
            checked=next(x for x in a['rows'] if x['arm']==arm and x['n']==n)
            assert checked['missing_true_edges']==missing and checked['extra_edges']==extra
            assert checked['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            structure.append({'seed':seed,'n':n,'arm':arm,'queried_noisy_digits':25*n,'calibration_queries':20000 if arm=='learned' else 0,'total_queries':57500 if arm=='learned' else 37500,'extra_edges':extra,
                'missing_true_edges':missing,'retained_edges':int(support.sum()),'exact_graph':extra==missing==0,
                'support_bytes_sha256':checked['support_bytes_sha256']})
        for name in ['receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue) and len(queue)==mapping['unique_searches']
    assert len(mapping['rows'])==96 and mapping['wrapper_sha256']==sha(HERE/'calibrated_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'calibrated_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[250000,260000] and r['test_rows']==10000
        assert r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==125 and r['hidden_target_state_comparisons']==110
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        support=supports[r['arm'],1500,r['seed']]
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
        assert np.array_equal(supports[r['arm'],1500,r['seed']],supports[m['arm'],1500,m['seed']])
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
    for arm,reference_arm in [('learned','exact'),('fixed_upper','exact'),('learned','fixed_upper')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        boot=resample_means(diff,20283000,5000)
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    c=contrasts[0];lo_ci,hi_ci=c['descriptive_ci95_pp']
    protected_wrong=sum(g['wrong_stops'] for g in grouped if g['arm'] in ('exact','learned'))
    accuracy_decision='supported_5pp_noninferiority' if lo_ci>=-5 else 'falsified_5pp_noninferiority' if hi_ci<-5 else 'inconclusive'
    decision='falsified_protected_wrong_stop_gate' if protected_wrong else accuracy_decision
    primary={'decision':decision,'accuracy_decision':accuracy_decision,'comparison':'learned minus exact','effect_pp':c['effect_pp'],
        'paired_two_way_bootstrap_ci95_pp':[lo_ci,hi_ci],'noninferiority_margin_pp':5,
        'protected_wrong_stops':protected_wrong,'zero_protected_wrong_stops_gate':protected_wrong==0,
        'bank_seeds':32,'distinct_test_inputs':10000,'bootstrap_draws':5000,'bootstrap_seed':20283000,'scope':p['uncertainty_limit']}
    structural_groups=[]
    for arm in p['arms']:
        rs=[r for r in structure if r['arm']==arm]
        structural_groups.append({'arm':arm,'banks':32,'exact_graphs':sum(r['exact_graph'] for r in rs),
            'all_true_edges_retained':sum(r['missing_true_edges']==0 for r in rs),
            'total_missing_true_edges':sum(r['missing_true_edges'] for r in rs),'total_extra_edges':sum(r['extra_edges'] for r in rs)})
    report={'status':'validated','utc':now(),'primary':primary,'structure':structure,'structural_groups':structural_groups,
        'calibration_results':calibrations,'label_costs':p['label_costs'],'calibration_coverage_banks':sum(c['postfit_covers_actual_eta'] for c in calibrations),'calibration_fallback_banks':sum(c['no_pruning_fallback'] for c in calibrations),'grouped':grouped,'secondary_contrasts':contrasts[1:],'rows':rows,'oracle_diagnostic':oracle,
        'saved_boards_verified':len(queue)*10000,'unique_searches':len(queue),'requested_searches':96,'search_reuse_mapping':mapping,
        'acquisition_audits':audits,'independent_masks_replayed':144000,'log_lr_values_verified':466560000,
        'prefix_supports_verified':96,'first_crossing_paths_verified':311040,'timings':timings,
        'search_wall_sum_s':sum(r['wall_s'] for r in rows),'four_gpu_queue_wall_envelope_s':4*span,
        'training_test_completeboard_overlap':overlaps,'calibration_test_completeboard_overlap':[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in calibration_sources.items()],'independent_input_audit':input_audit,
        'input_registration':registration,'standalone_policy_preflight':preflight,'audit_bootstrap_validation':bootstrap_validation,
        'artifact_sha256':hashes,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,
        'proof_sha256':p['proof_sha256'],'pilot_not_pooled':True,'test_data_sha256':sha(datafile),
        'limitations':p['limitations']+[p['uncertainty_limit'],'Correct zero-energy target fractions bound correct certified stops only,not all returned-best-board accuracy.']}
    save(OUT/'audit.json',report);save(PUBLIC/'calibrated.json',report)
    for name,rs in [('calibrated_structure.csv',structure),('calibrated_per_seed.csv',rows),('calibrated_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    lines=['# H1aa: repeated-query noise calibration','',
        f'The frozen 5 pp noninferiority plus zero primary-arm wrong-stop target is **{decision}**. Learned minus exact-known-noise accuracy is {primary["effect_pp"]:.4f} pp, with paired two-way 95% bootstrap interval [{lo_ci:.4f}, {hi_ci:.4f}] pp. Support requires a lower bound of at least -5 pp and zero wrong certified stops in both primary arms.','',
        '| Method | Mean solved (%) | Total queries per bank | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing true edges | Extra edges |',
        '|---|---:|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — | — | — | — |']
    labels={'exact':'Exact known 1% noise','learned':'Repeated-query learned bound','fixed_upper':'Fixed 2% upper bound'}
    for g in grouped:
        t=next(t for t in structural_groups if t['arm']==g['arm'])
        lines.append(f'| {labels[g["arm"]]} | {g["mean_accuracy_pct"]:.4f} | {p["label_costs"][g["arm"]]["total"]:,} | {g["wrong_stops"]} | {g["invalid_capped_returns"]} | {t["exact_graphs"]} | {t["total_missing_true_edges"]} | {t["total_extra_edges"]} |')
    bounds=[c['eta_upper']*100 for c in calibrations]
    lines+=['',f'The 32 frozen calibration outcomes give bounds from {min(bounds):.4f}% to {max(bounds):.4f}%. Post-fit coverage of the actual 1% rate is {report["calibration_coverage_banks"]}/32; no-pruning fallback count is {report["calibration_fallback_banks"]}. These diagnostics never select, truncate or change a model. Every bank uses 10,000 repeated-query pairs and 1,500 fresh training boards, with 25 noisy queries per training board. The learned arm pays 20,000 calibration queries plus 37,500 training queries; the known-parameter controls pay 37,500.','',
        'The estimator receives only 10,000 disagreement indicators. A one-sided 99.5% Clopper–Pearson disagreement bound is inverted on the monotone independent-replacement-noise branch and rounded outward on a 1e-6 rational grid. Exact integer binomial tails certify the endpoint; an independent 80-digit backwards sum verifies the endpoint and rejects the previous grid point. If the bound reaches 8/9, every pair is retained.','',
        'Calibration error budget .005 plus conditional edge-deletion budget .005 gives at most .01 true-edge deletion risk per learned-bound model under the declared independent noise family, fixed across calibration and training. Both known-parameter controls use the same .005 edge budget and LR boundary 648000. This protects true edges; it does not guarantee few extra edges, exact graph recovery, solve accuracy, or arbitrary capped returns. It is not a single global 1% event across banks or arms.','',
        'The source split and every stream were frozen before the implementation-only pilot. Fresh calibration and training pools are globally disjoint within this study. The 320,000 calibration instances contain 293,179 distinct boards; the 48,000 training instances contain 47,358 distinct boards. All 356,066 previous Ripper boards are excluded. Cross-bank reuse is reported, not treated as new distinct inputs. Source reconstruction checked all 340,537 distinct boards against the CSV with an independent parser.','',
        'All 144,000 adaptive masks, 466,560,000 log-LR values, training and calibration noise draws, 96 supports, and 311,040 first-crossing paths passed independent replay before inference. All three masks are finalized before any current training label is revealed. A separate six-bank pilot passed the same audits and is not pooled.','',
        f'Inference uses the same 10,000 full-all inputs [250000,260000) across 32 paired bank/search seeds. Exact model/settings/seed/input deduplication reduces 96 requested cells to {len(queue)} searches; all {len(queue)*10000:,} returned boards were independently audited. The four-GPU queue envelope is {4*span/60:.2f} GPU-minutes. Four chains, 262,144 moves per chain, unit1, B0, tau0, ebar0 and search temperatures remain fixed. Wrong certified stops and invalid capped returns are distinct outcomes.','',
        'The fixed-2% comparison is descriptive. Earlier H1z remains a separate falsified study. No pilot or previous result is pooled, and the 5 pp margin, M10000, N1500 and bank count were not changed after outcomes.','',
        '[Protocol](protocol-calibrated.json) · [Proof](REPEATED_QUERY_CALIBRATION.md) · [Estimator](calibration_estimator.py) · [Acquisition](calibrated_acquire.py) · [Replay](audit_calibrated_acquisition.py) · [Search](calibrated_search.py) · [Public results](../../../results/research/ripper/calibrated.json)','']
    (HERE/'REPORT-calibrated.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'structure':structural_groups,'calibration_bounds_pct':[min(bounds),max(bounds)],'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
