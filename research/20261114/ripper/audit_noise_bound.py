"""Independent noisy acquisition, saved-board and paired accuracy audit."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity
from noise_bound_common import load_protocol
from bootstrap_two_way import resample_means

OUT=ROOT/'artifacts/research/20261114/ripper/noise_bound'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-noise-bound.json'


def main():
    p=load_protocol();manifest=json.loads((ROOT/p['inputs']).read_text())
    registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'noise_bound_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'noise_bound_acquire.py') and registration['search_sha256']==sha(HERE/'noise_bound_search.py')
    assert sha(HERE/'noise_bound_policy.py')==p['policy_sha256']==registration['policy_sha256']
    assert sha(HERE/'NOISE_UPPER_BOUND_EPROCESS.md')==p['proof_sha256']==registration['proof_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256'] and sha(HERE/'exponential_policy.py')==p['original_exponential_sha256']
    assert sha(ROOT/p['policy_preflight_path'])==p['policy_preflight_sha256'];preflight=json.loads((ROOT/p['policy_preflight_path']).read_text())
    assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    assert preflight['decimal_threshold_checks']==4503 and preflight['unchanged_mask_checks']==192
    assert preflight['exact_control_thresholds_unchanged']
    assert preflight['conditional_null_expectations']['exact']['value']==1
    assert preflight['conditional_null_expectations']['upper']['value']<1<preflight['conditional_null_expectations']['under']['value']
    assert p['original_noisy_policy_sha256']==sha(HERE/'noisy_policy.py')
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['input_protocol_sha256']==p['input_protocol_sha256'] and input_audit['wrapper_sha256']==sha(HERE/'audit_noise_bound_inputs.py')
    assert input_audit['unique_source_rows']==input_audit['independent_csv_answers_verified']==manifest['fresh_unique_rows']
    assert input_audit['within_bank_overlap']==input_audit['prior_bank_overlap']==0
    bootstrap_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json').read_text())
    assert bootstrap_validation['status']=='validated' and bootstrap_validation['helper_sha256']==sha(HERE/'bootstrap_two_way.py')
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'noise_bound_acquire.py') and a['wrapper_sha256']==sha(HERE/'audit_noise_bound_acquisition.py')
        assert r['policy_sha256']==p['policy_sha256'] and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['status']=='validated' and a['independent_masks_replayed']==4500 and a['log_lr_values_verified']==14580000
        assert a['independent_noise_rows_verified']==1500 and a['supports_verified']==3 and a['first_crossing_paths_verified']==9720
        assert r['masks_fixed_before_all_reveals']==4500;audits.append(a)
        timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'independent_replay_s':a['elapsed_s']})
        e=next(e for e in manifest['inputs'] if e['seed']==seed);assert sha(ROOT/e['completion_path'])==e['completion_sha256']==r['source_sha256']
        completion[seed]=np.load(ROOT/e['completion_path'])['solutions']
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];m=np.load(ROOT/model['path']);support=m['support']
            assert support.dtype==bool and len(support)==3240
            arm,n=model['arm'],model['n'];supports[arm,n,seed]=support
            assert np.array_equal(support,m['first_crossing']<0)
            missing=int((peer&~support).sum());extra=int((~peer&support).sum())
            checked=next(x for x in a['rows'] if x['arm']==arm and x['n']==n)
            assert checked['missing_true_edges']==missing and checked['extra_edges']==extra
            assert checked['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            structure.append({'seed':seed,'n':n,'arm':arm,'queried_noisy_digits':25*n,'extra_edges':extra,
                'missing_true_edges':missing,'retained_edges':int(support.sum()),'exact_graph':extra==missing==0,
                'support_bytes_sha256':checked['support_bytes_sha256']})
        for name in ['receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue) and len(queue)==mapping['unique_searches']
    assert len(mapping['rows'])==96 and mapping['wrapper_sha256']==sha(HERE/'noise_bound_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'noise_bound_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[230000,240000] and r['test_rows']==10000
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
    for arm,reference_arm in [('upper','exact'),('under','exact'),('upper','under')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        boot=resample_means(diff,20275000,5000)
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    c=contrasts[0];lo_ci,hi_ci=c['descriptive_ci95_pp']
    protected_wrong=sum(g['wrong_stops'] for g in grouped if g['arm'] in ('exact','upper'))
    accuracy_decision='supported_10pp_noninferiority' if lo_ci>=-10 else 'falsified_10pp_noninferiority' if hi_ci<-10 else 'inconclusive'
    decision='falsified_protected_wrong_stop_gate' if protected_wrong else accuracy_decision
    primary={'decision':decision,'accuracy_decision':accuracy_decision,'comparison':'upper minus exact','effect_pp':c['effect_pp'],
        'paired_two_way_bootstrap_ci95_pp':[lo_ci,hi_ci],'noninferiority_margin_pp':10,
        'protected_wrong_stops':protected_wrong,'zero_protected_wrong_stops_gate':protected_wrong==0,
        'bank_seeds':32,'distinct_test_inputs':10000,'bootstrap_draws':5000,'bootstrap_seed':20275000,'scope':p['uncertainty_limit']}
    structural_groups=[]
    for arm in p['arms']:
        rs=[r for r in structure if r['arm']==arm]
        structural_groups.append({'arm':arm,'banks':32,'exact_graphs':sum(r['exact_graph'] for r in rs),
            'all_true_edges_retained':sum(r['missing_true_edges']==0 for r in rs),
            'total_missing_true_edges':sum(r['missing_true_edges'] for r in rs),'total_extra_edges':sum(r['extra_edges'] for r in rs)})
    report={'status':'validated','utc':now(),'primary':primary,'structure':structure,'structural_groups':structural_groups,
        'grouped':grouped,'secondary_contrasts':contrasts[1:],'rows':rows,'oracle_diagnostic':oracle,
        'saved_boards_verified':len(queue)*10000,'unique_searches':len(queue),'requested_searches':96,'search_reuse_mapping':mapping,
        'acquisition_audits':audits,'independent_masks_replayed':144000,'log_lr_values_verified':466560000,
        'prefix_supports_verified':96,'first_crossing_paths_verified':311040,'timings':timings,
        'search_wall_sum_s':sum(r['wall_s'] for r in rows),'four_gpu_queue_wall_envelope_s':4*span,
        'training_test_completeboard_overlap':overlaps,'independent_input_audit':input_audit,
        'input_registration':registration,'standalone_policy_preflight':preflight,'audit_bootstrap_validation':bootstrap_validation,
        'artifact_sha256':hashes,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,
        'proof_sha256':p['proof_sha256'],'pilot_not_pooled':True,'test_data_sha256':sha(datafile),
        'limitations':p['limitations']+[p['uncertainty_limit'],'Correct zero-energy target fractions bound correct certified stops only,not all returned-best-board accuracy.']}
    save(OUT/'audit.json',report);save(PUBLIC/'noise_bound.json',report)
    for name,rs in [('noise_bound_structure.csv',structure),('noise_bound_per_seed.csv',rows),('noise_bound_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['conservative_noise_calibration']=report;save(PUBLIC/'summary.json',summary)
    lines=['# H1z: conservative noise calibration','',
        f'The frozen joint target of 10 pp noninferiority and zero protected wrong stops is **{decision}**. Accuracy with an assumed 2% noise rate minus accuracy with the exact 1% rate is {primary["effect_pp"]:.4f} pp, with a paired two-way 95% bootstrap interval [{lo_ci:.4f}, {hi_ci:.4f}] pp. Noninferiority requires the lower bound to be at least -10 pp. The two protected arms produced {protected_wrong} wrong certified stops.','',
        '| Assumed noise rate | Mean solved (%) | Wrong certified stops | Invalid capped returns | Exact graphs / 32 | Missing true edges | Extra edges |',
        '|---|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — | — | — |']
    labels={'exact':'1% exact','upper':'2% conservative upper bound','under':'0.5% unprotected underbound'}
    for g in grouped:
        t=next(t for t in structural_groups if t['arm']==g['arm'])
        lines.append(f'| {labels[g["arm"]]} | {g["mean_accuracy_pct"]:.4f} | {g["wrong_stops"]} | {g["invalid_capped_returns"]} | {t["exact_graphs"]} | {t["total_missing_true_edges"]} | {t["total_extra_edges"]} |')
    lines+=['',f'All 32 independent banks use 1,500 boards, an actual noise rate of 1%, 25 queries per board, fixed q1=1/9, and the same exponential mask rule. N1500 and the 10 pp margin were fixed before the implementation pilot. The new banks contain {manifest["fresh_unique_rows"]:,} distinct source boards among 48,000 row instances and exclude 308,415 previously used boards.','',
        'The stored, fully labeled source boards simulate the label oracle. Each policy receives only 37,500 noisy queried digits per bank and finalizes each mask before seeing any current noisy values. Source, noise, and tie streams are independent across banks. Common potential noisy values pair the three calibrations within each bank.','',
        'When 0 <= eta_true <= eta_bound < 8/9, the never-equal noisy-equality probability increases with eta. With q0_bound < q1, the conditional expected likelihood-ratio multiplier is at most one. The predictable-mask supermartingale and Ville/union proof therefore retain the 1% per-model protection against deleting true edges. This does not guarantee few extra edges, solve accuracy, or the separate zero-wrong-stop condition. At eta_bound=8/9, the likelihood ratio is identically one and cannot exclude pairs.','',
        'The 0.5% underbound has a conditional expected null multiplier above one and lacks this probability guarantee. Its measured success or failure is not assumed; observing zero errors does not establish protection. All 144,000 masks, 466,560,000 log-LR values, noise draws, 96 supports, and 311,040 first-crossing paths passed independent audit. The exact control reproduced the original old-bank masks and paths during implementation validation.','',
        f'Inference uses the same 10,000 new full-all inputs [230000,240000) across 32 paired bank/search seeds. Exact support/settings/seed/input deduplication reduces 96 requested cells to {len(queue)} searches; all {len(queue)*10000:,} returned boards were independently checked. Reusing an identical model does not create independent input evidence. Four chains, 262,144 moves per chain, unit 1, B=0, tau=0, and ebar=0 remain fixed. Wrong certified stops and invalid capped returns are counted separately.','',
        'The earlier exact-calibration H1y result remains unchanged. The implementation pilot and earlier studies are not pooled. Other accuracy contrasts and structural comparisons are descriptive and unadjusted.','',
        '[Protocol](protocol-noise-bound.json) · [Upper-bound proof](NOISE_UPPER_BOUND_EPROCESS.md) · [Policy](noise_bound_policy.py) · [Acquisition](noise_bound_acquire.py) · [Replay](audit_noise_bound_acquisition.py) · [Search](noise_bound_search.py) · [Audit](audit_noise_bound.py) · [Public results](../../../results/research/ripper/noise_bound.json)','']
    (HERE/'REPORT-noise-bound.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'structure':structural_groups,'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
