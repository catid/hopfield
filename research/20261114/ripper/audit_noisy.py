"""Independent noisy acquisition, saved-board and paired accuracy audit."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity
from noisy_common import load_protocol
from bootstrap_two_way import resample_means

OUT=ROOT/'artifacts/research/20261114/ripper/noisy'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-noisy.json'


def main():
    p=load_protocol();manifest=json.loads((ROOT/p['inputs']).read_text())
    registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'noisy_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'noisy_acquire.py') and registration['search_sha256']==sha(HERE/'noisy_search.py')
    assert sha(HERE/'noisy_policy.py')==p['policy_sha256']==registration['policy_sha256']
    assert sha(HERE/'NOISY_ACQUISITION_EPROCESS.md')==p['proof_sha256']==registration['proof_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256'] and sha(HERE/'exponential_policy.py')==p['original_exponential_sha256']
    assert sha(ROOT/p['policy_preflight_path'])==p['policy_preflight_sha256'];preflight=json.loads((ROOT/p['policy_preflight_path']).read_text())
    assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    assert preflight['exact_null_probability_checks']==72 and preflight['exact_martingale_expectation']
    assert preflight['independent_decimal_crossing_thresholds']==3001 and preflight['pure_control_checks']==256
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['input_protocol_sha256']==p['input_protocol_sha256'] and input_audit['wrapper_sha256']==sha(HERE/'audit_noisy_inputs.py')
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
        assert r['wrapper_sha256']==sha(HERE/'noisy_acquire.py') and a['wrapper_sha256']==sha(HERE/'audit_noisy_acquisition.py')
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
            if arm=='exponential_naive':assert np.array_equal(support,m['equal_count']==0)
            else:assert np.array_equal(support,m['first_crossing']<0)
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
    assert len(mapping['rows'])==96 and mapping['wrapper_sha256']==sha(HERE/'noisy_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'noisy_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[210000,220000] and r['test_rows']==10000
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
    arrays={};wrong={};stopped_counts={}
    for m in mapping['rows']:
        r=next(r for r in rows if r['job_id']==m['canonical_job']);assert r['seed']==m['seed']
        assert np.array_equal(supports[r['arm'],1500,r['seed']],supports[m['arm'],1500,m['seed']])
        arrays[m['arm'],m['seed']]=outputs[m['canonical_job']]
        wrong[m['arm'],m['seed']]=r['wrong_stops']
    grouped=[]
    for arm in p['arms']:
        solved=[int(arrays[arm,s].sum()) for s in p['seeds']]
        grouped.append({'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),
            'wrong_stops':sum(wrong[arm,s] for s in p['seeds']),'wrong_stops_per_seed':[wrong[arm,s] for s in p['seeds']],'seeds':32,'n_per_seed':10000})
    contrasts=[]
    for arm,reference_arm in [('exponential_e','uniform_e'),('uniform_e','exponential_naive'),('exponential_e','exponential_naive')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        boot=resample_means(diff,20270000,5000)
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    c=contrasts[0];lo_ci,hi_ci=c['descriptive_ci95_pp']
    decision='supported_at_least_10pp_gain' if lo_ci>=10 else 'falsified_10pp_gain' if hi_ci<10 else 'inconclusive'
    primary={'decision':decision,'comparison':'exponential_e minus uniform_e','effect_pp':c['effect_pp'],
        'paired_two_way_bootstrap_ci95_pp':[lo_ci,hi_ci],'target_pp':10,'bank_seeds':32,'distinct_test_inputs':10000,
        'bootstrap_draws':5000,'bootstrap_seed':20270000,'scope':p['uncertainty_limit']}
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
    save(OUT/'audit.json',report);save(PUBLIC/'noisy.json',report)
    for name,rs in [('noisy_structure.csv',structure),('noisy_per_seed.csv',rows),('noisy_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['noisy_adaptive_acquisition']=report;save(PUBLIC/'summary.json',summary)
    lines=['# H1y: noisy adaptive acquisition','',
        f'The frozen >=10pp exponential-e-process versus uniform-e-process accuracy target is **{decision}**. The gain is{primary["effect_pp"]:.4f}pp,paired two-way95%bootstrap interval[{lo_ci:.4f},{hi_ci:.4f}]pp.','',
        '| Policy | Mean solved (%) | Wrong stops / 320,000 repeated evaluations | Exact graphs / 32 | Missing true edges | Extra edges |',
        '|---|---:|---:|---:|---:|---:|','| BP AdamW: outside this mechanism study; not run | — | — | — | — | — |']
    for g in grouped:
        t=next(t for t in structural_groups if t['arm']==g['arm'])
        lines.append(f'| {g["arm"]} | {g["mean_accuracy_pct"]:.4f} | {g["wrong_stops"]} | {t["exact_graphs"]} | {t["total_missing_true_edges"]} | {t["total_extra_edges"]} |')
    lines+=['',f'The32independent1500-board source-bank draws contain{manifest["fresh_unique_rows"]:,}distinct complete boards among48,000rowinstances. Cross-bank overlap is allowed; all260,724previoussourceboards are excluded. Each bank has its own fixed source/noise/tie stream.','',
        'The label oracle consists of100%complete stored boards. Each policy receives only25queriednoisylabelsperrow (37,500perbank). Every current mask is finalized before any current-row noisy label is exposed. Each queried label independently flips with eta=.01 to a uniformly different digit. The known law is supplied; eta and q1=1/9 are not fit.','',
        'The e-process excludes pairs only at their first likelihood-ratio crossing of324000. Its predictable-mask martingale and Ville/union proof bounds deleting any trueedge by1%perlearnedmodel. This does not control extras,exactrecovery,solveaccuracy,or arbitrary learned-model stopping certificates. The exposure score remains(8/9)^m until exclusion. The naive control instead excludes after its first observed equality and lacks this protection.','',
        'All144,000masks,466,560,000log-LRvalues, noise realizations,96supports and311,040first-crossing paths were independently checked before/after search. Exact rational crossing decisions are independently validated with70-digit Decimal calculations.','',
        f'Inference uses the same10,000fresh full-all inputs[210000,220000) across32paired bank/searchseeds; these are repeated evaluations,not320,000distinctpuzzles. Exact support/settings/seed/input dedup reduces96requestedcells to{len(queue)}searches. All{len(queue)*10000:,}returnedboards were independently checked. Fourchains,262,144movesperchain,unit1,B0,tau0,ebar0 remain fixed. Wrong stops are retained as measured outcomes.','',
        'The old six-bank pilot selected theN1500endpoint only and is not pooled. Structural and other accuracy comparisons are descriptive. A support containing all trueedges has a valid integer zero-energy certificate under preserved clues; budgeted best-board returns have no such guarantee.','',
        '[Protocol](protocol-noisy.json) · [Filtration proof](NOISY_ACQUISITION_EPROCESS.md) · [Policy](noisy_policy.py) · [Acquisition](noisy_acquire.py) · [Replay](audit_noisy_acquisition.py) · [Search](noisy_search.py) · [Audit](audit_noisy.py) · [Public results](../../../results/research/ripper/noisy.json)','']
    (HERE/'REPORT-noisy.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'structure':structural_groups,'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
