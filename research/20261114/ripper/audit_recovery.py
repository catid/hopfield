"""Independent N600 recovery probability and saved-board inference audit."""
import math
import ast
from pathlib import Path
from bootstrap_two_way import resample_means
from recovery_common import load_protocol
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity

OUT=ROOT/'artifacts/research/20261114/ripper/recovery'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-recovery.json'


def main():
    p=load_protocol();registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'recovery_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'recovery_acquire.py') and registration['search_sha256']==sha(HERE/'recovery_search.py')
    assert registration['clarification_sha256']==sha(HERE/'RECOVERY_PROTOCOL_CLARIFICATION.md')
    assert registration['extended_policy_formula_checks']==1001
    validation_path=ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json';validation=json.loads(validation_path.read_text())
    assert validation['status']=='validated' and validation['helper_sha256']==sha(HERE/'bootstrap_two_way.py')
    manifest=json.loads((ROOT/p['inputs']).read_text())
    assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    assert sha(HERE/'exponential_policy.py')==p['frozen_policy_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256']
    assert sha(HERE/'spike_policy.py')==p['mixture_policy_sha256']
    preflight_path=ROOT/p['policy_preflight_path'];assert sha(preflight_path)==p['policy_preflight_sha256']
    preflight=json.loads(preflight_path.read_text())
    assert preflight['status']=='validated' and preflight['exponential_policy_sha256']==p['frozen_policy_sha256']
    assert preflight['original_policy_sha256']==p['original_policy_sha256']
    assert preflight['mixture_policy_sha256']==p['mixture_policy_sha256']
    assert preflight['wrapper_sha256']==sha(HERE/'test_exponential_policy.py')
    assert preflight['exact_rational_score_checks']==301 and preflight['initial_tie_checks']==64
    assert preflight['unchanged_control_checks']==128 and preflight['read_only_purity_and_determinism_checks']==64
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['input_protocol_sha256']==p['input_protocol_sha256']
    assert input_audit['wrapper_sha256']==sha(HERE/'audit_recovery_inputs.py')
    assert input_audit['unique_source_rows']==input_audit['independent_csv_answers_verified']==manifest['fresh_unique_rows']
    assert input_audit['within_bank_overlap']==input_audit['prior_bank_overlap']==0
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'acquisition_audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'recovery_acquire.py') and r['policy_sha256']==sha(HERE/'exponential_policy.py')
        assert a['wrapper_sha256']==sha(HERE/'audit_recovery_acquisition.py')
        assert a['acquisition_receipt_sha256']==sha(d/'receipt.json') and a['status']=='validated'
        assert a['independent_masks_replayed']==3000 and a['prefix_supports_verified']==12 and a['hidden_value_recount_checks']==24
        assert r['current_row_permutation_invariance_checks']==r['premature_reveal_rejections']==3000
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
    extra={(arm,n):np.array([next(r['extra_edges'] for r in structure if (r['arm'],r['n'],r['seed'])==(arm,n,s)) for s in p['seeds']]) for arm in p['arms'] for n in p['sizes']}
    def cp_limits(k,n,alpha=.05):
        if k==0:lower=0.
        else:
            lo,hi=0.,1.
            for _ in range(100):
                prob=(lo+hi)/2
                tail=sum(math.comb(n,x)*prob**x*(1-prob)**(n-x) for x in range(k,n+1))
                if tail<alpha:lo=prob
                else:hi=prob
            lower=(lo+hi)/2
        if k==n:upper=1.
        else:
            lo,hi=0.,1.
            for _ in range(100):
                prob=(lo+hi)/2
                tail=sum(math.comb(n,x)*prob**x*(1-prob)**(n-x) for x in range(k+1))
                if tail>alpha:lo=prob
                else:hi=prob
            upper=(lo+hi)/2
        return lower,upper
    cp_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/recovery_cp_validation.json').read_text())
    cp_node=next(node for node in ast.walk(ast.parse(Path(__file__).read_text())) if isinstance(node,ast.FunctionDef) and node.name=='cp_limits')
    assert cp_validation['status']=='validated' and cp_validation['limits_checked']==130
    assert cp_validation['cp_function_ast_sha256']==hashlib.sha256(ast.unparse(cp_node).encode()).hexdigest()
    k=int((extra['exponential',600]==0).sum());lower,upper=cp_limits(k,64)
    decision='supported_at_least_90percent_recovery' if lower>=.90 else 'falsified_90percent_recovery' if upper<.90 else 'inconclusive'
    primary={'decision':decision,'arm':'exponential','n':600,'exact_graphs':k,'banks':64,'observed_fraction':k/64,
        'one_sided95_lower':lower,'one_sided95_upper':upper,'target':.90,'method':'Exact binomial Clopper-Pearson, each limit one-sided95percent.',
        'scope':'Independent fresh source-bank draws from fixed eligible pool. Within-bank sampling without replacement; cross-bank overlap allowed. Not a superiority test.'}
    curve=[]
    for n in p['sizes']:
        for arm in p['arms']:
            v=extra[arm,n];kk=int((v==0).sum());ll,uu=cp_limits(kk,64)
            curve.append({'arm':arm,'n':n,'exact_graphs':kk,'banks':64,'total_extra_edges':int(v.sum()),
                'mean_extra_edges':float(v.mean()),'one_sided95_lower':ll,'one_sided95_upper':uu,'descriptive':True})
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status']=='complete' for j in queue) and len(queue)==mapping['unique_searches']
    assert len(mapping['rows'])==192 and mapping['wrapper_sha256']==sha(HERE/'recovery_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256']
    data=np.load(datafile);lo,hi=p['rows_half_open'];target=data['solutions'][lo:hi].astype(np.int8)-1
    free=~data['clue_mask'][lo:hi].astype(bool);testbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps=[{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    rows=[];outputs={};oracle=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'recovery_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[200000,210000] and r['test_rows']==10000
        assert r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==125 and r['hidden_target_state_comparisons']==110
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        support=supports[r['arm'],600,r['seed']]
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
        assert np.array_equal(supports[r['arm'],600,r['seed']],supports[m['arm'],600,m['seed']])
        arrays[m['arm'],m['seed']]=outputs[m['canonical_job']]
    grouped=[]
    for arm in p['arms']:
        solved=[int(arrays[arm,s].sum()) for s in p['seeds']]
        grouped.append({'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),
            'wrong_stops':0,'seeds':64,'n_per_seed':10000})
    contrasts=[]
    for arm,reference_arm in [('candidate','uniform'),('exponential','uniform'),('exponential','candidate')]:
        diff=np.array([arrays[arm,s].astype(float)-arrays[reference_arm,s] for s in p['seeds']])
        boot=resample_means(diff,20261301,5000)
        contrasts.append({'arm':arm,'reference':reference_arm,'effect_pp':float(100*diff.mean()),
            'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])]})
    stamp=lambda x:datetime.datetime.fromisoformat(x)
    span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    assert 4*span<30*60
    report={'status':'validated','utc':now(),'primary':primary,'recovery_curve':curve,'structure':structure,'grouped':grouped,
        'secondary_contrasts':contrasts,'rows':rows,'oracle_diagnostic':oracle,'saved_boards_verified':len(queue)*10000,
        'unique_searches':len(queue),'requested_searches':192,'search_reuse_mapping':mapping,
        'acquisition_audits':audits,'independent_masks_replayed':192000,'prefix_supports_verified':768,
        'hidden_value_recount_checks':1536,'timings':timings,'search_wall_sum_s':sum(r['wall_s'] for r in rows),
        'four_gpu_queue_wall_envelope_s':4*span,'training_test_completeboard_overlap':overlaps,
        'artifact_sha256':hashes,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,
        'prior_result_not_pooled':True,'frozen_policy_sha256':p['frozen_policy_sha256'],
        'independent_input_audit':input_audit,'input_audit_sha256':sha(OUT/'input_audit.json'),
        'binomial_interval_validation':cp_validation,'input_registration':registration,'audit_bootstrap_validation':validation,'standalone_policy_preflight':preflight,'policy_preflight_sha256':sha(preflight_path),
        'original_control_policy_sha256':p['original_policy_sha256'],
        'board_validity_helper_sha256':sha(HERE/'audit_clues.py'),'test_data_sha256':sha(datafile),
        'limitations':p['limitations']+[p['uncertainty_limit'],
            'The graph-coverage target is the registered primary. Secondary accuracy intervals are descriptive and unadjusted for multiple comparisons.',
            'Correct returned best boards may have positive energy; zero-energy oracle fractions bound correct certified stops only.']}
    save(OUT/'audit.json',report);save(PUBLIC/'recovery.json',report)
    for name,rs in [('recovery_structure.csv',structure),('recovery_per_seed.csv',rows),('recovery_oracle.csv',oracle)]:
        with(PUBLIC/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['sample_count_recovery_confirmation']=report;save(PUBLIC/'summary.json',summary)
    lines=['# H1w: sample-count graph recovery confirmation','',
        f'The frozen >=90% exact-recovery-at-N600 target is **{decision}**: exponential recovered {k}/64 exact graphs. Exact binomial one-sided 95% lower and upper limits are {lower:.6f} and {upper:.6f}, respectively. These are separate one-sided limits, not a 95% two-sided interval.','',
        '| Policy | Exact graphs at N600 / 64 | Mean N600 test solved (%) | Wrong stops |',
        '|---|---:|---:|---:|','| BP AdamW: outside this acquisition study; not run | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {int((extra[g["arm"],600]==0).sum())} | {g["mean_accuracy_pct"]:.4f} | 0 |')
    lines+=['','This is a sample-count recovery confirmation, not a superiority claim over candidate. Exponential was selected by the stated H1v noninferiority decision rule; its decay8/9 and25queriesperboard remain fixed. The exploratory six-old-bank feasibility study was used only to choose a realistic endpoint and is not pooled.','',
        f'The study draws 64 independent 1000-board banks from the same prior-excluded finite source pool. Sampling is without replacement within each bank; overlap across banks is allowed. There are {manifest["fresh_unique_rows"]:,} distinct source boards among64,000 bank rows; {manifest["cross_bank_duplicate_row_instances"]} row instances repeat across banks. There is no prior-bank or within-bank overlap. Binomial uncertainty assumes the independent sampling and tie-stream mechanism.','',
        'Every policy queries25cellsperincomingboard, so N600 costs15,000 revealed digits per bank. Only past queried counts enter selection. All192,000 masks were independently replayed, all768prefixsupports and1,536hidden-valuerecounts passed. Stored completions serve only as the simulated oracle; geometry enters the final audit.','',
        f'Inference evaluates N600 only on the same10,000 new full-all inputs[200000,210000) across64pairedbank/searchseeds. Exact support/settings/seed/input dedup reduces192requestedcells to{len(queue)}unique searches. All{len(queue)*10000:,}returnedboards were independently checked, with zero wrong stops. Repeated inference inputs are not counted as independent puzzles. Four chains,262,144movesperchain,unit1,B0,tau0,ebar0 and the frozen search configuration are unchanged.','',
        '| Policy | N | Exact graphs / 64 | Total extra edges |','|---|---:|---:|---:|']
    for c in curve:lines.append(f'| {c["arm"]} | {c["n"]} | {c["exact_graphs"]} | {c["total_extra_edges"]} |')
    lines+=['','All construction curves outside the registered exponential-N600 primary and all accuracy comparisons are descriptive and unadjusted.','']
    for c in contrasts:lines.append(f'Secondary {c["arm"]} minus {c["reference"]} accuracy: {c["effect_pp"]:.4f} pp, descriptive 95% interval {c["descriptive_ci95_pp"]}.')
    lines+=['','The inherited RNG prose range was clarified before acquisition to cover the64seeds explicitly listed in the frozen protocol. No outcomes or settings were changed. See[pre-acquisition clarification](RECOVERY_PROTOCOL_CLARIFICATION.md).','',
        '[Protocol](protocol-recovery.json) · [Acquisition](recovery_acquire.py) · [Replay](audit_recovery_acquisition.py) · [Search](recovery_search.py) · [Audit](audit_recovery.py) · [Public results](../../../results/research/ripper/recovery.json)','']
    (HERE/'REPORT-recovery.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'secondary':contrasts,'gpu_envelope_s':4*span}))


if __name__=='__main__':main()
