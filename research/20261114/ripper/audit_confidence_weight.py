"""Independent weighted saved-board audit and frozen three-endpoint analysis."""
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,sha,save,now
from audit_clues import validity
from confidence_weight_common import load_protocol
from bootstrap_two_way import resample_means

OUT=ROOT/'artifacts/research/20261114/ripper/confidence_weight'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-confidence-weight.json'


def main():
    p=load_protocol();manifest=json.loads((OUT/'input_manifest.json').read_text());registration=json.loads((OUT/'input_registration.json').read_text())
    source_audit=json.loads((OUT/'input_audit.json').read_text());assert source_audit['status']=='validated' and source_audit['independent_csv_answers_verified']==100410
    assert source_audit['wrapper_sha256']==sha(HERE/'audit_confidence_weight_inputs.py') and source_audit['protocol_sha256']==sha(PROTOCOL)
    assert source_audit['calibration_training_overlap']==source_audit['within_bank_overlap']==source_audit['prior_bank_overlap']==0
    bootstrap_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/bootstrap_validation.json').read_text())
    assert bootstrap_validation['status']=='validated' and bootstrap_validation['helper_sha256']==sha(HERE/'bootstrap_two_way.py')
    preflight=json.loads((ROOT/p['policy_preflight_path']).read_text());assert preflight['status']=='validated' and preflight['policy_sha256']==p['policy_sha256']
    assert preflight['independent_fraction_sign_checks']==384 and preflight['independent_decimal_weight_and_shuffle_cases']==48
    pair=np.triu_indices(81,1);i,j=pair;peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    weights={};supports={};structure=[];fits=[];acquisition_audits=[];hashes={};sources={};timings=[]
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'audit.json').read_text())
        assert a['status']=='validated' and r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'confidence_weight_acquire.py') and a['wrapper_sha256']==sha(HERE/'audit_confidence_weight_acquisition.py')
        assert a['receipt_sha256']==sha(d/'receipt.json') and r['policy_sha256']==p['policy_sha256']
        assert a['independent_masks_replayed']==2220 and a['log_lr_values_verified']==7192800 and a['decimal_threshold_checks']==2221
        assert a['first_crossing_paths_verified']==3240 and a['weight_models_verified']==3 and a['weight_entries_verified']==9720
        assert a['noise_cached_readonly_and_unchanged'] and a['fast_mask_helper_sha256']==p['fast_reference_mask_sha256']
        for name,h in r['artifact_sha256'].items():assert sha(d/name)==h
        fit=json.loads((d/'calibration.json').read_text())['fit'];assert fit==a['calibration_fit'];fits.append(dict(seed=seed,**fit))
        base=np.load(d/'fit.npz');support=base['support'];assert np.array_equal(support,base['first_crossing']<0);supports[seed]=support
        assert hashlib.sha256(support.tobytes()).hexdigest()==a['support_bytes_sha256']
        entry=next(e for e in manifest['inputs'] if e['seed']==seed)
        for role in ['training','calibration']:
            assert sha(ROOT/entry[role+'_path'])==entry[role+'_sha256'];sources[role,seed]=np.load(ROOT/entry[role+'_path'])['solutions']
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];z=np.load(ROOT/model['path']);w=z['weight16'];arm=model['arm']
            assert np.array_equal(z['support'],support) and np.array_equal(w>0,support) and set(np.unique(w))<={0,1,16}
            assert model['weight16_bytes_sha256']==hashlib.sha256(w.tobytes()).hexdigest();weights[arm,seed]=w
            structure.append({'seed':seed,'arm':arm,'missing_true_edges':int((peer&~support).sum()),'extra_edges':int((~peer&support).sum()),
                'weak_true_edges':int((peer&(w==1)).sum()),'weak_extra_edges':int((~peer&(w==1)).sum()),'weak_weights':int((w==1).sum()),
                'strong_weights':int((w==16).sum()),'weight16_bytes_sha256':model['weight16_bytes_sha256']})
        assert np.array_equal(np.sort(weights['evidence',seed]),np.sort(weights['shuffled',seed]))
        acquisition_audits.append(a);timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'replay_s':a['elapsed_s']})
        for name in ['receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    queue=json.loads((OUT/'queue.json').read_text());mapping=json.loads((OUT/'search_mapping.json').read_text())
    assert all(job['status']=='complete' for job in queue) and len(queue)==mapping['unique_searches'] and len(mapping['rows'])==96
    assert mapping['wrapper_sha256']==sha(HERE/'confidence_weight_search.py')
    datafile=ROOT/'upstream/data/full-test/full-all.npz';assert sha(datafile)==mapping['test_data_sha256'];data=np.load(datafile);lo,hi=p['rows_half_open']
    target=data['solutions'][lo:hi].astype(np.int8)-1;free=~data['clue_mask'][lo:hi].astype(bool);assert len(target)==10000
    targetbytes=np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel();overlap=[]
    for (role,seed),boards in sources.items():overlap.append({'role':role,'seed':seed,'shared_complete_boards':int(np.isin(np.ascontiguousarray(boards).view(np.dtype((np.void,81))).ravel(),targetbytes).sum())})
    outputs={};stats={};rows=[]
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text());arm,seed=r['arm'],r['seed'];w=weights[arm,seed];support=w>0
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'confidence_weight_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256'] and r['integer_audit_sha256']==p['integer_audit_sha256']
        assert r['rows_half_open']==[380000,390000] and r['test_rows']==10000 and r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==150 and r['hidden_target_state_comparisons']==132 and r['integer_field_checks']==6
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':seed,'anneal':[.4,.31,524288]}
        assert r['weight16_bytes_sha256']==hashlib.sha256(w.tobytes()).hexdigest()
        for name,h in r['output_sha256'].items():assert sha(d/name)==h
        model=torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu')
        assert torch.equal(model['W'],torch.from_numpy(w.astype(np.float32)/16)[:,None,None]*torch.eye(9)) and not model['B'].any()
        assert np.array_equal(model['weight16'].numpy(),w)
        z=np.load(d/'boards.npz');boards=z['boards'];stopped=z['stopped'];moves=z['moves']
        assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
        assert np.array_equal(boards[~free],target[~free]);exact=(boards==target).all(1);valid=validity(boards,target,free)
        assert np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid']) and np.all(~exact|valid)
        assert int(exact.sum())==r['solved'] and int(valid.sum())==r['valid'] and int((~stopped).sum())==r['budget_failures']
        assert int((stopped&~valid).sum())==r['wrong_stops'] and np.all((moves>0)&(moves<=262144))
        per=json.loads((d/'result.json').read_text())['per_puzzle'];assert np.array_equal(per['solved'],exact) and np.array_equal(per['stopped'],stopped)
        assert np.array_equal(per['flops'],moves*r['fl_move']*4)
        ii,jj=i[support],j[support];ww=w[support].astype(np.int64)
        conflicts=target[:,ii]==target[:,jj];target_e16=(conflicts*ww).sum(1);returned_e16=((boards[:,ii]==boards[:,jj])*ww).sum(1)
        assert np.all(~stopped|(returned_e16==0)) and np.all(~(stopped&exact)|(target_e16==0))
        if support[peer].all():assert not (stopped&~valid).any()
        compatible=target_e16==0;given_conflicts=(conflicts&~free[:,ii]&~free[:,jj]).sum(1);assert not (stopped&(given_conflicts>0)).any()
        diagnostic={'canonical_compatible':int(compatible.sum()),'given_given_infeasible':int((given_conflicts>0).sum()),
            'correct_certified':int((stopped&exact).sum()),'correct_capped':int((~stopped&exact).sum()),
            'incompatible_correct_capped':int((~compatible&~stopped&exact).sum()),'compatible_target_misses':int((compatible&~exact).sum()),
            'wrong_stops':int((stopped&~valid).sum()),'invalid_capped':int((~stopped&~valid).sum()),
            'noncanonical_valid_returns':int((valid&~exact).sum()),'noncanonical_valid_certificates':int((stopped&valid&~exact).sum()),
            'mean_canonical_energy':float(target_e16.mean()/16)}
        outputs[job['job_id']]=exact;stats[job['job_id']]=diagnostic
        rows.append({**{k:r[k] for k in ['job_id','arm','seed','solved','valid','wrong_stops','budget_failures','mean_moves','wall_s','D','edges']},**diagnostic})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    arrays={};diagnostics=[]
    for entry in mapping['rows']:
        original=next(row for row in rows if row['job_id']==entry['canonical_job']);seed=entry['seed'];arm=entry['arm']
        assert seed==original['seed'] and np.array_equal(weights[arm,seed],weights[original['arm'],seed])
        arrays[arm,seed]=outputs[entry['canonical_job']];diagnostics.append({'arm':arm,'seed':seed,**stats[entry['canonical_job']]})
    grouped=[]
    for arm in p['arms']:
        solved=[int(arrays[arm,s].sum()) for s in p['seeds']];part=[x for x in diagnostics if x['arm']==arm]
        group={'arm':arm,'mean_accuracy_pct':float(np.mean(solved)/100),'solved_per_seed':solved,'banks':32,'n_per_bank':10000}
        for key in part[0]:
            if key in ('arm','seed'):continue
            if key=='mean_canonical_energy':group[key]=float(np.mean([x[key] for x in part]))
            else:group[key]=sum(x[key] for x in part);group[key+'_pct']=group[key]/3200
        grouped.append(group)
    contrasts=[]
    for reference in ['uniform','shuffled']:
        diff=np.array([arrays['evidence',s].astype(float)-arrays[reference,s] for s in p['seeds']]);boot=resample_means(diff,20401000,5000)
        lower,upper=[float(x) for x in np.quantile(boot,[.025,.975])]
        decision='supported_10pp_gain' if lower>=10 else 'falsified_10pp_gain' if upper<10 else 'inconclusive'
        contrasts.append({'arm':'evidence','reference':reference,'effect_pp':float(100*diff.mean()),'paired_two_way_ci95_pp':[lower,upper],'minimum_gain_pp':10,'decision':decision})
    matrix=np.array([arrays['evidence',s] for s in p['seeds']],dtype=float);absolute_boot=resample_means(matrix,20401000,5000)
    lower,upper=[float(x) for x in np.quantile(absolute_boot,[.025,.975])];mean=float(100*matrix.mean())
    floor='supported_70percent_floor' if lower>=70 else 'falsified_70percent_floor' if upper<70 else 'inconclusive'
    wrong=sum(g['wrong_stops'] for g in grouped);decisions=[c['decision'] for c in contrasts]+[floor]
    conjunction='falsified_conjunction' if wrong or any(x.startswith('falsified') for x in decisions) else 'supported_conjunction' if all(x.startswith('supported') for x in decisions) else 'inconclusive_conjunction'
    primary={'decision':conjunction,'gain_endpoints':contrasts,'mean_evidence_accuracy_pct':mean,'evidence_accuracy_ci95_pct':[lower,upper],
        'usefulness_floor_pct':70,'usefulness_decision':floor,'wrong_certified_stops':wrong,'zero_wrong_stop_gate':wrong==0,
        'accuracy_definition':p['primary_accuracy'],'wrong_certificate_definition':p['wrong_certificate'],'banks':32,'distinct_test_inputs':10000,'bootstrap_draws':5000,'bootstrap_seed':20401000,
        'interval_scope':'Three marginal nominal95% intervals, no simultaneous-coverage claim.','uncertainty':p['uncertainty_limit']}
    stamp=datetime.datetime.fromisoformat;span=(max(stamp(j['completed_utc']) for j in queue)-min(stamp(j['started_utc']) for j in queue)).total_seconds()
    gpu_seconds=p['gpu_preflight_seconds']+4*span;assert gpu_seconds<1800
    coverage=sum(f['postfit_covers_actual_eta'] for f in fits);fallback=sum(f['no_pruning_fallback'] for f in fits)
    report={'status':'validated','utc':now(),'primary':primary,'grouped':grouped,'rows':rows,'diagnostics':diagnostics,'structure':structure,
        'calibration_results':fits,'calibration_coverage_banks':coverage,'calibration_fallback_banks':fallback,'shared_training_fits':32,
        'requested_searches':96,'unique_searches':len(queue),'saved_boards_verified':len(queue)*10000,'search_mapping':mapping,'queries_per_shared_fit':57500,
        'source_test_completeboard_overlap':overlap,'input_audit':source_audit,'input_registration':registration,'acquisition_audits':acquisition_audits,
        'independent_masks_replayed':71040,'log_lr_values_verified':230169600,'first_crossing_paths_verified':103680,'weight_models_verified':96,
        'gpu_preflight_seconds':p['gpu_preflight_seconds'],'four_gpu_queue_envelope_s':4*span,'total_allocated_gpu_s':gpu_seconds,'search_wall_sum_s':sum(row['wall_s'] for row in rows),
        'timings':timings,'weight_preflight':preflight,'bootstrap_validation':bootstrap_validation,'artifact_sha256':hashes,
        'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'numpy':np.__version__,'limitations':p['limitations']}
    save(OUT/'audit.json',report);save(PUBLIC/'confidence_weight.json',report)
    for name,entries in [('confidence_weight_structure.csv',structure),('confidence_weight_per_seed.csv',rows),('confidence_weight_compatibility.csv',diagnostics)]:
        with (PUBLIC/name).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    lines=['# H1ah: confidence weights for capped-return accuracy','',f'The frozen conjunction is **{conjunction}**. Endpoints are reported separately:','']
    for c in contrasts:lines.append(f'- Evidence minus {c["reference"]}, required gain 10 pp: **{c["decision"]}**. Mean {c["effect_pp"]:.4f} pp; marginal paired two-way 95% interval [{c["paired_two_way_ci95_pp"][0]:.4f}, {c["paired_two_way_ci95_pp"][1]:.4f}] pp.')
    lines.extend([f'- Evidence accuracy, required floor 70%: **{floor}**. Mean {mean:.4f}%; marginal two-way 95% interval [{lower:.4f}, {upper:.4f}]%.',
        f'- Zero invalid certified stops across all three arms: **{"passed" if wrong==0 else "failed"}**, with {wrong} invalid certificates.','',
        'Primary accuracy is exact match to the canonical solution, including capped returns. All three intervals have marginal nominal 95% coverage; no simultaneous-coverage claim is made.','',
        '| Weights | Canonical accuracy (%) | Correct certified (%) | Correct capped (%) | Canonical compatibility (%) | Given–given infeasibility (%) | Invalid certificates | Noncanonical valid returns |',
        '|---|---:|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this study; not run | — | — | — | — | — | — | — |'])
    for g in grouped:lines.append(f'| {g["arm"]} | {g["mean_accuracy_pct"]:.4f} | {g["correct_certified_pct"]:.4f} | {g["correct_capped_pct"]:.4f} | {g["canonical_compatible_pct"]:.4f} | {g["given_given_infeasible_pct"]:.4f} | {g["wrong_stops"]} | {g["noncanonical_valid_returns"]} |')
    for arm in ['evidence','shuffled']:
        part=[x for x in structure if x['arm']==arm]
        lines.extend(['',f'Descriptive weight placement for {arm}: {sum(x["weak_true_edges"] for x in part)} weak true-edge–bank pairs and {sum(x["weak_extra_edges"] for x in part)} weak false-edge–bank pairs. Geometry enters only this postfit audit, never weight assignment.'])
    base=[x for x in structure if x['arm']=='uniform']
    lines.extend(['',f'All arms share the same 32 learned supports: {sum(x["missing_true_edges"] for x in base)} missing true edges and {sum(x["extra_edges"] for x in base)} extra edges in total. Each fit costs 57,500 queries (2,000 calibration plus 55,500 training); inference weightings do not consume additional labels or create independent training fits.',
        '', 'Uniform weights are 1. Evidence weights are 1 for exact rational LR≤1 and 1/16 otherwise. Shuffled weights have the same multiset, placed by a frozen full-pair permutation generated from an independent stream before calibration/training observations. No geometry or held-out label enters assignment.',
        '', 'Strong penalty unit 1, temperatures .4→.31, theta .45, tau0/ebar0 and tolerance .001 remain fixed. There is no per-model or mean-weight normalization. Positive weights preserve the zero set. Compatibility is for the stored canonical solution; uniqueness is not independently established. Given–given conflicts preclude zero-energy completion, but neither measure bounds positive-energy capped-return accuracy.',
        '', f'The 32 shared calibration bounds cover actual 5% noise in {coverage}/32 banks, with {fallback} no-pruning fallbacks. All outcomes are retained. Source audits verified 100,410 distinct boards (31,533 calibration and 68,877 training), with globally disjoint source roles and all 1,628,683 prior Ripper rows excluded.',
        '', 'Before inference, independent replay checked 71,040 masks, 230,169,600 log-LR values, 103,680 first-crossing paths and 96 exact weight/permutation assignments. Each fresh search passed 150 general/sparse and 132 hidden-target state comparisons plus six exact integer field/energy/focus checks, including 127 incremental moves without refresh. Integer energies independently verified every returned certificate.',
        '', f'All bank/search seeds share 10,000 inputs [380000,390000). Complete weight/settings/seed/input equality reduces 96 requested cells to {len(queue)} searches; all {len(queue)*10000:,} returned boards were audited. Total GPU allocation including standalone preflights is {gpu_seconds/60:.4f} GPU-min. Earlier studies and the six implementation pilots are separate and not pooled.',
        '', '[Protocol](protocol-confidence-weight.json) · [Design and proof](CONFIDENCE_WEIGHTING.md) · [Weight rule](confidence_weight_policy.py) · [Acquisition audit](audit_confidence_weight_acquisition.py) · [Results](../../../results/research/ripper/confidence_weight.json)',''])
    (HERE/'REPORT-confidence-weight.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'coverage':coverage,'fallback':fallback,'gpu_seconds':gpu_seconds}))


if __name__=='__main__':main()
