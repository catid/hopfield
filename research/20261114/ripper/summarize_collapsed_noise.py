"""Source-only destructive-correlation endpoint with archived paired iid controls."""
from fractions import Fraction
import csv
import json
import numpy as np
from run import ROOT,HERE,sha,save,now
from collapsed_noise_common import paths,load_protocol


def main():
    out,pp=paths();p=load_protocol();registration=json.loads((out/'input_registration.json').read_text())
    law=json.loads((ROOT/'artifacts/research/20261114/ripper/collapsed_noise_law.json').read_text())
    assert law['status']=='validated' and law['auditor_sha256']==sha(HERE/'test_collapsed_noise_law.py')
    prior=ROOT/'artifacts/research/20261114/ripper/correlated_noise';previous=json.loads((prior/'audit.json').read_text())
    rows=[];hashes={};new_audits=[];pair=np.triu_indices(81,1);i,j=pair
    true=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    for seed in p['seeds']:
        for arm in ('iid','collapse'):
            d=prior/f's{seed}'/'iid' if arm=='iid' else out/f's{seed}'
            audit=json.loads((d/'audit.json').read_text());receipt=json.loads((d/'receipt.json').read_text())
            assert audit['status']=='validated' and audit['receipt_sha256']==sha(d/'receipt.json')
            if arm=='iid':
                for name in ('receipt.json','audit.json'):assert sha(d/name)==previous['artifact_sha256'][str((d/name).relative_to(ROOT))]
                assert audit['protocol_sha256']==previous['protocol_sha256']
            else:
                assert audit['protocol_sha256']==receipt['protocol_sha256']==sha(pp)
                assert audit['auditor_sha256']==sha(HERE/'audit_collapsed_noise_acquisition.py')
                assert receipt['collector_sha256']==sha(HERE/'collapsed_noise_acquire.py')
                assert audit['independent_masks_replayed']==1110 and audit['log_lr_values_verified']==3596400
                assert audit['first_crossing_paths_verified']==3240 and audit['decimal_threshold_checks']==1111
                new_audits.append(audit)
            for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
            model=np.load(d/'fit.npz');exposures=int(model['joint_count'][true].sum());equal=int(model['equal_count'][true].sum())
            if arm=='collapse':assert audit['true_pair_exposures']==exposures and audit['true_pair_equal_count']==equal
            fit=audit['calibration_fit']
            rows.append({'seed':seed,'arm':arm,'archived_control':arm=='iid','missing_true_edges':audit['missing_true_edges'],
                'extra_edges':audit['extra_edges'],'calibration_disagreements':fit['disagreements'],'eta_upper':fit['eta_upper'],
                'calibration_covers_nominal_eta':fit['postfit_covers_actual_eta'],'calibration_error_queries':audit['calibration_error_queries'],
                'training_error_queries':audit['training_error_queries'],'true_pair_exposures':exposures,'true_pair_equal_count':equal,
                'observed_true_pair_equal_rate':equal/exposures})
            for name in ('receipt.json','audit.json'):hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    grouped=[]
    for arm in ('iid','collapse'):
        part=[row for row in rows if row['arm']==arm];group={'arm':arm,'banks_with_missing_true_edges':sum(row['missing_true_edges']>0 for row in part)}
        for key in ('missing_true_edges','extra_edges','calibration_disagreements','calibration_error_queries','training_error_queries','true_pair_exposures','true_pair_equal_count'):
            group[key]=sum(row[key] for row in part)
        group.update(calibration_coverage_banks=sum(row['calibration_covers_nominal_eta'] for row in part),
            calibration_error_pct=group['calibration_error_queries']/640,training_error_pct=group['training_error_queries']/8880,
            observed_true_pair_equal_rate=group['true_pair_equal_count']/group['true_pair_exposures'],
            minimum_missing_per_bank=min(row['missing_true_edges'] for row in part),maximum_missing_per_bank=max(row['missing_true_edges'] for row in part))
        grouped.append(group)
    collapse=next(row for row in grouped if row['arm']=='collapse');lost=collapse['banks_with_missing_true_edges']
    assert collapse['calibration_disagreements']==0 and collapse['calibration_coverage_banks']==0
    primary={'decision':'supported_at_least_90percent_banks_lose_true_edges' if lost>=29 else 'falsified_at_least_90percent_banks_lose_true_edges',
        'banks_losing_true_edges':lost,'total_banks':32,'minimum_banks':29,'fraction':lost/32,
        'scope':'Exact finite-bank target on reused source banks under fresh collapse draws; no independent-edge confidence claim.'}
    report={'status':'validated','utc':now(),'primary':primary,'grouped':grouped,'rows':rows,'corruption_law_audit':law,
        'new_acquisition_audits':new_audits,'input_registration':registration,'source_manifest_sha256':sha(ROOT/p['inputs']),
        'new_source_boards':0,'new_solver_inputs':0,'gpu_seconds':0,'new_queries_per_bank':29750,'new_fits':32,'archived_iid_fits':32,
        'masks_replayed':35520,'log_lr_values_verified':115084800,'first_crossing_paths_verified':103680,
        'theorem_prediction_checks':{'zero_repeated_disagreements':True,'zero_calibration_coverage':True,
            'nominal_fixed_prequery_cell_error':'1/20','conditional_true_pair_equality':'9/160','conditional_identical_clean_pair_equality':'1'},
        'artifact_sha256':hashes,'protocol_sha256':sha(pp),'summarizer_sha256':sha(__file__),'limitations':p['limitations']}
    save(out/'audit.json',report);public=ROOT/'results/research/ripper';save(public/'collapsed_noise.json',report)
    with (public/'collapsed_noise_per_bank.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1am: destructive shared corruption defeats the repeated-query calibration assumption','',
        f'The frozen finite-bank target is **{primary["decision"]}**: {lost}/32 banks lose at least one true edge; at least29/32 was required.','',
        '| Noise model | Banks losing true edges | Missing true edges | Retained extras | Calibration coverage | Realized calibration error (%) | Realized adaptive training error (%) | Observed true-pair equality (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — | — |']
    for group in grouped:lines.append(f'| {group["arm"]} | {group["banks_with_missing_true_edges"]}/32 | {group["missing_true_edges"]} | {group["extra_edges"]} | {group["calibration_coverage_banks"]}/32 | {group["calibration_error_pct"]:.4f} | {group["training_error_pct"]:.4f} | {group["observed_true_pair_equal_rate"]*100:.4f} |')
    lines+=['','The collapse model draws one Bernoulli(9/160) event and one uniform output digit per board, after the training mask is finalized. On a bad board every query returns that digit. Both calibration repeats share the same event/output. For any fixed pre-query cell, marginal error is exactly(9/160)(8/9)=1/20. Realized errors on the adaptive query sequence are measured separately above; no draw is filtered or rescaled.', '',
        'Conditional on past history, a current clean board and its pre-noise mask, an unequal clean pair has equality probability9/160 under collapse; an equal clean pair stays equal with probability1. These exact law statements were independently checked for all81 clean digit pairs. Actual co-query exposures and equal counts were independently reconstructed for every row and compared with the saved counts.', '',
        f'Repeated calibration queries always agree, so the iid-based estimator produces eta_upper={law["count0_calibration_fit"]["eta_upper"]:.6f} and misses the nominal5% marginal error in all32 banks. The resulting learned true-pair null is {float(Fraction(law["learned_true_pair_null"])):.8f}, below the actual conditional9/160. Those calibration identities are separate from the empirical pruning outcome.', '',
        f'The collapse arm loses between{collapse["minimum_missing_per_bank"]} and{collapse["maximum_missing_per_bank"]} true edges per bank. This removes a premise of zero-energy Sudoku certificate soundness. No solver was run, so this study does not claim observed invalid certificates or solver accuracy.', '',
        'All32 H1al source banks are reused without selection, with their archived iid controls. Both acquisition methods use M1000/N1110,25 training queries per board and29,750 queries per fit, matched source/tie/cell streams and independent current-noise draws. No new source cohort, solver cohort or GPU allocation is used. New acquisition replay checks35,520 masks,115,084,800 LR values and103,680 first-crossing paths.', '',
        'H1al shared derangements preserved equality despite calibration failure; this explicitly nonbijective collapse law destroys inequalities. These are two specified correlation models, not a universal ordering of correlated noise difficulty. The frozen target is a finite-bank structural failure criterion.', '',
        '[Protocol](protocol-collapsed-noise.json) · [Corruption law](collapsed_noise_policy.py) · [Independent audit](audit_collapsed_noise_acquisition.py) · [Results](../../../results/research/ripper/collapsed_noise.json)','']
    (HERE/'REPORT-collapsed-noise.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped}),flush=True)


if __name__=='__main__':main()
