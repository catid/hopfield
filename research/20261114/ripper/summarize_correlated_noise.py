"""Separate finite-bank structural result from the corruption invariance theorem."""
import csv
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from correlated_noise_common import paths, load_protocol


def main():
    out,pp=paths();p=load_protocol();source=json.loads((out/'input_audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(pp)
    assert source['wrapper_sha256']==sha(HERE/'audit_correlated_noise_inputs.py')
    assert source['calibration_training_overlap']==source['within_bank_overlap']==source['prior_bank_overlap']==0
    lawpath=ROOT/'artifacts/research/20261114/ripper/correlated_noise_law.json';law=json.loads(lawpath.read_text())
    assert law['status']=='validated' and law['auditor_sha256']==sha(HERE/'test_correlated_noise_law.py')
    assert law['derangements']==133496 and law['off_diagonal_digit_mapping_count']==16687
    rows=[];hashes={};audits=[]
    for seed in p['seeds']:
        for arm in p['arms']:
            d=out/f's{seed}'/arm;audit=json.loads((d/'audit.json').read_text());receipt=json.loads((d/'receipt.json').read_text())
            assert audit['status']=='validated' and audit['protocol_sha256']==receipt['protocol_sha256']==sha(pp)
            assert audit['auditor_sha256']==sha(HERE/'audit_correlated_noise_acquisition.py')
            assert receipt['collector_sha256']==sha(HERE/'correlated_noise_acquire.py')
            assert audit['receipt_sha256']==sha(d/'receipt.json')
            for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
            assert audit['independent_masks_replayed']==1110 and audit['log_lr_values_verified']==3596400
            assert audit['first_crossing_paths_verified']==3240 and audit['decimal_threshold_checks']==1111
            fit=audit['calibration_fit'];row={'seed':seed,'arm':arm,'missing_true_edges':audit['missing_true_edges'],'extra_edges':audit['extra_edges'],
                'calibration_disagreements':fit['disagreements'],'eta_upper':fit['eta_upper'],'covers_nominal_eta':fit['postfit_covers_actual_eta'],
                'fallback':fit['no_pruning_fallback'],'calibration_error_queries':audit['calibration_error_queries'],'training_error_queries':audit['training_error_queries'],
                'clean_trajectory_rows_identical':audit['clean_corrupted_trajectory_rows_identical'],'acquisition_s':receipt['elapsed_s'],'replay_s':audit['elapsed_s']}
            rows.append(row);audits.append(audit)
            for name in ('receipt.json','audit.json'):hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    grouped=[]
    for arm in p['arms']:
        part=[r for r in rows if r['arm']==arm]
        grouped.append({'arm':arm,'missing_true_edges':sum(r['missing_true_edges'] for r in part),'extra_edges':sum(r['extra_edges'] for r in part),
            'calibration_coverage_banks':sum(r['covers_nominal_eta'] for r in part),'fallback_banks':sum(r['fallback'] for r in part),
            'calibration_disagreements':sum(r['calibration_disagreements'] for r in part),
            'calibration_error_queries':sum(r['calibration_error_queries'] for r in part),'training_error_queries':sum(r['training_error_queries'] for r in part),
            'calibration_error_pct':sum(r['calibration_error_queries'] for r in part)/640,
            'training_error_pct':sum(r['training_error_queries'] for r in part)/8880,
            'clean_trajectory_rows_identical':sum(r['clean_trajectory_rows_identical'] for r in part)})
    iid=next(r for r in grouped if r['arm']=='iid');shared=next(r for r in grouped if r['arm']=='shared')
    extra_target=10*shared['extra_edges']<=iid['extra_edges'];retention=all(r['missing_true_edges']==0 for r in grouped)
    decision='supported_finite_bank_conjunction' if extra_target and retention else 'falsified_finite_bank_conjunction'
    predicted=shared['calibration_coverage_banks']==0 and shared['calibration_disagreements']==0 and shared['clean_trajectory_rows_identical']==35520
    assert predicted
    reduction=100*(1-shared['extra_edges']/iid['extra_edges']) if iid['extra_edges'] else None
    manifest=json.loads((out/'input_manifest.json').read_text())
    result={'status':'validated','utc':now(),'primary':{'decision':decision,'shared_extras_at_most_10percent_iid':extra_target,
        'zero_missing_true_edges_both_arms':retention,'shared_to_iid_extra_ratio':shared['extra_edges']/iid['extra_edges'] if iid['extra_edges'] else None,
        'extra_edge_reduction_pct':reduction,'scope':'Exact finite-bank descriptive target; no independent-edge sampling, confidence interval or universal robustness claim.'},
        'theorem_predictions_verified':predicted,'theorem_scope':'Uniform shared derangement preserves equality for every board. Shared repeats always agree. The iid calibration family is violated; its upper bound need not cover marginal digit error. This says nothing about all correlated noise.',
        'grouped':grouped,'rows':rows,'corruption_law_audit':law,'input_audit':source,'acquisition_audits':audits,
        'queries_per_arm_per_bank':29750,'paired_banks':32,'training_masks_replayed':71040,'log_lr_values_verified':230169600,
        'first_crossing_paths_verified':207360,'whole_clean_corrupted_trajectory_rows_verified':35520,
        'new_solver_inputs':0,'gpu_seconds':0,'all_source_manifest_sha256':sha(out/'all_source_manifest.json'),
        'future_source_exclusion_union':manifest['excluded_unique_source_rows']+source['independent_csv_answers_verified'],
        'artifact_sha256':hashes,'protocol_sha256':sha(pp),'summarizer_sha256':sha(__file__),'limitations':p['limitations']}
    save(out/'audit.json',result);public=ROOT/'results/research/ripper';save(public/'correlated_noise.json',result)
    with (public/'correlated_noise_per_bank.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1al: shared corruption can break calibration while preserving constraints','',
        f'The frozen finite-bank structural conjunction is **{decision}**: shared retained extras must be at most10% of iid extras, and both arms must retain every true edge.','',
        '| Corruption model | Retained extras | Missing true edges | Calibration coverage | Calibration disagreements | Realized calibration error (%) | Realized training error (%) |',
        '|---|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {g["extra_edges"]} | {g["missing_true_edges"]} | {g["calibration_coverage_banks"]}/32 | {g["calibration_disagreements"]} | {g["calibration_error_pct"]:.4f} | {g["training_error_pct"]:.4f} |')
    lines+=['',f'The observed pooled extra-edge reduction is {reduction:.4f}%. Counts are dependent pair-bank totals, not independent edge trials. This empirical target is separate from the mathematical invariance checks.','',
        'Both models have exactly5% marginal digit error and a uniform wrong digit among eight alternatives. The shared model draws one Bernoulli(1/20) corruption event and one uniform derangement of nine digits per board. Both repeats on a calibration board share that draw; all training queries on a board share its draw. Rejection sampling of uniform permutations produces uniform derangements. Exhaustive enumeration checked all133,496 derangements and16,687 mappings for every ordered pair of distinct digits. A merely nonidentity permutation would not preserve the requested5% marginal error.','',
        f'Every shared bijection preserves pair equality. Shared calibration repeats therefore disagree zero times; the iid-based estimator returns eta_upper={law["zero_disagreement_calibration_fit"]["eta_upper"]:.6f}, below the true marginal0.05, in every bank. This coverage failure was predicted mathematically. It does not imply unsafe edge pruning here: true-edge equality stays exactlyzero under the shared corruption.','',
        'The adaptive acquisition audit independently replayed each history-only mask before reconstructing current noise. For all35,520 shared training rows, the complete clean and corrupted query trajectories, joint/equal counts, exclusions and first crossings agreed exactly at the same fitted threshold. Unqueried/future values never enter the mask selector. Clean labels enter only this postfit reference audit.','',
        f'All32 paired banks use identical clean calibration/training sources across arms, M1000/N1110,25 queries per training board and29,750 queries per arm. Source roles are globally disjoint; all1,795,591 prior source rows were excluded. {source["independent_csv_answers_verified"]:,} distinct new boards were independently verified. Every draw is retained. No solver input, trajectory or GPU allocation was used.','',
        'This is a constraint-preserving noise family. It demonstrates that failed scalar noise calibration and successful relational learning can coexist; it does not show that correlation is generally easier or that the original iid calibration theorem applies.','',
        '[Protocol](protocol-correlated-noise.json) · [Corruption law](correlated_noise_policy.py) · [Whole-trajectory audit](audit_correlated_noise_acquisition.py) · [Results](../../../results/research/ripper/correlated_noise.json)','']
    (HERE/'REPORT-correlated-noise.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':result['primary'],'grouped':grouped,'future_source_exclusion_union':result['future_source_exclusion_union']}),flush=True)


if __name__=='__main__':main()
