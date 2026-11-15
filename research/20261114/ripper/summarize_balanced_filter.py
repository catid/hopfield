"""Keep the finite-bank recovery target separate from filtering identities."""
import csv
import json
from run import ROOT,HERE,sha,save,now
from balanced_filter_common import paths,load_protocol


def main():
    out,pp=paths();p=load_protocol();rows=[];audits=[];hashes={};calibrations=[]
    source=json.loads((out/'input_audit.json').read_text());assert source['status']=='validated'
    for seed in p['seeds']:
        bank=out/f's{seed}';cr=json.loads((bank/'calibration_receipt.json').read_text())
        assert cr['estimator_calls']==1 and cr['shared_across_both_arms'];calibrations.append(cr)
        hashes[str((bank/'calibration_receipt.json').relative_to(ROOT))]=sha(bank/'calibration_receipt.json')
        paired=[]
        for arm in p['arms']:
            d=bank/arm;receipt=json.loads((d/'receipt.json').read_text());audit=json.loads((d/'audit.json').read_text())
            assert audit['status']=='validated' and audit['protocol_sha256']==sha(pp)
            assert audit['auditor_sha256']==sha(HERE/'audit_balanced_filter_acquisition.py') and audit['receipt_sha256']==sha(d/'receipt.json')
            assert audit['shared_calibration_receipt_sha256']==sha(bank/'calibration_receipt.json')
            assert audit['independent_masks_replayed']==1110 and audit['log_lr_values_verified']==3596400
            assert audit['first_crossing_paths_verified']==3240 and audit['decimal_threshold_checks']==1111
            for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
            fit=audit['calibration_fit'];row={key:audit[key] for key in ['seed','arm','missing_true_edges','extra_edges','exact_graph',
                'accepted_boards','rejected_boards','accepted_corrupt_boards','rejected_clean_boards','training_collapse_events',
                'training_error_queries','accepted_error_queries','accepted_queries','clean_subsequence_trajectory_rows_identical']}
            row.update(calibration_disagreements=fit['disagreements'],eta_upper=fit['eta_upper'],covers_nominal_eta=fit['postfit_covers_actual_eta'],
                calibration_error_queries=audit['calibration_error_queries'],acquisition_s=receipt['elapsed_s'],replay_s=audit['elapsed_s'])
            rows.append(row);audits.append(audit);paired.append(audit)
            for name in ('receipt.json','audit.json'):hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
        for key in ['calibration_fit','calibration_error_queries','calibration_collapse_events','training_collapse_events']:
            assert paired[0][key]==paired[1][key]
        # Potential corruptions are paired even though adaptive masks can differ.
        assert sha(bank/'unfiltered/potential_noise.npz')==sha(bank/'filtered/potential_noise.npz')
    grouped=[]
    total_keys=['missing_true_edges','extra_edges','exact_graph','accepted_boards','rejected_boards','accepted_corrupt_boards','rejected_clean_boards',
        'training_error_queries','accepted_error_queries','accepted_queries','training_collapse_events','clean_subsequence_trajectory_rows_identical']
    for arm in p['arms']:
        part=[r for r in rows if r['arm']==arm];totals={key:sum(r[key] for r in part) for key in total_keys}
        totals['exact_graph_banks']=totals.pop('exact_graph')
        grouped.append(dict(arm=arm,**totals,calibration_coverage_banks=sum(r['covers_nominal_eta'] for r in part),
            training_error_pct=100*totals['training_error_queries']/(32*27750),
            accepted_error_pct=100*totals['accepted_error_queries']/totals['accepted_queries']))
    filtered=next(r for r in grouped if r['arm']=='filtered')
    purity=filtered['accepted_corrupt_boards']==filtered['rejected_clean_boards']==0
    retention=filtered['missing_true_edges']==0;replay=filtered['clean_subsequence_trajectory_rows_identical']==35520
    assert purity and retention and replay
    primary={'target':'At least 29 of 32 filtered graphs are exact.','exact_filtered_banks':filtered['exact_graph_banks'],
        'decision':'supported_finite_bank_target' if filtered['exact_graph_banks']>=29 else 'falsified_finite_bank_target',
        'scope':'Exact preregistered finite-bank target. No independent-edge or population-confidence claim.'}
    manifest=json.loads((out/'input_manifest.json').read_text())
    result={'status':'validated','utc':now(),'protocol_sha256':sha(pp),'summarizer_sha256':sha(__file__),
        'primary':primary,'theorem_checks':{'filter_purity':purity,'zero_missing_true_edges':retention,'clean_subsequence_replay':replay},
        'theorem_scope':'Supplied digit-balance prior: no completed board has more than 9 copies of a digit. Every 25-query collapse is rejected; every clean subset is accepted. This restores clean pair evidence but does not guarantee enough samples for exact recovery or fix scalar noise calibration.',
        'grouped':grouped,'rows':rows,'input_audit':source,'acquisition_audits':audits,'paired_banks':32,'new_fits':64,'shared_calibrations':32,
        'queries_per_arm_including_shared_calibration':29750,'physical_queries_per_pair':57500,'physical_queries_total':32*57500,
        'training_masks_replayed':71040,'log_lr_values_verified':230169600,'first_crossing_paths_verified':207360,
        'clean_subsequence_trajectory_rows_verified':35520,'gpu_seconds':0,'new_solver_inputs':0,
        'all_source_manifest_sha256':sha(out/'all_source_manifest.json'),
        'future_source_exclusion_union':manifest['excluded_unique_source_rows']+source['independent_csv_answers_verified'],
        'artifact_sha256':hashes,'limitations':p['limitations']}
    save(out/'audit.json',result);public=ROOT/'results/research/ripper';save(public/'balanced_filter.json',result)
    with (public/'balanced_filter_per_bank.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1an: an observed-label balance filter under shared collapse','',
        f'The preregistered target of at least 29 exact filtered graphs is **{primary["decision"]}**: {filtered["exact_graph_banks"]}/32 fresh banks recovered the graph exactly.','',
        '| Arm | Exact graphs | Missing true edges | Retained extras | Rejected training boards | Accepted corrupt boards | Rejected clean boards |',
        '|---|---:|---:|---:|---:|---:|---:|','| BP AdamW: outside this source-only study; not run | — | — | — | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {g["exact_graph_banks"]}/32 | {g["missing_true_edges"]} | {g["extra_edges"]} | {g["rejected_boards"]} | {g["accepted_corrupt_boards"]} | {g["rejected_clean_boards"]} |')
    lines+=['',
        'The filter uses an explicitly supplied global Sudoku prior: each digit occurs at most nine times in a completed board. After all 25 labels are observed, the complete row is rejected if any observed digit count exceeds nine. The spent queries are charged; rejected rows update no pair evidence. No peer graph, unqueried value, or clean label enters filtering or mask selection.','',
        'Shared collapse draws one Bernoulli(9/160) event and a uniform output digit per board; all queries and calibration repeats share that draw. Fixed pre-query cells therefore have marginal error (9/160)(8/9) = 5%. Clean 25-cell subsets obey the balance prior; collapsed subsets contain 25 identical labels. Filter purity is a mathematical consequence for this law, separate from the empirical exact-recovery target.','',
        f'The independent replay checked every mask, observed label, likelihood ratio and first crossing in 64 acquisitions. All 35,520 filtered rows matched a separate clean accepted-subsequence state, including tie-RNG advancement on rejected rows and original row-indexed crossings. Filtered accepted data contain zero corrupt rows and zero label errors; no clean row was rejected and no true edge was pruned. The number of retained extras remains an empirical outcome.','',
        'Both arms spend 27,750 training queries per bank with M1000/N1110 and 25 labels per training board. They share one 2,000-query calibration fit, so there are 32 calibrations and 57,500 physical queries per paired bank, not 64 independent calibrations. Both arms have identical clean sources, potential collapse draws and tie streams; adaptive masks may diverge through history.','',
        f'All {source["independent_csv_answers_verified"]:,} new distinct source boards were independently checked against the source CSV after excluding 1,862,070 prior boards. Source roles are globally disjoint. All 32 paired banks and every outcome are retained. This source-only study uses no solver cohort or GPU.','',
        'Calibration repeats still agree under collapse. The nominal scalar noise estimate remains below 5% in every bank; the filter repairs observed pair evidence for this specific corruption law, not calibration coverage. Its supplied balance prior need not hold in other tasks, and other correlated corruptions may evade the filter.','',
        '[Protocol](protocol-balanced-filter.json) · [Filter](balanced_filter_policy.py) · [Independent replay](audit_balanced_filter_acquisition.py) · [Results](../../../results/research/ripper/balanced_filter.json)','']
    (HERE/'REPORT-balanced-filter.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'future_source_exclusion_union':result['future_source_exclusion_union']}),flush=True)


if __name__=='__main__':main()
