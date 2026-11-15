"""Separate exact filter blindness from finite-bank constraint loss."""
import csv,json
from run import ROOT,HERE,sha,save,now
from permuted_noise_common import paths,load_protocol

def main():
    out,pp=paths();p=load_protocol();rows=[];audits=[];hashes={}
    lawpath=ROOT/'artifacts/research/20261114/ripper/permuted_noise_law_audit.json';law=json.loads(lawpath.read_text())
    assert law['status']=='validated'and law['auditor_sha256']==sha(HERE/'test_permuted_noise_law.py')
    parent=json.loads((ROOT/'artifacts/research/20261114/ripper/balanced_filter/audit.json').read_text())
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated'and a['protocol_sha256']==sha(pp)and a['auditor_sha256']==sha(HERE/'audit_permuted_noise_acquisition.py')
        assert a['receipt_sha256']==sha(d/'receipt.json')
        for name,digest in r['artifact_sha256'].items():assert sha(d/name)==digest
        fit=a['calibration_fit'];row={k:a[k]for k in ['seed','arm','missing_true_edges','extra_edges','calibration_error_queries','training_error_queries','training_permutation_events','accepted_corrupt_boards','rejected_boards','unfiltered_replay_rows_identical','true_pair_exposures','true_pair_equal_count']}
        row.update(calibration_disagreements=fit['disagreements'],eta_upper=fit['eta_upper'],covers_nominal_eta=fit['postfit_covers_actual_eta'],acquisition_s=r['elapsed_s'],replay_s=a['elapsed_s'])
        rows.append(row);audits.append(a)
        for name in ['receipt.json','audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    keys=['missing_true_edges','extra_edges','calibration_error_queries','training_error_queries','training_permutation_events','accepted_corrupt_boards','rejected_boards','unfiltered_replay_rows_identical','true_pair_exposures','true_pair_equal_count','calibration_disagreements']
    total={k:sum(r[k]for r in rows)for k in keys};total.update(arm='permuted_filtered',banks_with_missing_true_edges=sum(r['missing_true_edges']>0 for r in rows),calibration_coverage_banks=sum(r['covers_nominal_eta']for r in rows),training_error_pct=sum(r['training_error_queries']for r in rows)/8880,calibration_error_pct=sum(r['calibration_error_queries']for r in rows)/640)
    assert total['rejected_boards']==total['calibration_disagreements']==total['calibration_coverage_banks']==0
    assert total['accepted_corrupt_boards']==total['training_permutation_events']and total['unfiltered_replay_rows_identical']==35520
    primary=dict(target='At least16/32 banks lose at least one true edge under histogram-preserving permutation noise.',banks_with_missing_true_edges=total['banks_with_missing_true_edges'],decision='supported_finite_bank_target'if total['banks_with_missing_true_edges']>=16 else'falsified_finite_bank_target')
    old=next(g for g in parent['grouped']if g['arm']=='filtered')
    result=dict(status='validated',utc=now(),protocol_sha256=sha(pp),summarizer_sha256=sha(__file__),primary=primary,grouped=[dict(old,arm='archived_collapse_filtered'),total],rows=rows,acquisition_audits=audits,corruption_law_audit=law,
        theorem_checks=dict(filter_accepts_every_row=True,calibration_repeats_agree=True,filtered_unfiltered_trajectories_identical=True),
        theorem_scope='Uniform position permutations conserve nine copies of each digit; every25-cell subset passes the balance filter. Every repeated query on a board shares one output. This is mathematically separate from finite-bank pruning failures.',
        source_banks=32,new_calibrations=32,new_acquisitions=32,queries_per_new_bank=29750,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0,
        source_scope='Same already-seen H1an sources and tie/cell streams; new permutation-noise streams. Archived collapse results are descriptive controls. The unfiltered trajectory is independently replayed from the identical observed stream, not redundantly reacquired.',
        future_source_exclusion_union=1928477,artifact_sha256=hashes,parent_audit_sha256=sha(ROOT/'artifacts/research/20261114/ripper/balanced_filter/audit.json'),limitations=p['limitations'])
    save(out/'audit.json',result);public=ROOT/'results/research/ripper';save(public/'permuted_noise.json',result)
    with (public/'permuted_noise_per_bank.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1ap: a histogram-preserving corruption evades the balance filter','',
        f'The frozen target of at least 16 banks losing a true edge is **{primary["decision"]}**: {total["banks_with_missing_true_edges"]}/32 banks lose edges.','',
        '| Corruption and filter | Missing true edges | Retained extras | Rejected training rows | Accepted corrupt rows |','|---|---:|---:|---:|---:|',
        '| BP AdamW: outside this source-only study; not run | — | — | — | — |',
        f'| Archived H1an collapse + balance filter | {old["missing_true_edges"]} | {old["extra_edges"]} | {old["rejected_boards"]} | {old["accepted_corrupt_boards"]} |',
        f'| Uniform position permutation + balance filter | {total["missing_true_edges"]} | {total["extra_edges"]} | {total["rejected_boards"]} | {total["accepted_corrupt_boards"]} |','',
        'Each board draws one Bernoulli(9/160) corruption event and one uniform permutation of its 81 cell positions. All current queries and repeated calibration labels share that shuffled board. The current adaptive mask is fixed before either draw. Every digit still occurs exactly nine times, so no 25-label subset can violate the supplied digit-balance prior. The filter accepts every row.','',
        'The exact fixed-cell marginal error is (9/160)(8/9) = 5%; each wrong digit has probability 1/160. A uniform permutation maps two distinct queried positions uniformly to 6,480 ordered source-position pairs, of which 648 have equal labels. Thus true-pair equality is (9/160)(1/10) = 9/1600 = 0.5625%, while calibration repeats always agree. These laws were checked independently using exact fractions and exhaustive ordered-pair counts, not factorial permutation enumeration.','',
        f'All 35,520 adaptive masks, noise draws, observed labels, log-likelihood rows and first crossings were independently replayed. Every permuted board preserved its full digit histogram. An independent replay without filtering matched the complete filtered state on every row. All {total["accepted_corrupt_boards"]:,} corrupted training rows passed the filter; no row was rejected.','',
        f'The observed adaptive-query error was {total["training_error_pct"]:.4f}%; realized true-pair equality was {100*total["true_pair_equal_count"]/total["true_pair_exposures"]:.4f}%. These measurements are distinct from the exact fixed-cell and conditional-pair laws. The iid-based calibration estimate remains below nominal 5% in all 32 banks.','',
        'This is a prospective corruption-law boundary test on already-seen H1an source banks, with new observations and frozen noise streams. Each new fit uses M1000/N1110 and 29,750 queries. There are 32 new calibrations/acquisitions; the identical unfiltered replay is an audit, not another independent experiment. No new source board, solver input or GPU work is used.','',
        'The supplied multiplicity prior can detect shared collapse, but it cannot certify cleanliness under arbitrary shared corruption. This study measures graph damage only; it does not measure invalid solver certificates.','',
        '[Protocol](protocol-permuted-noise.json) · [Exact law check](test_permuted_noise_law.py) · [Independent replay](audit_permuted_noise_acquisition.py) · [Results](../../../results/research/ripper/permuted_noise.json)','']
    (HERE/'REPORT-permuted-noise.md').write_text('\n'.join(lines));print(json.dumps(dict(primary=primary,grouped=result['grouped'])))
if __name__=='__main__':main()
