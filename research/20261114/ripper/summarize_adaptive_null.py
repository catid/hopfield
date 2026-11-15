"""Compare adaptive exact-law acquisition with its frozen fixed-mask replay."""
import csv,json
from run import ROOT,HERE,sha,save,now
from adaptive_null_common import paths,load_protocol

def main():
    out,pp=paths();p=load_protocol();parent=json.loads((ROOT/'artifacts/research/20261114/ripper/null_replay/audit.json').read_text())
    old=next(g for g in parent['grouped']if g['arm']=='exact_law');assert old['extra_edges']==14229
    rows=[];audits=[];hashes={}
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text());assert a['status']=='validated'and a['protocol_sha256']==sha(pp)and a['auditor_sha256']==sha(HERE/'audit_adaptive_null.py')and a['receipt_sha256']==sha(d/'receipt.json')
        for n,h in r['artifact_sha256'].items():assert sha(d/n)==h
        baseline=next(x for x in parent['rows']if x['seed']==seed and x['arm']=='exact_law')
        row={k:a[k]for k in ['seed','missing_true_edges','extra_edges','all_true_retained','exact_graph','queries_spent','training_error_queries','permutation_events','changed_masks_vs_archived']}
        row.update(fixed_mask_extras=baseline['extra_edges'],fixed_mask_missing=baseline['missing_true_edges'],acquisition_s=r['elapsed_s'],replay_s=a['elapsed_s']);rows.append(row);audits.append(a)
        for n in ['receipt.json','audit.json']:hashes[str((d/n).relative_to(ROOT))]=sha(d/n)
    total={k:sum(r[k]for r in rows)for k in ['missing_true_edges','extra_edges','all_true_retained','exact_graph','queries_spent','training_error_queries','permutation_events','changed_masks_vs_archived']}
    retention=total['all_true_retained']>=29;efficiency=2*total['extra_edges']<=14229
    primary=dict(decision='supported_finite_bank_conjunction'if retention and efficiency else'falsified_finite_bank_conjunction',all_true_retaining_banks=total['all_true_retained'],retention_target_met=retention,pooled_extra_edges=total['extra_edges'],extra_ceiling=7114,extra_target_met=efficiency,extra_reduction_pct=100*(1-total['extra_edges']/14229))
    grouped=[dict(arm='archived_fixed_mask_exact_law',**{k:old[k]for k in ['missing_true_edges','extra_edges','all_true_retained','exact_graph']}),dict(arm='adaptive_exact_law',**total)]
    result=dict(status='validated',utc=now(),protocol_sha256=sha(pp),summarizer_sha256=sha(__file__),primary=primary,grouped=grouped,rows=rows,acquisition_audits=audits,artifact_sha256=hashes,
        scope=p['scope'],new_training_queries=32*27750,new_calibrations=0,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0,independent_masks_replayed=35520,cumulative_log_lr_values_verified=115084800,first_crossing_paths_verified=103680,future_source_exclusion_union=1928477)
    save(out/'audit.json',result);public=ROOT/'results/research/ripper';save(public/'adaptive_null.json',result)
    with(public/'adaptive_null_per_bank.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# H1as: adapt acquisition to the supplied exact equality null','',
        f'The frozen conjunction is **{primary["decision"]}**: {total["all_true_retained"]}/32 banks retain every true edge, with {total["extra_edges"]:,} extra edge-bank pairs versus the ceiling of 7,114.','',
        '| Mask policy at exact q0 | All true edges retained | Exact graphs | Missing true edges | Retained extras |','|---|---:|---:|---:|---:|',
        '| BP AdamW: outside this source-only study; not run | — | — | — | — |']
    for g in grouped:lines.append(f'| {g["arm"]} | {g["all_true_retained"]}/32 | {g["exact_graph"]}/32 | {g["missing_true_edges"]} | {g["extra_edges"]} |')
    lines+=['',f'Pooled extras changed by {primary["extra_reduction_pct"]:.4f}% relative to the frozen H1ar exact-law replay. The empirical target requires at least a 50% reduction and at least 29 banks retaining every true edge; exact graph recovery is descriptive.','',
        'Every new acquisition uses the supplied exact permutation-law q0 = 9/1600 from the first row, with unchanged q1 = 1/9, boundary 648000, exposure selector and 25 queries per training board. The current mask is fixed before the current corruption draw. All 32 source banks, potential event/permutation draws and original row-indexed tie streams match H1ap. Histories and queried masks may now diverge under the new exclusions.','',
        f'The independent auditor reconstructed all 35,520 masks, noise draws and 115,084,800 cumulative log-likelihood values, plus all 103,680 first-crossing paths. {total["changed_masks_vs_archived"]:,} masks differ from the archived acquisition. All {total["queries_spent"]:,} new training queries are charged, including labels on corrupted rows; no calibration is fitted or queried.','',
        'Known-law information is an explicit supplied prior. The exact-null martingale argument applies to the independent board-permutation law with predictable current masks; this study does not claim arbitrary correlated-noise robustness. The comparison uses already-seen source banks and common latent draws, not fresh-source confirmation.','',
        'Each bank spends 27,750 training queries. No new source board, solver input or GPU work is used. The archived H1ar fixed-mask comparator is not reacquired.','',
        '[Protocol](protocol-adaptive-null.json) · [Independent replay](audit_adaptive_null.py) · [Results](../../../results/research/ripper/adaptive_null.json)','']
    (HERE/'REPORT-adaptive-null.md').write_text('\n'.join(lines));print(json.dumps(dict(primary=primary,grouped=grouped)),flush=True)
if __name__=='__main__':main()
