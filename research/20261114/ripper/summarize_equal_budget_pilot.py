"""Audit-only old-board implementation pilot, never pooled with H1ab."""
import json
from run import ROOT,HERE,sha,save,now
from equal_budget_common import paths,load_protocol


def main():
    out,pp=paths(True);p=load_protocol(True);ia=json.loads((out/'input_audit.json').read_text());assert ia['status']=='validated'
    rows=[];cal=[];receipts=[]
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated' and a['wrapper_sha256']==sha(HERE/'audit_equal_budget_acquisition.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['protocol_sha256']==r['protocol_sha256']==sha(pp) and a['independent_masks_replayed']==8320
        assert a['log_lr_values_verified']==26956800 and a['decimal_threshold_checks']==8324
        rows.extend(a['rows']);cal.extend(dict(seed=seed,**c) for c in a['calibration_audit']['fits'])
        receipts.append({'seed':seed,'audit_sha256':sha(d/'audit.json'),'receipt_sha256':sha(d/'receipt.json')})
    report={'status':'validated','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'scope':p['scope'],'no_solver_inference':True,
        'independent_masks_replayed':49920,'log_lr_values_verified':161740800,'calibration_pairs_audited':60000,
        'rows':rows,'calibration':cal,'input_audit':ia,'receipts':receipts,'label_costs':p['label_costs'],'not_pooled':True}
    save(out/'audit.json',report);save(ROOT/'results/research/ripper/equal_budget_pilot.json',report)
    lines=['# H1ab equal-budget implementation pilot','',
       'The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.','',
       'All 49,920 masks, 161,740,800 log-LR values and 60,000 nested calibration pairs were checked. Every arm stops querying at its declared N and receives exactly 57,500 total noisy labels.','']
    for arm in ['learned_small','learned_large']:
        rr=[r for r in cal if r['arm']==arm];lines.append(f'{arm}: upper bounds {min(r["eta_upper"] for r in rr)*100:.4f}%–{max(r["eta_upper"] for r in rr)*100:.4f}%; coverage {sum(r["covers_actual_eta"] for r in rr)}/6; fallback {sum(r["no_pruning_fallback"] for r in rr)}. Diagnostics do not select or change models.')
    lines+=['','All allocations, the 5 pp primary margin, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.','']
    (HERE/'REPORT-equal-budget-pilot.md').write_text('\n'.join(lines));print(json.dumps({'status':'validated','banks':6,'masks':49920}))


if __name__=='__main__':main()
