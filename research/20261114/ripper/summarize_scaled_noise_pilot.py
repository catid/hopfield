"""Audit-only old-board implementation pilot, never pooled with H1ad."""
import json
from run import ROOT,HERE,sha,save,now
from scaled_noise_common import paths,load_protocol


def main():
    out,pp=paths(True);p=load_protocol(True);ia=json.loads((out/'input_audit.json').read_text());assert ia['status']=='validated'
    rows=[];cal=[];receipts=[]
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated' and a['wrapper_sha256']==sha(HERE/'audit_scaled_noise_acquisition.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['protocol_sha256']==r['protocol_sha256']==sha(pp) and a['independent_masks_replayed']==13320
        assert a['log_lr_values_verified']==43156800 and a['decimal_threshold_checks']==13324
        rows.extend(a['rows']);cal.extend(dict(seed=seed,**c) for c in a['calibration_audit']['fits'])
        receipts.append({'seed':seed,'audit_sha256':sha(d/'audit.json'),'receipt_sha256':sha(d/'receipt.json')})
    report={'status':'validated','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'scope':p['scope'],'no_solver_inference':True,
        'independent_masks_replayed':79920,'log_lr_values_verified':258940800,'calibration_pairs_audited':6000,
        'rows':rows,'calibration':cal,'input_audit':ia,'receipts':receipts,'label_costs':p['label_costs'],'not_pooled':True}
    save(out/'audit.json',report);save(ROOT/'results/research/ripper/scaled_noise_pilot.json',report)
    lines=['# H1ad scaled-noise implementation pilot','',
       'The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.','',
       'All 79,920 masks, 258,940,800 log-LR values and 6,000 calibration boards at fixed 5% noise were checked. Short trajectories were verified as literal prefixes of the long trajectories. Each learned model pays 2,000 calibration queries in addition to its training queries.','']
    for arm in ['learned_short','learned_long']:
        rr=[r for r in cal if r['arm']==arm];lines.append(f'{arm}: upper bounds {min(r["eta_upper"] for r in rr)*100:.4f}%–{max(r["eta_upper"] for r in rr)*100:.4f}%; coverage {sum(r["covers_actual_eta"] for r in rr)}/6; fallback {sum(r["no_pruning_fallback"] for r in rr)}. Diagnostics do not select or change models.')
    lines+=['','All allocations, the 5 pp margin, 70% floor, noise rates, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.','']
    (HERE/'REPORT-scaled-noise-pilot.md').write_text('\n'.join(lines));print(json.dumps({'status':'validated','banks':6,'masks':79920}))


if __name__=='__main__':main()
