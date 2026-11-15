"""Export the fixed implementation pilot without pooling it with fresh evidence."""
import json
from run import ROOT,HERE,sha,save,now
from calibrated_common import paths,load_protocol


def main():
    out,pp=paths(True);p=load_protocol(True);ia=json.loads((out/'input_audit.json').read_text());assert ia['status']=='validated'
    rows=[];cal=[];receipts=[]
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated' and a['wrapper_sha256']==sha(HERE/'audit_calibrated_acquisition.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['protocol_sha256']==r['protocol_sha256']==sha(pp) and a['independent_masks_replayed']==4500
        assert a['log_lr_values_verified']==14580000 and a['decimal_threshold_checks']==4503
        rows.extend(a['rows']);cal.append(dict(seed=seed,**a['calibration_audit']));receipts.append({'seed':seed,'audit_sha256':sha(d/'audit.json'),'receipt_sha256':sha(d/'receipt.json')})
    report={'status':'validated','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'scope':p['scope'],'no_solver_inference':True,
        'independent_masks_replayed':27000,'log_lr_values_verified':87480000,'calibration_pairs_audited':60000,
        'rows':rows,'calibration':cal,'input_audit':ia,'receipts':receipts,'label_costs':p['label_costs'],'not_pooled':True}
    save(out/'audit.json',report);save(ROOT/'results/research/ripper/calibrated_pilot.json',report)
    lines=['# H1aa repeated-query calibration: implementation pilot','',
       'The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, and support audits. It contains no solver inference and is not pooled with fresh evidence.',
       '',f'All 27,000 masks, 87,480,000 log-LR values and 60,000 repeated-query pairs were checked. Learned bounds span {min(c["eta_upper"] for c in cal)*100:.4f}% to {max(c["eta_upper"] for c in cal)*100:.4f}%; all six cover the true 1% rate, with no fallback. Coverage diagnostics do not select or modify models.',
       '', 'M10000, N1500, 5 pp noninferiority, streams, error budgets and 32 fresh banks were fixed before this pilot. The learned arm costs 57,500 queries per bank; each known-parameter control costs 37,500. No outcome-dependent setting changes were made.','']
    (HERE/'REPORT-calibrated-pilot.md').write_text('\n'.join(lines));print(json.dumps({'status':'validated','banks':6,'masks':27000}))


if __name__=='__main__':main()
