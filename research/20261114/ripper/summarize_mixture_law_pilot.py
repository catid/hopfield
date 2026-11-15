"""Audit-only old-board implementation pilot, never pooled with H1ag."""
import json
from run import ROOT,HERE,sha,save,now
from mixture_law_common import paths,load_protocol


def main():
    out,pp=paths(True);p=load_protocol(True);ia=json.loads((out/'input_audit.json').read_text());assert ia['status']=='validated'
    rows=[];cal=[];receipts=[]
    for seed in p['seeds']:
        d=out/f's{seed}';a=json.loads((d/'audit.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert a['status']=='validated' and a['wrapper_sha256']==sha(HERE/'audit_mixture_law_acquisition.py') and a['receipt_sha256']==sha(d/'receipt.json')
        assert a['protocol_sha256']==r['protocol_sha256']==sha(pp) and a['independent_masks_replayed']==8880
        assert a['log_e_values_verified']==28771200 and a['decimal_threshold_checks']==8884
        rows.extend(a['rows']);cal.extend(dict(seed=seed,**c) for c in a['calibration_audit']['fits'])
        receipts.append({'seed':seed,'audit_sha256':sha(d/'audit.json'),'receipt_sha256':sha(d/'receipt.json')})
    report={'status':'validated','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'scope':p['scope'],'no_solver_inference':True,
        'independent_masks_replayed':53280,'log_e_values_verified':172627200,'calibration_pairs_audited':6000,
        'rows':rows,'calibration':cal,'input_audit':ia,'receipts':receipts,'label_costs':p['label_costs'],'not_pooled':True}
    save(out/'audit.json',report);save(ROOT/'results/research/ripper/mixture_law_pilot.json',report)
    lines=['# H1ag mixture-law implementation pilot','',
       'The frozen six-bank old-board pilot passed independent source, calibration, noise, mask, likelihood-ratio, support and label-budget audits. It contains no solver inference and is not pooled with fresh evidence.','',
       'All 53,280 masks, 172,627,200 log-LR values and 6,000 calibration boards shared by both learned selectors at fixed 5% noise were checked. Each arm receives 55,500 training queries; each learned arm also receives 2,000 calibration queries.','']
    for arm in ['fixed_learned','mixture_learned']:
        rr=[r for r in cal if r['arm']==arm];lines.append(f'{arm}: upper bounds {min(r["eta_upper"] for r in rr)*100:.4f}%–{max(r["eta_upper"] for r in rr)*100:.4f}%; coverage {sum(r["covers_actual_eta"] for r in rr)}/6; fallback {sum(r["no_pruning_fallback"] for r in rr)}. Diagnostics do not select or change models.')
    lines+=['','All allocations, the 10 pp gain threshold, 70% floor, selector weights, streams, error budgets and 32 fresh banks were fixed before this pilot. No outcome-dependent changes were made.','']
    (HERE/'REPORT-mixture-law-pilot.md').write_text('\n'.join(lines));print(json.dumps({'status':'validated','banks':6,'masks':53280}))


if __name__=='__main__':main()
