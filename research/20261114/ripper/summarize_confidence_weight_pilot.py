"""Keep the six implementation pilots separate from fresh inference."""
import json
from run import ROOT,HERE,sha,save,now


def main():
    out=ROOT/'artifacts/research/20261114/ripper/confidence_weight_pilot';rows=[]
    assert json.loads((out/'input_audit.json').read_text())['status']=='validated'
    for seed in range(6):
        a=json.loads((out/f's{seed}/audit.json').read_text());assert a['status']=='validated'
        assert a['independent_masks_replayed']==2220 and a['log_lr_values_verified']==7192800 and a['weight_models_verified']==3
        rows.append({'seed':seed,'extra_edges':a['extra_edges'],'missing_true_edges':a['missing_true_edges'],'models':a['models'],'calibration':a['calibration_fit'],'replay_s':a['elapsed_s']})
    report={'status':'validated','utc':now(),'scope':'Six old-board implementation pilots; no inference, no pooling or selection.',
        'rows':rows,'independent_masks_replayed':13320,'log_lr_values_verified':43156800,'weight_models_verified':18,
        'protocol_sha256':sha(HERE/'protocol-confidence-weight-pilot.json'),'script_sha256':sha(__file__)}
    save(ROOT/'results/research/ripper/confidence_weight_pilot.json',report)
    (HERE/'REPORT-confidence-weight-pilot.md').write_text('# H1ah implementation pilot\n\nAll six old-board pilots passed independent source, calibration, mask/noise/path and exact weight/permutation audits. This covers 13,320 masks, 43,156,800 log-LR values and 18 weight assignments. There was no solver inference and no pooling with the fresh study.\n')
    print({'status':'validated','banks':6,'masks':13320,'weight_models':18})


if __name__=='__main__':main()
