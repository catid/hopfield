"""Compare all scientific audit outputs while preserving original receipts."""
import json
from run import ROOT,HERE,sha,save,now


def main():
    out=ROOT/'artifacts/research/20261114/ripper/scaled_noise_pilot/s0';original=out/'audit.json';optimized=out/'audit-fast-readonly.json'
    a=json.loads(original.read_text());b=json.loads(optimized.read_text());assert a['status']==b['status']=='validated'
    assert a['wrapper_sha256']==sha(HERE/'audit_scaled_noise_acquisition.py')
    assert b['wrapper_sha256']==sha(HERE/'audit_scaled_noise_fast_readonly.py')
    assert b['cached_noise_readonly_and_unchanged'] and b['potential_noise_cached_once']
    assert b['fast_mask_helper_sha256']==sha(HERE/'fast_reference_mask.py')
    compared=[]
    for key,value in a.items():
        if key in ['utc','wrapper_sha256','elapsed_s']:continue
        assert value==b[key],key;compared.append(key)
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'original_audit_sha256':sha(original),'optimized_audit_sha256':sha(optimized),
        'original_implementation_sha256':a['wrapper_sha256'],'optimized_implementation_sha256':b['wrapper_sha256'],'helper_sha256':b['fast_mask_helper_sha256'],
        'compared_fields':compared,'masks_verified':b['independent_masks_replayed'],'log_lr_values_verified':b['log_lr_values_verified'],
        'original_observed_wall_s':a['elapsed_s'],'optimized_observed_wall_s':b['elapsed_s'],'observed_ratio':a['elapsed_s']/b['elapsed_s'],
        'timing_scope':'Same complete pilotseed0 outputs and frozen data. Original ran with sixparallelCPUaudits; optimized ran during fourGPUworkers. This is observed walltime,not a controlled equal-loadthroughput comparison.',
        'root_cause':'The original lazy NPZ noise object decompressed its full arrays on every row. The optimized replay loads each array once, marks it read-only, verifies unchanged bytes, and reuses exact left-to-right mask sums.',
        'existing_audits_preserved':True,'scientific_fits_or_searches_rerun':False}
    save(ROOT/'artifacts/research/20261114/ripper/replay_optimization_validation.json',r);print(json.dumps(r))


if __name__=='__main__':main()
