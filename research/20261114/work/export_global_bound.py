"""Publish exact coefficients and allocation proof with a public-only verifier."""
import json
import shutil
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_window_fallback.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_window_fallback
    export_window_fallback.main();source=S.OUT/'global_bound';out=S.ROOT/'results/research/work';target=out/'global-bound';target.mkdir(exist_ok=True)
    for seed in range(3):
        dst=target/f'seed{seed}';dst.mkdir(exist_ok=True)
        for name in ['constants.npz','coefficients.json','half-result.json','allocation-certificate.json','allocation-result.json']:shutil.copyfile(source/f'seed{seed}'/name,dst/name)
    verify=json.loads((source/'independent-verification.json').read_text());assert verify['status']=='passed'
    shutil.copyfile(source/'independent-verification.json',out/'global-bound-verification.json');half=json.loads((source/'half-summary.json').read_text());allocation=json.loads((source/'allocation-summary.json').read_text())
    result={'status':'exact_certificates_verified','scope':'One preselected90grid graph, all3frozen models and archived readouts. Unique globalminimum over free binaryanswer cells with walls/endpoints clamped; no continuousminimum or populationclaim.','primary_half_allocation':'No certificate for all3models; retained negative slacks.','predeclared_secondary_asymmetric_allocation':'Strict exactrational certificate for all3models.','records':verify['records'],'half_allocation_source_receipts':half,'allocation_source_receipts':allocation,'protocol_sha256':S.sha(S.HERE/'global-bound-protocol.json'),'verifier_sha256':S.sha(S.HERE/'verify_global_bound.py'),'derivation_sha256':S.sha(S.HERE/'global-bound-derivation.md'),'public_files':{str(p.relative_to(out)):S.sha(p) for p in sorted(target.rglob('*')) if p.is_file()},'new_inference_or_training':False,'new_independent_inputs':0,'source_to_coefficient':'Publicrawconstants plus independentexactper-node reconstruction in verifier; upstream source hashes, checkpoint hashes and archivedreadout hashes retained in receipts.'}
    S.save(out/'global-bound-summary.json',result)
    summary=json.loads((out/'summary.json').read_text());summary['global_binary_certificate']={'selected_graph_only':True,'models':3,'unique_global_binary_endpoint_minima':3,'half_allocation_successes':0,'secondary_asymmetric_successes':3,'new_inference_or_training':False,'public_summary':'global-bound-summary.json'};summary['updated_utc']=S.utc();S.save(out/'summary.json',summary);manifest(out);S.emit('global_bound_public_export',models=3,files=len(result['public_files']))


if __name__=='__main__':main()
