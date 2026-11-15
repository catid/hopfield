"""Publish independent verifier optimization with original budget failure retained."""
import json
import shutil
import study as S


def main():
    source=S.OUT/'local_factor_audit';gate=S.OUT/'local_factor_preflight';public=S.ROOT/'results/research/work';out=public/'local-factor-audit';out.mkdir(exist_ok=True);a=json.loads((source/'audit.json').read_text());assert a['status']=='passed' and a['verification_hypothesis']=='H2az';benchmark=json.loads((source/'candidate-benchmark.json').read_text());full=json.loads((source/'full-job-comparison.json').read_text());assert full['both_completed'] and full['all_scientific_audit_fields_identical']
    for name in ['protocol.json','numerical-gate.json','candidate-benchmark.json','full-job-full_grid.json','full-job-local_factors.json','full-job-comparison.json','audit-launch-source-check.json']:shutil.copyfile(source/name,out/name)
    for name in ['public-fixture-verification.json','public-fixture-import-isolation.json']:shutil.copyfile(gate/name,out/name)
    fixtures=out/'fixtures';fixtures.mkdir(exist_ok=True)
    for p in (gate/'fixtures').iterdir():
        if p.is_file():shutil.copyfile(p,fixtures/p.name)
    r={k:v for k,v in a.items() if k not in ['rows','run_rows','worker_receipts']};r['complete_audit_public_path']='../density-transfer-summary.json';r['complete_audit_source_sha256']=S.sha(source/'audit.json');S.save(out/'verification-summary.json',r)
    summary=json.loads((public/'summary.json').read_text());assert 'local_factor_audit' not in summary;summary['local_factor_audit']={'hypothesis':'H2az','fresh_inputs':0,'new_inference_trajectories':0,'candidate_benchmark_count':512,'median_candidate_process_speedup':benchmark['median_paired_process_speedup'],'primary':benchmark['primary_at_least_2x'],'full_source_job_wall_speedup':full['wall_speedup'],'full_source_job_both_complete':True,'all_scientific_source_audit_fields_identical':True,'full_density_audit_seconds':a['seconds'],'full_density_final_states':a['all_final_states'],'full_density_candidate_checks':a['candidate_literal_checks'],'original_density_audit':'Failed at300s, preserved; successor separately frozen600s verification budget. No inference rerun; no original gate retroactive pass.'};summary['hypotheses'].append({'campaign':'local_factor_audit','hypothesis':'H2az fixed512candidate medianprocessspeedup>=2x with algebraic equivalence','outcome':benchmark['primary_at_least_2x']});p=json.loads((source/'protocol.json').read_text());summary['protocols']['local-factor-audit-protocol']={'sha256':S.sha(source/'protocol.json'),'frozen_utc':p['frozen_utc'],'hypothesis':'H2az','status':'completed'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
