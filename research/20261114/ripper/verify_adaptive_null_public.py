"""Public graph statistics and separate failed/additional-budget provenance."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    s=json.loads((PUBLIC/'adaptive_null.json').read_text());manifest=json.loads((PUBLIC/'adaptive_null_public_manifest.json').read_text());oldp=json.loads((HERE/'protocol-adaptive-null.json').read_text());newp=json.loads((HERE/'protocol-adaptive-null-continuation.json').read_text())
    assert s['status']=='validated'and s['protocol_sha256']==manifest['scientific_protocol_sha256']==sha(HERE/'protocol-adaptive-null.json')
    assert s['continuation_protocol_sha256']==manifest['continuation_protocol_sha256']==sha(HERE/'protocol-adaptive-null-continuation.json')
    assert s['original_gate_status']==newp['original_gate_status']=='failed_budget_limit'
    for p,d in manifest['files_sha256'].items():assert sha(ROOT/p)==d
    for name,d in (oldp['code_sha256']|newp['code_sha256']).items():assert sha(HERE/name)==d
    original=json.loads((HERE/'adaptive-null-original-artifacts.json').read_text());assert len(original['artifact_sha256'])==42 and original['completed_acquisition_banks']==list(range(8))
    checks=json.loads((PUBLIC/'adaptive_null_preflight_checks.json').read_text());pre=json.loads((PUBLIC/'adaptive_null_preflight_runtime.json').read_text())
    assert checks['status']=='passed'and len(checks['arrays_checked'])==80 and checks['preserved_original_files']==42
    assert sha(PUBLIC/'adaptive_null_preflight_checks.json')==pre['checks_sha256']==newp['input_sha256']['artifacts/research/20261114/ripper/adaptive_null_preflight/checks.json']
    assert sha(PUBLIC/'adaptive_null_preflight_runtime.json')==newp['input_sha256']['artifacts/research/20261114/ripper/adaptive_null_preflight/runtime.json']
    before=(HERE/'audit_adaptive_null.py').read_text();after=(HERE/'audit_adaptive_null_eager.py').read_text();helper='\ndef eager_npz(path):\n    with np.load(path,allow_pickle=False) as archive:\n        return {name:archive[name] for name in archive.files}\n\n';expected=before.replace('\ndef main():',helper+'\ndef main():',1)
    for a,b in [("obs=np.load(d/'observations.npz')","obs=eager_npz(d/'observations.npz')"),("noise=np.load(d/'potential_noise.npz')","noise=eager_npz(d/'potential_noise.npz')"),("oldnoise=np.load(olddir/'potential_noise.npz')","oldnoise=eager_npz(olddir/'potential_noise.npz')"),("oldobs=np.load(olddir/'observations.npz')","oldobs=eager_npz(olddir/'observations.npz')")]:assert expected.count(a)==1;expected=expected.replace(a,b)
    assert expected==after and ast.dump(ast.parse(expected))==ast.dump(ast.parse(after))
    assert checks['original_auditor_sha256']==sha(HERE/'audit_adaptive_null.py')and checks['corrected_auditor_sha256']==sha(HERE/'audit_adaptive_null_eager.py')
    with np.load(PUBLIC/'adaptive_null_supports.npz',allow_pickle=False)as z:
        supports=z['supports'];assert np.array_equal(z['banks'],np.arange(32))
    assert supports.shape==(32,3240)and supports.dtype==bool
    i,j=np.triu_indices(81,1);truth=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));parent=json.loads((PUBLIC/'null_replay.json').read_text())
    for bank,(r,w)in enumerate(zip(s['rows'],supports,strict=True)):
        assert r['seed']==bank;a=next(a for a in s['acquisition_audits']if a['seed']==bank)
        assert a['auditor_sha256']==sha(HERE/'audit_adaptive_null_eager.py')and a['support_bytes_sha256']==hashlib.sha256(w.tobytes()).hexdigest()
        missing=int((truth&~w).sum());extra=int((~truth&w).sum());assert r['missing_true_edges']==missing and r['extra_edges']==extra and r['all_true_retained']==(missing==0)and r['exact_graph']==(missing==extra==0)
        baseline=next(x for x in parent['rows']if x['seed']==bank and x['arm']=='exact_law');assert baseline['extra_edges']==r['fixed_mask_extras']and baseline['missing_true_edges']==r['fixed_mask_missing']
    g=s['grouped'][1]
    for field in g:
        if field!='arm':assert g[field]==sum(r[field]for r in s['rows'])
    assert s['grouped'][0]['extra_edges']==sum(r['fixed_mask_extras']for r in s['rows'])==14229
    primary=s['primary'];retention=g['all_true_retained']>=29;efficiency=2*g['extra_edges']<=14229
    assert primary['retention_target_met']==retention and primary['extra_target_met']==efficiency
    assert primary['all_true_retaining_banks']==g['all_true_retained']and primary['pooled_extra_edges']==g['extra_edges']and primary['extra_ceiling']==7114
    assert primary['decision']==('supported_finite_bank_conjunction'if retention and efficiency else'falsified_finite_bank_conjunction')
    assert abs(primary['extra_reduction_pct']-100*(1-g['extra_edges']/14229))<1e-12
    runtime=s['runtime'];assert runtime==json.loads((PUBLIC/'adaptive_null_runtime.json').read_text())and runtime['original_gate_status']=='failed_budget_limit'
    assert runtime['original_trigger_cpu_s']==original['trigger']['aggregate_cpu_s']==120.004219 and runtime['original_complete_final_cpu']=='unavailable'
    assert runtime['preflight_cpu_s']==pre['full_parent_child_cpu_s']and runtime['preflight_wall_s']==pre['wall_s']
    assert abs(runtime['additional_aggregate_cpu_s']-runtime['continuation_cpu_s']-runtime['preflight_cpu_s'])<1e-9
    assert abs(runtime['additional_execution_wall_s']-runtime['continuation_wall_s']-runtime['preflight_wall_s'])<1e-9
    assert runtime['additional_aggregate_cpu_s']<120 and runtime['additional_execution_wall_s']<60
    assert runtime['reused_acquisition_banks']==list(range(8))and runtime['new_acquisition_banks']==list(range(8,32))and runtime['reused_preflight_audits']==[0]
    assert sum(c['script']=='adaptive_null_acquire.py'for c in runtime['commands'])==24
    assert sum(c['script']=='audit_adaptive_null_eager.py'for c in runtime['commands'])==31
    print(json.dumps(dict(status='passed',banks=32,primary=primary,original_gate='failed',additional_cpu_s=runtime['additional_aggregate_cpu_s'],exact_eager_only_diff_verified=True,full_array_equivalence_and_acquisition_replay_require_raw_artifacts=True)))
if __name__=='__main__':main()
