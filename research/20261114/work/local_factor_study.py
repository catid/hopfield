"""H2az freeze, paired candidate benchmark, and bounded old/new source-job checks."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import statistics
import time
import numpy as np
import study as S
from verify_training_feasibility import energy,LD
from local_factor_delta import deltas
PP=S.HERE/'local-factor-audit-protocol.json'
DEST=S.OUT/'local_factor_audit'
SOURCE=S.OUT/'density_transfer'
GATE=S.OUT/'local_factor_preflight'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    gate=json.loads((GATE/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['small_ternary_states']==3**9 and gate['actual_candidates']=={'H2aw':168,'H2ay':344};assert gate['local_factor_sha256']==S.sha(S.HERE/'local_factor_delta.py') and gate['script_sha256']==S.sha(S.HERE/'preflight_local_factor_delta.py');failure=json.loads((SOURCE/'audit-budget-exhausted.json').read_text());assert failure['external_exit_code']==124 and not (SOURCE/'audit.json').exists();manifest=json.loads((GATE/'fixtures/manifest.json').read_text());assert manifest['candidate_count']==512
    names=['local_factor_study.py','local_factor_delta.py','preflight_local_factor_delta.py','audit_density_local_factors.py','audit_density_transfer.py','verify_training_feasibility.py','verify_large_cycle_geometry.py','verify_graph_block_source.py','verify_cycle_cut_geometry_checked.py','verify_cycle_cut_geometry.py','audit_connectivity_screen.py','LOCAL_FACTOR_AUDIT.md'];p={'id':'retained-density180-independent-local-factor-audit','hypothesis_id':'H2az','status':'frozen_after_complete_equivalence_gate_before_timed_benchmarks','frozen_utc':S.utc(),'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'original_failed_audit_receipt_sha256':S.sha(SOURCE/'audit-budget-exhausted.json'),'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'gate_sha256':S.sha(GATE/'numerical-gate.json'),'fixture_manifest_sha256':S.sha(GATE/'fixtures/manifest.json'),'benchmark':{'candidate_count':512,'repeats':3,'order':[['full_grid','local_factors'],['local_factors','full_grid'],['full_grid','local_factors']],'primary':'Median paired process-time ratio full_grid/local_factors >=2; support if at least2, otherwise not met. This is a fixed source implementation benchmark, not statistical generalization.','warmup':'One untimed equivalence pass for both methods over the fixed512 candidates; all loaded inputs promoted losslessly before timing.','process_affinity':'Pin micro and full-job benchmarks to the lowest allowed logical CPU; record allowed/pinned CPUs and environment.','budget_seconds':120},'full_job_benchmark':{'job':'shard00-s0','order':['full_grid','local_factors'],'per_method_wall_seconds':120,'decision':'Only if both finish, compare all scientific audit fields excluding elapsed time and floating-point error maxima; then report measured process/wall ratios. If either times out, report timeout and any conservative bound, never a measured completed ratio.'},'full_verification':{'source_jobs':48,'all_final_states':73728,'all_candidates':'Every H2ay trace candidate, no subsampling; unchanged graph/native margin/final energy/flags/readout checks','wall_seconds':600,'cpu_processes':16,'persistence':'Atomic per-job rows; collect as completed; retain all partial rows on failure. Write successor audit only under local_factor_audit, never original density_transfer/audit.json.'},'scientific_accounting':{'new_independent_inputs':0,'new_inference_trajectories':0,'new_optimizer_updates':0,'source_input_count':4096,'source_model_count':3,'source_arms':6,'original_H2ay_audit_budget_status':'Failed300s limit remains failed; later complete verification uses separately declared600s budget. No original budget extension or inference rerun.'},'math':'Direct original-factor cancellation for C: unary changes plus unique valid directed source sites C union(C-offset), evaluating original before/after pair-table and directed-link products in longdouble. Self-links once; no production flip/quadratic scorer imported. Full-grid final energies and native margins remain unchanged.'};S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);S.save(DEST/'numerical-gate.json',gate);S.emit('local_factor_study_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text())
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    assert S.sha(SOURCE/'protocol.json')==p['source_protocol_sha256'] and S.sha(GATE/'numerical-gate.json')==p['gate_sha256'] and S.sha(GATE/'fixtures/manifest.json')==p['fixture_manifest_sha256'];return p


def pin():
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]});return {'allowed_cpus':allowed,'pinned_cpu':allowed[0],'numpy_version':np.__version__,'longdouble_mantissa_bits':int(np.finfo(LD).nmant)}


def fixtures():
    groups=[];m=json.loads((GATE/'fixtures/manifest.json').read_text())
    for record in m['groups']:
        path=GATE/'fixtures'/record['path'];assert S.sha(path)==record['sha256']
        with np.load(path) as z:d={k:z[k] for k in z.files}
        d['theta']=d['theta'].astype(LD);d['readout']=d['readout'].astype(LD);d['items']=[(int(owner),tuple(map(int,d['cells'][lo:hi]))) for owner,lo,hi in zip(d['owners'],d['cell_offsets'][:-1],d['cell_offsets'][1:])];groups.append(d)
    assert sum(len(d['items']) for d in groups)==512;return groups


def evaluate(groups,method):
    output=[]
    for d in groups:
        t,b,a,theta,mu,ao,items=[d[k] for k in ['tokens','boards','readout','theta','mu_offsets','aoffs','items']]
        if method=='local_factors':output.extend(deltas(t,b,a,theta,mu,ao,items));continue
        for lo in range(0,len(items),32):
            part=items[lo:lo+32];owners=np.array([i for i,_ in part]);old=b[owners];new=old.copy()
            for j,(_,cells) in enumerate(part):new[j].flat[list(cells)]=False
            output.extend(energy(t[owners],new,a[owners],theta,mu,ao)-energy(t[owners],old,a[owners],theta,mu,ao))
    return np.array(output,dtype=LD)


def benchmark():
    p=protocol();inventory=pin();started=time.perf_counter();groups=fixtures();reference=evaluate(groups,'full_grid');local=evaluate(groups,'local_factors');assert abs(reference-local).max()<1e-9;rows=[]
    for rep,order in enumerate(p['benchmark']['order']):
        result={'repeat':rep,'order':order}
        for method in order:
            wall=time.perf_counter();cpu=time.process_time();out=evaluate(groups,method);cpu=time.process_time()-cpu;wall=time.perf_counter()-wall;assert abs(out-reference).max()<1e-9;result[method]={'process_seconds':cpu,'wall_seconds':wall}
        result['process_speedup']=result['full_grid']['process_seconds']/result['local_factors']['process_seconds'];rows.append(result);assert time.perf_counter()-started<120
    ratio=statistics.median(r['process_speedup'] for r in rows);r={'status':'passed','protocol_sha256':S.sha(PP),'inventory':inventory,'fixed_deletions':512,'rows':rows,'median_paired_process_speedup':ratio,'primary_at_least_2x':'met' if ratio>=2 else 'not_met','maximum_equivalence_error':float(abs(reference-local).max()),'seconds':time.perf_counter()-started,'scope':'Microbenchmark only; full-job result required for end-to-end claim.'};S.save(DEST/'candidate-benchmark.json',r);print(json.dumps(r,indent=2))


def full_job(method):
    p=protocol();inventory=pin();module=importlib.import_module('audit_density_transfer' if method=='full_grid' else 'audit_density_local_factors');source=json.loads((SOURCE/'protocol.json').read_text());started=time.perf_counter();cpu=time.process_time();status='complete';row=None
    def alarm(signum,frame):raise TimeoutError('Frozen120s full-job benchmark ceiling')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:row=module.job_check((S.ROOT,SOURCE,source['jobs'][0],source))
    except TimeoutError:status='timeout'
    finally:signal.alarm(0)
    r={'status':status,'method':method,'job':source['jobs'][0],'protocol_sha256':S.sha(PP),'inventory':inventory,'wall_seconds':time.perf_counter()-started,'process_seconds':time.process_time()-cpu,'budget_seconds':120,'row':row};S.save(DEST/f'full-job-{method}.json',r);print(json.dumps({k:v for k,v in r.items() if k!='row'},indent=2))


def compare_jobs():
    protocol();rows={m:json.loads((DEST/f'full-job-{m}.json').read_text()) for m in ['full_grid','local_factors']};complete=all(r['status']=='complete' for r in rows.values());r={'status':'complete_comparison' if complete else 'incomplete_comparison','protocol_sha256':S.sha(PP),'both_completed':complete,'source_job':'shard00-s0','measured_end_to_end_speedup_available':complete}
    if complete:
        a,b=[rows[m]['row'] for m in ['full_grid','local_factors']];excluded={'seconds','max_literal_error','max_final_margin_error','max_final_energy_error'};assert {k:v for k,v in a.items() if k not in excluded}=={k:v for k,v in b.items() if k not in excluded};r.update(all_scientific_audit_fields_identical=True,wall_speedup=rows['full_grid']['wall_seconds']/rows['local_factors']['wall_seconds'],process_speedup=rows['full_grid']['process_seconds']/rows['local_factors']['process_seconds'])
    S.save(DEST/'full-job-comparison.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','benchmark','full-job','compare-jobs']);a.add_argument('--method',choices=['full_grid','local_factors']);v=a.parse_args()
    if v.command=='full-job':full_job(v.method)
    elif v.command=='compare-jobs':compare_jobs()
    else:globals()[v.command]()
