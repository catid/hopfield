"""Portable H1ce public manifest, interval/proof replay and resource accounting."""
import json
from estimated_channel_study import ROOT,PUBLIC,PROTOCOL,ARMS,sha,read
from audit_estimated_channel import main as audit


def main():
    manifest=read(ROOT/'results/research/ripper/estimated_channel_public_manifest.json')
    for rel,digest in manifest['sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(PUBLIC/'summary.json')==manifest['raw_audit_sha256']
    audit();s=read(PUBLIC/'summary.json');runtime=read(PUBLIC/'runtime.json')
    assert s['runtime']==runtime and runtime['status']=='validated' and runtime['protocol_sha256']==sha(PROTOCOL)
    assert runtime['wall_s']<120 and runtime['aggregate_cpu_s']<450 and runtime['max_cpu_workers']==8 and runtime['gpu_seconds']==0
    commands=runtime['commands'];assert len(commands)==9 and all(c['returncode']==0 for c in commands)
    assert all(c['action']=='worker' for c in commands[:8]) and sorted(c['seed'] for c in commands[:8])==list(range(8)) and commands[-1]['action']=='summary'
    progress=read(PUBLIC/'execution/progress.json')
    assert progress['status']=='complete' and progress['active_pids']==[] and progress['completed_commands']==9
    assert progress['completed_LPs']==s['attempted_LPs']<=2560 and progress['audited_banks']==64
    assert progress['wall_s']==runtime['wall_s'] and progress['cpu_s']==runtime['aggregate_cpu_s']
    assert s['new_source_IDs']==s['new_noisy_queries']==0 and s['parent_physical_noisy_union']==2073600 and s['official_noisy_label_sequence']==16760919
    timing=read(PUBLIC/'solver-timing-accounting.json')
    records=[r for seed in range(64) for arm in ARMS for r in read(PUBLIC/f's{seed}/{arm}-receipt.json')['records']]
    cpu=sum(r['solver']['solver_cpu_s'] for r in records)
    assert timing['solver_calls']==len(records)==s['attempted_LPs'] and timing['solver_call_cpu_seconds']==cpu
    assert timing['charged_aggregate_cpu_seconds']==runtime['aggregate_cpu_s'] and timing['solver_cpu_fraction']==cpu/runtime['aggregate_cpu_s']
    assert timing['remaining_cpu_seconds']==runtime['aggregate_cpu_s']-cpu
    from fractions import Fraction
    numeric=read(PUBLIC/'numeric-accounting.json')
    eligible=[r for r in s['rows'] if r['estimator']['fit_allowed']]
    expected_costs=3240*len(eligible)
    assert numeric['integer_costs']==expected_costs and numeric['producer80_digit_costs']+numeric['producer160_digit_costs']==expected_costs
    assert numeric['producer160_digit_costs']==sum(r['numeric']['producer160_digit_edges'] for r in eligible)
    assert numeric['numerical_unresolved_banks']==sum(not r['numeric']['fit_allowed'] for r in eligible)
    assert numeric['estimator_unresolved_banks']==64-len(eligible)
    widths=[Fraction(r['max_cost_interval_width']) for r in s['rows'] if r['max_cost_interval_width'] is not None]
    width=max(widths) if widths else None
    assert numeric['maximum_cost_interval_width_exact']==(str(width) if width is not None else None)
    assert numeric['maximum_cost_interval_width_approx']==(float(width) if width is not None else None)
    assert numeric['independent_digits']==200 and numeric['independent_terms']==240
    assert s['archived_selected_certificates']==64 and s['all64_banks_retained']
    print(json.dumps(dict(status='passed',manifest_paths=len(manifest['sha256'])+1,raw_artifacts_required=False,new_queries=0)))


if __name__=='__main__':main()
