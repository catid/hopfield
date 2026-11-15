"""Portable H1ci manifest, acquired observations, exact costs/proofs and budget replay."""
import json
from fractions import Fraction
from diversity_after_correlation_study import ROOT,PUBLIC,PROTOCOL,ARMS,sha,read
from audit_diversity_after_correlation import main as audit


def main():
    manifest=read(ROOT/'results/research/ripper/diversity_after_correlation_public_manifest.json')
    for rel,digest in manifest['sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(PUBLIC/'summary.json')==manifest['raw_audit_sha256']
    audit();s=read(PUBLIC/'summary.json');runtime=read(PUBLIC/'runtime.json');p=read(PROTOCOL)
    assert s['runtime']==runtime and runtime['status']=='validated' and runtime['protocol_sha256']==sha(PROTOCOL)
    assert runtime['wall_s']<p['budget']['wall_seconds']==180 and runtime['aggregate_cpu_s']<p['budget']['aggregate_cpu_seconds']==900
    assert runtime['max_cpu_workers']==8 and runtime['gpu_seconds']==0
    commands=runtime['commands'];assert len(commands)==14 and all(c['returncode']==0 for c in commands)
    assert [c['action'] for c in commands[:5]]==['select','predraw-all','materialize','source-audit','prepare'] and all(c['seed'] is None for c in commands[:5])
    assert all(c['action']=='worker' for c in commands[5:13]) and sorted(c['seed'] for c in commands[5:13])==list(range(8)) and commands[-1]['action']=='summary'
    progress=read(PUBLIC/'execution/progress.json')
    assert progress['status']=='complete' and progress['active_pids']==[] and progress['completed_commands']==14
    assert progress['completed_LPs']==s['attempted_LPs']<=5760 and progress['audited_banks']==72
    assert progress['wall_s']==runtime['wall_s'] and progress['cpu_s']==runtime['aggregate_cpu_s']
    deviation=read(PUBLIC/'monitor-deviation.json')
    assert deviation['status']=='completed_science_with_monitoring_deviation'
    assert deviation['frozen_sources_and_raw_results_unchanged'] and deviation['no_rerun']
    assert deviation['protocol_sha256']==sha(PROTOCOL)
    assert deviation['runner_sha256']==runtime['wrapper_sha256']
    assert deviation['runtime_sha256']==sha(PUBLIC/'runtime.json') and deviation['summary_sha256']==sha(PUBLIC/'summary.json')
    assert deviation['runner_log_sha256']==sha(PUBLIC/'execution/runner-full.log')
    assert deviation['samples_sha256']==sha(PUBLIC/'execution/samples.jsonl')
    samples=[json.loads(line) for line in (PUBLIC/'execution/samples.jsonl').read_text().splitlines()]
    assert deviation['successful_monitor_samples']==len(samples)==385
    assert deviation['last_successful_sample']==samples[-1]
    assert deviation['terminal_wall_s']==runtime['wall_s'] and deviation['terminal_aggregate_cpu_s']==runtime['aggregate_cpu_s']
    assert deviation['last_sample_to_terminal_wall_s']==runtime['wall_s']-samples[-1]['wall_s']
    assert deviation['observed_terminal_wall_within_frozen_budget'] and deviation['observed_terminal_cpu_within_frozen_budget']
    assert deviation['all14_children_reaped_before_terminal_measurement'] and deviation['continuous_monitoring_completed'] is False
    log=(PUBLIC/'execution/runner-full.log').read_text()
    assert deviation['exception']=='ProcessLookupError: [Errno 3] No such process' and deviation['exception'] in log
    assert json.loads(log.splitlines()[-1])['runtime']==runtime
    raw=read(PUBLIC/'raw-replay-receipt.json')
    assert raw['status']=='passed' and raw['read_only'] and raw['new_fits']==0
    assert raw['protocol_sha256']==sha(PROTOCOL) and raw['raw_csv_and_all72_noise_archives_replayed'] and raw['all144_costs_grids_and_proofs_replayed']
    query=s['query_accounting'];assert query['full_physical_noisy_union']==2332800 and query['supplied_correlated90_deployment']==query['supplied_single270_deployment']==1399680 and query['shared_between_methods']==466560
    assert query['official_noisy_label_sequence_before']==20959959 and query['official_noisy_label_sequence_after']==23292759
    assert query['source_instances']==19440 and query['clean_unique_oracle_cells']<=query['clean_materialized_cell_instances']==1574640
    timing=read(PUBLIC/'solver-timing-accounting.json')
    records=[r for seed in range(72) for arm in ARMS for r in read(PUBLIC/f's{seed}/{arm}-receipt.json')['records']]
    cpu=sum(r['solver']['solver_cpu_s'] for r in records)
    assert timing['solver_calls']==len(records)==s['attempted_LPs'] and timing['solver_call_cpu_seconds']==cpu
    assert timing['charged_aggregate_cpu_seconds']==runtime['aggregate_cpu_s'] and timing['solver_cpu_fraction']==cpu/runtime['aggregate_cpu_s']
    assert timing['remaining_cpu_seconds']==runtime['aggregate_cpu_s']-cpu
    numeric=read(PUBLIC/'numeric-accounting.json');eligible=[r for r in s['rows'] if r['numeric'] is not None];expected_costs=3240*len(eligible)
    assert numeric['integer_costs']==expected_costs and numeric['producer80_digit_costs']+numeric['producer160_digit_costs']==expected_costs
    assert numeric['producer160_digit_costs']==sum(r['numeric']['producer160_digit_edges'] for r in eligible)
    assert numeric['numerical_unresolved_arms']==sum(not r['numeric']['fit_allowed'] for r in eligible)
    assert len(eligible)==144
    widths=[Fraction(r['max_cost_interval_width']) for r in s['rows'] if r['max_cost_interval_width'] is not None];width=max(widths) if widths else None
    assert numeric['maximum_cost_interval_width_exact']==(str(width) if width is not None else None)
    assert numeric['maximum_cost_interval_width_approx']==(float(width) if width is not None else None)
    assert numeric['independent_digits']==200 and numeric['independent_terms']==240
    assert s['all144_arms_retained'] and len(s['rows'])==144 and s['source_audit']['source_instances']==19440
    print(json.dumps(dict(status='passed',manifest_paths=len(manifest['sha256'])+1,raw_artifacts_required=False,new_queries=2332800)))


if __name__=='__main__':main()
