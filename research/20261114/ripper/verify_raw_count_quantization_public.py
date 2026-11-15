"""Portable manifest, scientific proof replay and H1by resource accounting."""
import json
from raw_count_quantization import ROOT, PUBLIC, PROTOCOL, sha, read
from audit_raw_count_quantization import main as audit


def main():
    manifest = read(ROOT/'results/research/ripper/raw_count_quantization_public_manifest.json')
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    assert sha(PUBLIC/'summary.json') == manifest['raw_audit_sha256']
    audit()
    summary = read(PUBLIC/'summary.json')
    runtime = read(PUBLIC/'runtime.json')
    assert summary['runtime'] == runtime and runtime['status'] == 'validated'
    assert runtime['protocol_sha256'] == sha(PROTOCOL) and runtime['all_banks_retained']
    assert runtime['wall_s'] < 180 and runtime['aggregate_cpu_s'] < 900 and runtime['max_cpu_workers'] == 8 and runtime['gpu_seconds'] == 0
    commands = runtime['commands']
    assert len(commands) == 9 and all(c['returncode'] == 0 for c in commands)
    assert all(c['action'] == 'worker' for c in commands[:8]) and sorted(c['seed'] for c in commands[:8]) == list(range(8))
    assert commands[-1]['action'] == 'summary'
    progress = read(PUBLIC/'execution/progress.json')
    assert progress['status'] == 'complete' and progress['active_pids'] == []
    assert progress['completed_commands'] == 9 and progress['completed_LPs'] == 5120 and progress['audited_banks'] == 64
    assert progress['wall_s'] == runtime['wall_s'] and progress['cpu_s'] == runtime['aggregate_cpu_s']
    q = summary['query_accounting']
    assert q['new_source_draws'] == q['new_noisy_queries'] == q['new_clean_oracle_queries'] == 0
    assert q['parent_physical_noisy_union'] == 2073600 and q['official_noisy_label_sequence'] == 14687319
    assert summary['all128_arms_retained'] and summary['archived_selected_certificates'] == summary['analytic_zero_proofs'] == 128
    timing = read(PUBLIC/'solver-timing-accounting.json')
    records = [r for seed in range(64) for arm in ['single270','triple90'] for r in read(PUBLIC/f's{seed}/{arm}-receipt.json')['records']]
    cpu = sum(r['solver']['solver_cpu_s'] for r in records)
    assert timing['solver_calls'] == len(records) == 5120 and timing['solver_call_cpu_seconds'] == cpu
    assert timing['charged_aggregate_cpu_seconds'] == runtime['aggregate_cpu_s']
    assert timing['solver_cpu_fraction'] == cpu/runtime['aggregate_cpu_s'] and timing['remaining_cpu_seconds'] == runtime['aggregate_cpu_s']-cpu
    print(json.dumps(dict(status='passed', manifest_paths=len(manifest['sha256'])+1, new_queries=0, raw_artifacts_required=False)))


if __name__ == '__main__':
    main()
