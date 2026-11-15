"""Portable H1bw replay with exactly two published registry aliases."""
import json
import audit_repeat_query_tradeoff as audit
from repeat_query_tradeoff_study import ROOT, PUBLIC, REGISTRY, PROTOCOL, sha


def portable_verify(raw=False):
    assert not raw, 'Original-label audits require original raw artifacts'
    p = json.loads(PROTOCOL.read_text())
    aliases = {str((REGISTRY/name).relative_to(ROOT)): PUBLIC/name for name in ['excluded_ids.npz', 'ledger.json']}
    for relative, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(aliases.get(relative, ROOT/relative)) == digest, relative
    return p


def portable_job(seed):
    audit.verify = portable_verify
    return audit.audit_bank(seed, True)


def main():
    manifest = ROOT/'results/research/ripper/repeat_query_tradeoff_public_manifest.json'
    m = json.loads(manifest.read_text())
    for relative, digest in m['sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    assert sha(PUBLIC/'summary.json') == m['raw_audit_sha256']
    audit.verify = portable_verify
    audit.public_job = portable_job
    audit.public_audit()
    s = json.loads((PUBLIC/'summary.json').read_text())
    runtime = json.loads((PUBLIC/'runtime.json').read_text())
    assert runtime == s['runtime'] and runtime['status'] == 'validated' and runtime['protocol_sha256'] == sha(PROTOCOL)
    assert runtime['wall_s'] < 180 and runtime['aggregate_cpu_s'] < 900 and runtime['max_cpu_workers'] == 8 and runtime['gpu_seconds'] == 0
    commands = runtime['commands']
    assert len(commands) == 13 and all(c['returncode'] == 0 for c in commands)
    assert [c['action'] for c in commands[:4]] == ['select', 'predraw-all', 'materialize', 'source-audit']
    assert all(c['action'] == 'worker' for c in commands[4:12]) and sorted(c['seed'] for c in commands[4:12]) == list(range(8))
    assert commands[-1]['action'] == 'summary'
    partitions = [list(range(worker, 64, 8)) for worker in range(8)]
    assert sorted(seed for part in partitions for seed in part) == list(range(64))
    q = s['query_accounting']
    assert q['full_physical_noisy_union'] == 2073600 and q['official_noisy_label_sequence'] == 12613719+2073600
    assert q['single270_deployment'] == q['triple90_deployment'] == 1244160
    assert q['shared_first_query90'] == 414720 and q['extra_repeat_queries'] == 829440
    assert q['clean_unique_oracle_cells'] == 81*s['source_audit']['new_unique_source_ids'] <= 1399680
    assert q['clean_materialized_cell_instances'] == 1399680 and s['analytic_zero_proofs'] == s['complete_label_gates'] == 128
    progress = json.loads((PUBLIC/'execution/progress.json').read_text())
    assert progress['status'] == 'complete' and progress['active_pids'] == [] and progress['completed_commands'] == 13
    assert progress['completed_LPs'] == 5120 and progress['audited_banks'] == 64
    assert progress['wall_s'] == runtime['wall_s'] and progress['cpu_s'] == runtime['aggregate_cpu_s']
    from repeat_query_math import exact_channels
    from repeat_query_tradeoff_study import HERE
    assert json.loads((HERE/'repeat-query-channel.json').read_text()) == dict(status='passed', **exact_channels())
    noise = json.loads((PUBLIC/'noise-accounting.json').read_text())
    for name in ['first', 'repeat1', 'repeat2', 'decoded']:
        for key in ['full_cells', 'queried_cells', 'full_cell_errors', 'queried_cell_errors']:
            assert noise[name][key] == sum(row['realized_noise'][name][key] for row in s['raw_audits'])
        assert noise[name]['queried_error_rate'] == noise[name]['queried_cell_errors']/noise[name]['queried_cells']
        assert noise[name]['queried_error_offsets'] == [sum(row['realized_noise'][name]['queried_error_offsets'][i] for row in s['raw_audits']) for i in range(8)]
    assert noise['ties'] == {key:sum(row['realized_noise']['ties'][key] for row in s['raw_audits']) for key in ['all_cells', 'queried_cells']}
    timing = json.loads((PUBLIC/'solver-timing-accounting.json').read_text())
    records = [r for seed in range(64) for arm in ['single270', 'triple90'] for r in json.loads((PUBLIC/f's{seed}/{arm}-receipt.json').read_text())['records']]
    cpu = sum(r['solver']['solver_cpu_s'] for r in records)
    assert timing['solver_calls'] == len(records) == 5120 and timing['solver_call_cpu_seconds'] == cpu
    assert timing['charged_aggregate_cpu_seconds'] == runtime['aggregate_cpu_s']
    assert timing['solver_cpu_fraction'] == cpu/runtime['aggregate_cpu_s'] and timing['remaining_cpu_seconds'] == runtime['aggregate_cpu_s']-cpu
    print(json.dumps(dict(status='passed', manifest_paths=len(m['sha256'])+1, portable_registry_mappings=2,
                          all_IDs_and64_full_masks_sealed_before_label_commands=True, resource_and_query_caps_passed=True)))


if __name__ == '__main__':
    main()
