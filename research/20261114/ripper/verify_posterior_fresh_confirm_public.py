"""Portable H1ca replay with exactly two published registry aliases."""
import json
import audit_posterior_fresh_confirm as audit
from posterior_fresh_confirm_study import ROOT, PUBLIC, REGISTRY, PROTOCOL, sha


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
    manifest = ROOT/'results/research/ripper/posterior_fresh_confirm_public_manifest.json'
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
    assert len(commands) == 14 and all(c['returncode'] == 0 for c in commands)
    assert [c['action'] for c in commands[:5]] == ['select', 'predraw-all', 'materialize', 'source-audit', 'prepare']
    assert all(c['action'] == 'worker' for c in commands[5:13]) and sorted(c['seed'] for c in commands[5:13]) == list(range(8))
    assert commands[-1]['action'] == 'summary'
    partitions = [list(range(worker, 64, 8)) for worker in range(8)]
    assert sorted(seed for part in partitions for seed in part) == list(range(64))
    q = s['query_accounting']
    assert q['full_physical_noisy_union'] == 2073600 and q['official_noisy_label_sequence_after'] == 14687319+2073600 and q['official_noisy_label_sequence_before'] == 14687319
    assert q['single270_deployment'] == q['hard_triple90_deployment'] == q['posterior_triple90_deployment'] == q['shared_two_triple_methods'] == 1244160
    assert q['shared_first_query90'] == 414720 and q['extra_repeat_queries'] == 829440
    assert q['clean_unique_oracle_cells'] == 81*s['source_audit']['new_unique_source_ids'] <= 1399680
    assert s['all192_arms_retained'] and sum(len(row['complete_label_gates']) for row in s['raw_audits']) == 192
    assert q['clean_materialized_cell_instances'] == 1399680 and s['analytic_zero_proofs'] == s['complete_label_gates'] == 192
    progress = json.loads((PUBLIC/'execution/progress.json').read_text())
    assert progress['status'] == 'complete' and progress['active_pids'] == [] and progress['completed_commands'] == 14
    assert progress['completed_LPs'] == 7680 and progress['audited_banks'] == 64
    assert progress['wall_s'] == runtime['wall_s'] and progress['cpu_s'] == runtime['aggregate_cpu_s']
    from repeat_query_math import exact_channels
    from posterior_fresh_confirm_study import HERE
    assert json.loads((HERE/'repeat-query-channel.json').read_text()) == dict(status='passed', **exact_channels())
    noise = json.loads((PUBLIC/'noise-accounting.json').read_text())
    for name in ['first', 'repeat1', 'repeat2', 'decoded']:
        for key in ['full_cells', 'queried_cells', 'full_cell_errors', 'queried_cell_errors']:
            assert noise[name][key] == sum(row['realized_noise'][name][key] for row in s['raw_audits'])
        assert noise[name]['queried_error_rate'] == noise[name]['queried_cell_errors']/noise[name]['queried_cells']
        assert noise[name]['queried_error_offsets'] == [sum(row['realized_noise'][name]['queried_error_offsets'][i] for row in s['raw_audits']) for i in range(8)]
    assert noise['ties'] == {key:sum(row['realized_noise']['ties'][key] for row in s['raw_audits']) for key in ['all_cells', 'queried_cells']}
    assert noise['triple_query_patterns'] == {key:sum(row['triple_query_patterns'][key] for row in s['raw_audits']) for key in ['three_agree','two_agree','all_distinct']}
    timing = json.loads((PUBLIC/'solver-timing-accounting.json').read_text())
    records = [r for seed in range(64) for arm in ['single270', 'hard_triple90', 'posterior_triple90'] for r in json.loads((PUBLIC/f's{seed}/{arm}-receipt.json').read_text())['records']]
    cpu = sum(r['solver']['solver_cpu_s'] for r in records)
    assert timing['solver_calls'] == len(records) == 7680 and timing['solver_call_cpu_seconds'] == cpu
    assert timing['charged_aggregate_cpu_seconds'] == runtime['aggregate_cpu_s']
    assert timing['solver_cpu_fraction'] == cpu/runtime['aggregate_cpu_s'] and timing['remaining_cpu_seconds'] == runtime['aggregate_cpu_s']-cpu
    print(json.dumps(dict(status='passed', manifest_paths=len(m['sha256'])+1, portable_registry_mappings=2,
                          all_IDs_and64_full_masks_sealed_before_label_commands=True, resource_and_query_caps_passed=True)))


if __name__ == '__main__':
    main()
