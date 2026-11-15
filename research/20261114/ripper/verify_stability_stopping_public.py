"""Portable H1bu replay with exact aliases for the two sealed registry files."""
import json
import audit_stability_stopping as audit
from stability_stopping_study import ROOT, PUBLIC, REGISTRY, PROTOCOL, sha


def portable_verify(raw=False):
    assert not raw, 'Original-label audits require original raw artifacts'
    p = json.loads(PROTOCOL.read_text())
    mapping = {str((REGISTRY/name).relative_to(ROOT)): PUBLIC/name for name in ['excluded_ids.npz', 'ledger.json']}
    for relative, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(mapping.get(relative, ROOT/relative)) == digest, relative
    return p


def portable_job(seed):
    audit.verify = portable_verify
    return audit.audit_bank(seed, True)


def main():
    manifest = ROOT/'results/research/ripper/stability_stopping_public_manifest.json'
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
    assert len(commands) == 68 and all(c['returncode'] == 0 for c in commands)
    assert commands[0]['action'] == 'select' and all(c['action'] == 'predraw' for c in commands[1:33])
    assert sorted(c['seed'] for c in commands[1:33]) == list(range(32))
    assert commands[33]['action'] == 'materialize' and commands[34]['action'] == 'source-audit'
    assert all(c['action'] == 'bank' for c in commands[35:67]) and sorted(c['seed'] for c in commands[35:67]) == list(range(32))
    assert commands[-1]['action'] == 'summary'
    q = s['query_accounting']
    assert q['full_physical_noisy_union'] == 829440 and q['official_noisy_label_sequence'] == 10540119+829440
    assert 414720 <= q['virtual_stopped_queries'] <= 829440 and q['virtual_stopped_queries'] == 72*sum(row['stop_rows'] for row in s['stopped'])
    assert q['clean_unique_oracle_cells'] == 81*s['source_audit']['new_unique_source_ids'] <= 933120
    assert q['clean_materialized_cell_instances'] == 933120 and s['analytic_zero_proofs'] == s['complete_label_gates'] == 128
    print(json.dumps(dict(status='passed', manifest_paths=len(m['sha256'])+1, portable_registry_mappings=2,
                          all_IDs_and32_full_masks_sealed_before_label_commands=True, resource_and_query_caps_passed=True)))


if __name__ == '__main__':
    main()
