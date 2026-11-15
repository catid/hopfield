"""Portable H1bt proof, randomization, accounting and resource replay."""
import json
from signal_detection_study import ROOT, PUBLIC, PROTOCOL, sha
from audit_signal_detection import public_audit


def main():
    manifest = ROOT/'results/research/ripper/signal_detection_public_manifest.json'
    metadata = json.loads(manifest.read_text())
    for relative, digest in metadata['sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    assert sha(PUBLIC/'summary.json') == metadata['raw_audit_sha256']
    public_audit()
    saved = json.loads((PUBLIC/'summary.json').read_text())
    runtime = json.loads((PUBLIC/'runtime.json').read_text())
    assert runtime == saved['runtime'] and runtime['protocol_sha256'] == sha(PROTOCOL)
    assert runtime['status'] == 'validated' and runtime['wall_s'] < 600 and runtime['aggregate_cpu_s'] < 3600
    commands = runtime['commands']
    assert len(commands) == 65 and all(c['returncode'] == 0 for c in commands)
    assert all(c['action'] == 'case' for c in commands[:64]) and sorted(c['seed'] for c in commands[:64]) == list(range(64))
    assert commands[-1]['action'] == 'summary' and runtime['max_cpu_workers'] == 8 and runtime['gpu_seconds'] == 0
    protocol = json.loads(PROTOCOL.read_text())
    q = saved['query_accounting']
    assert q == protocol['query_accounting']
    for key in ['new_official_source_rows', 'new_official_cell_queries', 'new_clean_oracle_cells']:
        assert q[key] == 0
    assert q['synthetic_shuffled_entry_assignments'] == 51425280 and q['synthetic_shuffles'] == 3968
    assert q['new_IID_generated_cells'] == 466560 and q['new_IID_observed_union'] == 460721
    assert q['official_noisy_label_sequence_unchanged'] == 10540119
    assert q['ripper_source_ledger_unchanged'] == 2132001 and q['combined_source_ledger_unchanged'] == 2368426
    assert saved['analytic_zero_witnesses_checked'] == 4096
    assert sum(len(a['complete_label_gates']) for a in saved['raw_audits']) == 128
    assert saved['all_cases_retained'] is True
    print(json.dumps(dict(status='passed', manifest_paths=len(metadata['sha256'])+1,
                          all64_cases_retained=True, new_official_labels=0, resource_and_query_caps_passed=True)))


if __name__ == '__main__':
    main()
