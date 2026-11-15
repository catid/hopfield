"""Summarize the complete fresh paired endpoint after independent auditing."""
import csv
import gzip
import json
import shutil
from common import ROOT, SRC, read, save, sha, now
P = ROOT/'results/research/loki/h3by'
A = ROOT/'artifacts/research/20261114/loki'


def main():
    audit = read(A/'h3by-full-raw-audit.json')
    export = read(A/'h3by-export-audit.json')
    g = read(P/'capacity-gate.json')
    shutil.copyfile(SRC/'seed-scope-note.json', P/'seed-scope-note.json')
    ledger = gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes())
    outcome = json.loads(ledger.splitlines()[-1])['data']
    assert audit['status'] == 'passed_complete_fresh_two_arm_audit'
    assert export['status'] == 'passed_lossless_scientific_export'
    assert audit['actual_paid_queries'] == outcome['actual_membership_calls']
    assert audit['primary_decision'] == outcome['decision']
    names = ['verify_complete.py', 'verify_public.py', 'verify_tree.py', 'verify_gates.py',
             'audit_io.py', 'audit_queries.py', 'audit_sources.py', 'audit_runtime.py',
             'audit_original.py', 'audit_trees.py', 'audit_costs.py', 'audit_baseline_gate.py', 'audit_export.py', 'export.py', 'parallel_export.py', 'seed-scope-note.json']
    deps = [SRC/name for name in names]+[SRC.parent/name for name in
            ['archive_export_layer.py', 'export_layer.py', 'h3bj/verify_public.py',
             'h3bi/verify_public.py', 'h3bl/verify_public.py', 'h3bv/verify_tree.py']]
    deps.append(ROOT/'research/20261114/ripper/protocol-signal-detection.json')
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in deps}
    save(P/'full-archive-audit.json', dict(scientific_audit=audit, lossless_export_audit=export,
         protocol_sha256=sha(SRC/'protocol.json'), proof_manifest_sha256=sha(P/'proof-export.json'),
         audit_source_sha256=hashes, full_replay_requires_external_archive=True,
         audit_sequence='Complete immutable raw-science replay, followed by independent lossless source-to-export identity verification'))
    protocol = read(SRC/'protocol.json')
    pairs = read(P/'case-results.json')
    descriptive = []
    for k in [0, 1, 2]:
        subset = [r for r in pairs if len(protocol['truths'][r['candidate']]['orbits']) == k]
        D, B = sum(r['deeper_calls'] for r in subset), sum(r['baseline_calls'] for r in subset)
        descriptive.append(dict(true_orbit_count=k, cases=len(subset), deeper_calls=D, baseline_calls=B,
                                exact_cases=sum(r['deeper_exact'] for r in subset), saving_calls=B-D,
                                scope='Descriptive true-orbit stratum only; not an additional scientific endpoint'))
    summary = dict(descriptive_by_true_orbit_count=descriptive, status='validated', hypothesis='H3by', utc=now(), **outcome,
                   aggregate_CPU_seconds=g['aggregate_cpu_seconds'], wall_seconds=g['wall_seconds'], GPU_seconds=0,
                   ledger_records=g['ledger_records'], ledger_sha256=g['ledger_sha256'],
                   protocol_sha256=sha(SRC/'protocol.json'), audit=audit, export_audit=export,
                   analyze_source_sha256=sha(__file__), verifier_source_sha256=sha(SRC/'verify_public.py'),
                   audit_dependency_sha256=hashes,
                   public_sha256={str(path.relative_to(P)): sha(path) for path in sorted(P.rglob('*'))
                     if path.is_file() and path.name not in ['summary.json', 'README.md', 'plot-provenance.json']
                     and path.suffix not in ['.png', '.svg', '.csv']})
    save(P/'summary.json', summary)
    with (P/'orbit-count-costs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(descriptive[0]))
        writer.writeheader()
        writer.writerows(descriptive)
    rows = []
    for pair, new, old in zip(read(P/'case-results.json'), read(P/'deeper-results.json'), read(P/'baseline-results.json')):
        rows.append(dict(case=pair['case'], candidate=pair['candidate'], exact=pair['deeper_exact'],
                    deeper_calls=pair['deeper_calls'], baseline_calls=pair['baseline_calls'],
                    difference=pair['deeper_calls']-pair['baseline_calls'],
                    deeper_MAP_calls=new['map_calls'], deeper_tree_calls=new['tree_calls'],
                    baseline_MAP_calls=old['map_calls'], baseline_tree_calls=old['tree_calls'],
                    handoff_MAP_calls=new['handoff_MAP_calls'], handoff_depth=new['handoff_depth'],
                    independently_paid_cross_arm_reacquisitions=pair['independently_paid_cross_arm_reacquisitions']))
    with (P/'case-costs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dict(status=summary['status'], decision=summary['decision'],
               deeper=summary['deeper_policy_calls'], baseline=summary['baseline_policy_calls']))


if __name__ == '__main__':
    main()
