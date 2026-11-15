"""Publish every complete paired case with the external-proof boundary explicit."""
import csv
import gzip
import json
from common import ROOT,SRC,read,save,sha,now
P=ROOT/'results/research/loki/h3bw'


def main():
    audit=read(ROOT/'artifacts/research/20261114/loki/h3bw-full-archive-audit.json');g=read(P/'capacity-gate.json');raw=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());outcome=json.loads(raw.splitlines()[-1])['data'];assert audit['status']=='passed' and audit['actual_paid_queries']==outcome['new_policy_calls'] and audit['primary_decision']==outcome['decision'];deps=[SRC/name for name in ['verify_complete.py','verify_tree.py','verify_gates.py','audit_trees.py','audit_costs.py']]+[SRC.parent/'archive_export_layer.py',SRC.parent/'export_layer.py',SRC.parent/'h3bj/verify_public.py',SRC.parent/'h3bl/verify_public.py',SRC.parent/'h3bv/verify_tree.py'];save(P/'full-archive-audit.json',dict(audit=audit,protocol_sha256=sha(SRC/'protocol.json'),proof_manifest_sha256=sha(P/'proof-export.json'),audit_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in deps},external_archive_required_for_full_replay=True))
    summary=dict(status='validated',hypothesis='H3bw',utc=now(),**outcome,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,ledger_records=g['ledger_records'],ledger_sha256=g['ledger_sha256'],protocol_sha256=sha(SRC/'protocol.json'),audit=audit,audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),verifier_dependency_sha256={str(path.relative_to(ROOT)):sha(path) for path in deps},public_sha256={str(path.relative_to(P)):sha(path) for path in sorted(P.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['.png','.svg','.csv']});save(P/'summary.json',summary)
    rows=[]
    for case,old in zip(read(P/'case-results.json'),read(ROOT/'results/research/loki/h3br/case-results.json')):rows.append(dict(case=case['case'],candidate=case['candidate'],exact=case['exact_true_singleton'],handoff_MAP_calls=case['handoff_MAP_calls'],handoff_depth=case['handoff_depth'],MAP_calls=case['map_calls'],tree_calls=case['tree_calls'],new_policy_calls=case['membership_calls'],old_policy_calls=old['arms']['posterior']['policy_calls'],difference=case['membership_calls']-old['arms']['posterior']['policy_calls'],archived_query_reacquisitions=case['archived_query_reacquisitions']))
    with (P/'case-costs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(status=summary['status'],decision=summary['decision'],new_calls=summary['new_policy_calls'],old_calls=summary['old_policy_calls']))


if __name__=='__main__':main()
