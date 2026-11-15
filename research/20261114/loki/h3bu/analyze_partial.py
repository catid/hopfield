"""Publish a computationally incomplete study without a partial primary endpoint."""
import csv
import hashlib
import json
from common import ROOT,SRC,read,save,sha,now
P=ROOT/'results/research/loki/h3bu'


def main():
    audit=read(ROOT/'artifacts/research/20261114/loki/h3bu-full-archive-audit.json');g=read(P/'capacity-gate.json');assert audit['scientific_status']=='incomplete' and audit['primary_decision'] is None and audit['all_actual_paid_queries_replayed']==g['actual_membership_calls'];deps=[SRC/'verify_partial.py',SRC/'verify_tree.py',SRC/'audit_trees.py',SRC/'audit_costs.py',SRC.parent/'archive_export_layer.py',SRC.parent/'export_layer.py',SRC.parent/'h3bj/verify_public.py',SRC.parent/'h3bl/verify_public.py']
    save(P/'full-archive-audit.json',dict(audit=audit,protocol_sha256=sha(SRC/'protocol.json'),proof_manifest_sha256=sha(P/'proof-export.json'),audit_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in deps},external_archive_required_for_full_replay=True))
    summary=dict(status='incomplete',primary_decision=None,hypothesis='H3bu',utc=now(),reason='Global 2,000,000-state cap; no full-cohort primary result',planned_cases=4096,completed_cases=len(g['completed_results']),pending_cases=4096-len(g['completed_results']),actual_membership_calls=g['actual_membership_calls'],global_memo_states=g['global_memo_states'],global_split_evaluations=g['global_split_evaluations'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,protocol_sha256=sha(SRC/'protocol.json'),audit=audit,audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),verifier_dependency_sha256={str(path.relative_to(ROOT)):sha(path) for path in deps},public_sha256={str(path.relative_to(P)):sha(path) for path in sorted(P.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['.png','.svg','.csv']});save(P/'summary.json',summary)
    completed={r['case']:r for r in g['completed_results']};rows=[]
    for status in read(P/'case-statuses.json'):
        case=status['case'];row=completed.get(case);rows.append(dict(case=case,status=status['status'],membership_calls=row['membership_calls'] if row else '',exact_true_singleton=row['exact_true_singleton'] if row else '',primary_eligible=False))
    with (P/'planned-case-statuses.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(status=summary['status'],primary_decision=None,completed_cases=summary['completed_cases'],pending_cases=summary['pending_cases']))


if __name__=='__main__':main()
