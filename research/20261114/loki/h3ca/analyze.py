"""Summarize the completely audited tree-weight intervention without exclusions."""
import csv
import gzip
import json
from common import ROOT,SRC,read,save,sha,now
from describe import describe
P=ROOT/'results/research/loki/h3ca';A=ROOT/'artifacts/research/20261114/loki'


def main():
    audit=read(A/'h3ca-full-raw-audit.json');export=read(A/'h3ca-export-audit.json')
    g=read(P/'capacity-gate.json');p=read(SRC/'protocol.json')
    outcome=json.loads(gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes()).splitlines()[-1])['data']
    assert audit['status']=='passed_complete_fresh_two_score_two_arm_audit' and export['status']=='passed_lossless_scientific_export'
    assert audit['decision']==outcome['decision'] and audit['actual_paid_queries']==outcome['actual_membership_calls']
    assert g['primary_decision_emitted'] and not g['internally_complete_but_not_accepted']
    names=['verify_complete.py','verify_public.py','verify_tree.py','verify_gates.py','audit_io.py','audit_queries.py',
           'audit_sources.py','audit_runtime.py','audit_arm.py','audit_namespace_gate.py','audit_trees.py','audit_costs.py',
           'audit_pairing.py','audit_cap_gate.py','role_gate.py','audit_baseline_gate.py','audit_export.py','export.py','parallel_export.py','describe.py']
    paths=[SRC/name for name in names]+[SRC.parent/name for name in ['archive_export_layer.py','export_layer.py',
           'h3bj/verify_public.py','h3bi/verify_public.py','h3bl/verify_public.py','h3bv/verify_tree.py']]
    hashes={str(x.relative_to(ROOT)):sha(x) for x in paths}
    save(P/'full-archive-audit.json',dict(scientific_audit=audit,lossless_export_audit=export,
         protocol_sha256=sha(SRC/'protocol.json'),proof_manifest_sha256=sha(P/'proof-export.json'),audit_source_sha256=hashes,
         full_replay_requires_external_archive=True,audit_sequence='Complete immutable raw-science replay and independent lossless source-to-export verification'))
    pairs=read(P/'case-results.json');rows={a:read(P/a/'arm-results.json') for a in ['H','T']}
    strata=[]
    for k in [0,1,2]:
        subset=[r for r in pairs if len(p['truths'][r['candidate']]['orbits'])==k]
        strata.append(dict(true_orbit_count=k,cases=len(subset),H_calls=sum(r['H_calls'] for r in subset),T_calls=sum(r['T_calls'] for r in subset),
            H_exact=sum(r['H_exact'] for r in subset),T_exact=sum(r['T_exact'] for r in subset),
            H_only_exact=sum(r['H_exact'] and not r['T_exact'] for r in subset),T_only_exact=sum(r['T_exact'] and not r['H_exact'] for r in subset)))
    summary=dict(status='validated',hypothesis='H3ca',utc=now(),**outcome,descriptive_by_true_orbit_count=strata,
                 descriptive=describe(pairs,rows,p),aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,
                 ledger_records=g['ledger_records'],ledger_sha256=g['ledger_sha256'],protocol_sha256=sha(SRC/'protocol.json'),
                 audit=audit,export_audit=export,analyze_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),
                 audit_dependency_sha256=hashes,public_sha256={str(x.relative_to(P)):sha(x) for x in sorted(P.rglob('*')) if x.is_file()
                     and x.name not in ['summary.json','README.md','plot-provenance.json'] and x.suffix not in ['.png','.svg','.csv']})
    save(P/'summary.json',summary)
    table=[]
    for pair,h,t in zip(pairs,rows['H'],rows['T']):
        expected=pair['matched_expected_tree_costs']
        table.append(dict(case=pair['case'],candidate=pair['candidate'],H_calls=pair['H_calls'],T_calls=pair['T_calls'],
            difference_T_minus_H=pair['T_calls']-pair['H_calls'],H_exact=pair['H_exact'],T_exact=pair['T_exact'],
            H_MAP_calls=h['map_calls'],T_MAP_calls=t['map_calls'],H_tree_calls=h['tree_calls'],T_tree_calls=t['tree_calls'],
            handoff_MAP_calls=h['handoff_MAP_calls'],H_matched_expectation=expected['H_expected'] if expected else '',
            T_matched_expectation=expected['T_expected'] if expected else ''))
    for name,data in [('case-costs.csv',table),('orbit-count-costs.csv',strata),('handoff-costs.csv',summary['descriptive']['by_handoff'])]:
        with (P/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(dict(status='validated',decision=summary['decision'],H=summary['H_calls'],T=summary['T_calls'],exact=summary['exact_transitions'],
               differences=summary['descriptive']['actual_query_difference_T_minus_H']))


if __name__=='__main__':main()
