"""Audit the lossless export, then summarize every paired case."""
import csv
import gzip
import json
from common import ROOT,SRC,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bt'


def main():
    audit=verify(ROOT,False);raw=gzip.decompress((PUBLIC/'query-ledger.jsonl.gz').read_bytes());outcome=json.loads(raw.splitlines()[-1])['data'];g=read(PUBLIC/'capacity-gate.json');deps=[ROOT/f'research/20261114/loki/{name}/verify_public.py' for name in ['h3bi','h3bj','h3bl']]+[SRC/'audit_costs.py',SRC.parent/'export_layer.py']
    s=dict(status='validated',utc=now(),hypothesis='H3bt',**outcome,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,ledger_records=g['ledger_records'],ledger_sha256=g['ledger_sha256'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),verifier_dependency_sha256={str(path.relative_to(ROOT)):sha(path) for path in deps},public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['png','svg','csv']});save(PUBLIC/'summary.json',s)
    cases=read(PUBLIC/'case-results.json');comparisons=read(PUBLIC/'baseline-comparison.json');rows=[]
    for case,comparison in zip(cases,comparisons):rows.append(dict(case=case['case'],candidate=case['candidate'],exact=case['exact_true_singleton'],handoff_MAP_calls=case['handoff_MAP_calls'],guard_evaluations=len(case['cost_guards']),guard_rejections=sum(not item['handoff'] for item in case['cost_guards']),new_MAP_calls=case['map_calls'],new_tree_calls=case['tree_calls'],old_policy_calls=comparison['old_policy_calls'],new_policy_calls=case['membership_calls'],difference=case['membership_calls']-comparison['old_policy_calls'],archived_query_reacquisitions=case['archived_query_reacquisitions']))
    with (PUBLIC/'case-costs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(decision=s['decision'],old_calls=s['old_policy_calls'],new_calls=s['new_policy_calls'],audit=audit),flush=True)


if __name__=='__main__':main()
