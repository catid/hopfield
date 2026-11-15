"""Publish all complete catalogs and certificates, then independently audit them."""
import collections
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bi'


def main():
    frozen();PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():target=PUBLIC/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    g=read(PUBLIC/'capacity-gate.json');audit=verify(ROOT,False);keys=['decision','distinct_checkpoint_sets','unresolved_cases','feasible_checkpoint_cases','infeasible_checkpoint_cases','known_success_cases','global_memo_states','global_canonical_split_attempts','new_source_draws','new_oracle_labels'];s=dict(status='validated',utc=now(),hypothesis='H3bi',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,witness_worst_depth_group_counts=dict(sorted(collections.Counter(r['worst_leaf_depth'] for r in g['groups']).items())),witness_worst_depth_case_counts=dict(sorted(collections.Counter(d for r in g['groups'] for d in [r['worst_leaf_depth']]*len(r['case_ids'])).items())),audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['png','svg','csv']});save(PUBLIC/'summary.json',s)
    rows=[dict(group=r['group'],candidates=';'.join(map(str,r['candidates'])),case_count=len(r['case_ids']),case_ids=';'.join(map(str,r['case_ids'])),feasible=r['feasible'],witness_worst_depth=r['worst_leaf_depth'],signature_count=r['signature_count'],memo_states=r['memo_states'],split_attempts=r['canonical_split_attempts']) for r in g['groups']]
    with (PUBLIC/'tree-groups.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(decision=s['decision'],feasible=s['feasible_checkpoint_cases'],groups=s['distinct_checkpoint_sets'],audit=audit))


if __name__=='__main__':main()
