"""Export exact optimality certificates and representation-repair evidence."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bl'


def main():
    frozen();PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():target=PUBLIC/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    g=read(PUBLIC/'capacity-gate.json');audit=verify(ROOT,False);keys=['decision','distinct_checkpoint_sets','unresolved_cases','known_four_query_success_cases','case_weighted_expected_old_sum','case_weighted_expected_optimal_sum','mean_expected_old_continuation','mean_expected_optimal_continuation','relative_expected_reduction','exact_threshold_comparison','strictly_improved_sets','strictly_improved_checkpoint_cases','guaranteed_nonincrease_is_correctness_not_hypothesis','descriptive_old_continuation_cost','descriptive_optimized_continuation_cost','descriptive_baseline_MAP_continuation_cost','global_memo_states','global_canonical_split_attempts','new_source_draws','new_oracle_labels'];dependency=ROOT/'research/20261114/loki/h3bi/verify_public.py';s=dict(status='validated',utc=now(),hypothesis='H3bl representation repair of H3bk',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),verifier_dependency_sha256={str(dependency.relative_to(ROOT)):sha(dependency)},public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['png','svg','csv']});save(PUBLIC/'summary.json',s)
    rows=[dict(group=r['group'],candidate_count=len(r['candidates']),checkpoint_cases=r['case_multiplicity'],old_expected=r['old_expected_additional_queries']['display_float'],optimal_expected=r['optimal_expected_additional_queries']['display_float'],strict_improvement=r['strict_expected_improvement'],old_worst_depth=r['old_worst_leaf_depth'],optimal_worst_depth=r['optimal_worst_leaf_depth'],states=r['memo_states'],splits=r['canonical_split_attempts']) for r in g['groups']]
    with (PUBLIC/'group-costs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(decision=s['decision'],expected_relative_reduction=s['relative_expected_reduction']['display_float'],audit=audit))


if __name__=='__main__':main()
