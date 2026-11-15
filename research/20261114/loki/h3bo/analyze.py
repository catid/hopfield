"""Publish exact posterior/rank diagnostics and guarded numerical decision."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bo'


def main():
    frozen();PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():target=PUBLIC/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    g=read(PUBLIC/'capacity-gate.json');audit=verify(ROOT,False);keys=['decision','primary','threshold_bits','numerical_guard_bits','decimal_precision','diagnostic_cases','distinct_checkpoint_sets','context_cases','infeasible_context_cases','already_identified_context_cases','descriptive_MAP_mean_confidence','descriptive_MAP_lex_accuracy','descriptive_MAP_overconfidence','descriptive_fixed_bin_ECE','descriptive_actual_cost_change','new_source_draws','new_oracle_labels'];s=dict(status='validated',utc=now(),hypothesis='H3bo',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['png','svg','csv']});save(PUBLIC/'summary.json',s)
    rows=[dict(case=r['case'],candidate=r['candidate'],candidate_count=r['candidate_count'],true_rank_best=r['true_rank_best'],true_rank_worst=r['true_rank_worst'],posterior_logloss_bits=r['posterior_logloss_bits'],uniform_logloss_bits=r['uniform_logloss_bits'],excess_logloss_bits=r['excess_logloss_bits'],actual_cost_change=r['actual_cost_change'],MAP_lex_correct=r['MAP_lex_correct']) for r in read(PUBLIC/'case-diagnostics.json')]
    with (PUBLIC/'case-diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(dict(decision=s['decision'],mean_excess_bits=s['primary']['mean_excess_logloss_bits'],audit=audit))


if __name__=='__main__':main()
