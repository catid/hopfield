"""Export all exact ceilings and pairwise existence witnesses."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bg'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','campaign.log','N13-partition.json','case-ceilings.json','pair-difference-witnesses.json']
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    audit=verify(ROOT,False);keys=['decision','truth_results','cases','known_success_cases','failed_cases','failed_cases_identifiable','unavoidable_ambiguous_cases','global_N13_equivalence_classes','global_nontrivial_classes','distinct_truth_competitor_pairs','new_oracle_labels'];s=dict(status='validated',utc=now(),hypothesis='H3bg',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with (PUBLIC/'truth-ceilings.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['truth_results'][0]));writer.writeheader();writer.writerows(s['truth_results'])
    print(dict(decision=s['decision'],failed_identifiable=s['failed_cases_identifiable'],unavoidable=s['unavoidable_ambiguous_cases'],global_classes=s['global_N13_equivalence_classes'],pairs=s['distinct_truth_competitor_pairs'],audit=audit))


if __name__=='__main__':main()
