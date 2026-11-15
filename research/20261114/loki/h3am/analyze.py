"""Publish paired evidence after public-only independent reconstruction."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3am'


def main():
    p=frozen('target');source=read(OUT/'source-outcomes.json');files=['capacity-gate.json','study-result.json','source-outcomes.json','source-validation.json','distribution-validation.json','filtering-completeness-seal.json','campaign.log']+[f'b{b}/source-patterns.npz'for b in range(4)]+[f"support{r['group']}/{name}"for r in source['supports']for name in('F.npz','capacity.npz')]
    for rel in files:
        dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    audit=verify(ROOT,require_summary=False);g=read(PUBLIC/'capacity-gate.json');summary=dict(status='validated',utc=now(),hypothesis='H3am',decision=g['decision'],rows=g['rows'],support_results=g['support_results'],total_paired_bank_masks=g['total_paired_bank_masks'],distinct_full_supports=g['distinct_full_supports'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,target_cohorts_created=0,raw_positive_labels_per_draw=32,attempted_D4_views_per_draw=256,same_seen_eligible_inputs=831605,reference_true_states=26960,scope=p['scope'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel)for rel in files})
    flat=[{k:v for k,v in r.items()if not isinstance(v,dict)}for r in summary['rows']]
    with(PUBLIC/'reliability.csv').open('w')as handle:
        w=csv.DictWriter(handle,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    save(PUBLIC/'summary.json',summary);print(dict(decision=summary['decision'],audit=audit))


if __name__=='__main__':main()
