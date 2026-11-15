"""Preserve a pre-target technical failure without inventing a scientific result."""
import shutil
from common import ROOT,SRC,OUT,read,save,sha
PUBLIC=ROOT/'results/research/loki/h3ap'
def main():
    g=read(OUT/'capacity-gate.json');r=read(OUT/'study-result.json');assert g['status']==r['status']=='failed_or_incomplete'and r['completed_rows']==r['completed_supports']==[]and not(OUT/'source-outcomes.json').exists()and not(OUT/'construction-safety-seal.json').exists();files=[str(path.relative_to(OUT))for path in OUT.rglob('*')if path.is_file()]
    for rel in files:dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    save(PUBLIC/'summary.json',dict(status='verified_technical_failure',scientific_decision='inconclusive_not_evaluated',reason='numpy.int64 retained offset entries cannot be JSON serialized',source_outcomes_sealed=False,target_capacity_stage_reached=False,scored_rows=0,partial_artifacts_usable=False,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,protocol_sha256=sha(SRC/'protocol.json'),public_sha256={rel:sha(PUBLIC/rel)for rel in files}))
    report='''# H3ap: technical failure before target evaluation

**The scientific hypothesis was not evaluated.** The source-only constructor failed while serializing its first raw-unfiltered construction receipt: the retained offset list contains `numpy.int64` values, which Python JSON rejects. This is an implementation failure, not evidence against the ≥95% capacity hypothesis.

The frozen runner and protocol remain unchanged. The monitored attempt consumed **0.149611 aggregate CPU seconds / 0.161599 wall seconds**, with **zero GPU work**. No source-outcomes seal was produced, so the subsequent target-integrity reads, safety-construction check, complete capacity evaluation and decisions did not run. There are zero scored rows. Partial source model arrays and their logs are retained as unusable evidence; they are not published as completed experimental outcomes.

The exact error is `TypeError: Object of type int64 is not JSON serializable`, originating from `source.py` when `common.save` serializes the construction receipt. The raw-unfiltered support is already constructed at that point; the failure concerns receipt encoding, not a model-selection rule. No target outcome was inspected to choose a fix.

A separately frozen successor can convert receipt offsets to ordinary Python integers and rerun the identical hypothesis, arms, inputs and budgets after a synthetic serialization gate. Such a successor does not retroactively complete H3ap or overwrite this failure.

[Frozen protocol](protocol.json), [unchanged constructor](construct.py), [failed source process](source.py), [stage order](study.py), [tracked failure summary](../../../../results/research/loki/h3ap/summary.json), [full traceback](../../../../results/research/loki/h3ap/campaign.log), [failed resource gate](../../../../results/research/loki/h3ap/capacity-gate.json), [public failure verifier](verify_public.py).
'''
    (SRC/'REPORT.md').write_text(report);(PUBLIC/'README.md').write_text('# H3ap: preserved technical failure\n\n[Full report](../../../../research/20261114/loki/h3ap/REPORT.md). Source-only JSON serialization failed before target evaluation; zero scored rows, no scientific decision. Frozen code/protocol unchanged; partial source artifacts unusable.0.149611CPU s/0.161599wall s,0GPU.\n')
if __name__=='__main__':main()
