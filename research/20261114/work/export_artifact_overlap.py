"""Publish exact H2bi artifact throughput without adding scientific evaluations."""
import json
import shutil
import study as S
from artifact_overlap_study import DEST,PP

PUBLIC=S.ROOT/'results/research/work/artifact-overlap'


def evidence():
    PUBLIC.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed'
    for name in ['protocol.json','audit.json','source-gate-gpu0.json','source-gate-gpu1.json','benchmark-gpu0.json','benchmark-gpu1.json','timing-worker0.json','timing-worker1.json','gate-worker0.json','gate-worker1.json','live-progress.json']:shutil.copyfile(DEST/name,PUBLIC/name)
    shutil.copyfile(S.OUT/'artifact_overlap_preflight/synthetic-gate.json',PUBLIC/'synthetic-gate.json');shutil.copytree(DEST/'public-fixtures',PUBLIC/'public-fixtures',dirs_exist_ok=True)


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text());assert v['status']=='passed' and v['all_timed_state_copies_compared']==49152;root=PUBLIC.parent;S.save(root/'artifact-overlap-summary.json',a);summary=json.loads((root/'summary.json').read_text());assert 'artifact_overlap' not in summary;summary['artifact_overlap']={'hypothesis':'H2bi','fresh_inputs':0,'new_scientific_evaluations':0,'median_complete_research_pipeline_speedup':a['median_paired_complete_pipeline_speedup'],'target':1.15,'decision':a['decision'],'byte_identical_scientific_npz_files':1368,'source_cohort_inputs':256,'source_models':3,'scope':'Bounded asynchronous lossless artifact writing on frozen H2bh shard00. Four paired timings; full per-replay compute, diagnostics, copy, hash, flush and join charged; 2 threads, 2 pending tasks and 8 GiB peak RSS limit. Model/data preload and warmup are excluded from the timing endpoint but included in worker budgets. No model-inference latency, fsync durability or new accuracy claim. The original H2bh absolute 99.9% endpoint remains inconclusive.'};summary['hypotheses'].append({'campaign':'artifact_overlap','hypothesis':'H2bi >=1.15 median paired complete research-pipeline speedup with exact scientific-archive parity and bounded memory','outcome':a['decision']});p=json.loads(PP.read_text());summary['protocols']['artifact-overlap-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bi','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(root/'summary.json',summary);S.save(root/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(f.relative_to(root)):S.sha(f) for f in sorted(root.rglob('*')) if f.is_file() and f!=root/'manifest.json'}})


if __name__=='__main__':
    import sys
    publish() if '--publish' in sys.argv else evidence()
