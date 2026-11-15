"""Seal source-only H2br compatibility packet before either GPU gate starts."""
import ast
import json
import shutil
from pathlib import Path
import study as S
DEST=S.OUT/'bp_first_endpoint_dispatch'
APPROVED='a3927424310f2b475e0519689502dc095862d02dab4c6711b85c54e9d6860eda'


def modules():
    names={p.name for p in S.HERE.glob('*endpoint_dispatch*.py')};pending=list(names)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((S.HERE/name).read_text())):
            imports=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in imports:
                file=(module or '').split('.')[0]+'.py'
                if file not in names and (S.HERE/file).exists():names.add(file);pending.append(file)
    return sorted(names)


def prepare():
    target=DEST/'source-compatibility-packet.json';assert not target.exists()
    proposal=S.HERE/'bp-first-endpoint-dispatch-proposal.json';assert S.sha(proposal)==APPROVED
    parent=json.loads((S.OUT/'bp_first_endpoint_shift/protocol.json').read_text())
    evidence=[proposal,DEST/'initial-design-and-seed-registry.json',DEST/'seed-registry.json']
    for name in ['policy-synthetic-gate.json','generator-synthetic-gate.json','accounting-synthetic-gate.json','compatibility-synthetic-gate.json']:
        file=DEST/name;gate=json.loads(file.read_text());assert gate['status']=='passed' and gate['fresh_inputs_drawn']==0
        for source,digest in gate['source_sha256'].items():assert S.sha(S.HERE/source)==digest,source
        evidence.append(file)
    registry=json.loads((DEST/'seed-registry.json').read_text());assert registry['reserved_seed_base']==2026111532 and registry['existing_protocol_references']==registry['fresh_inputs_drawn']==0
    assert len(registry['prior_data_manifests'])==29
    sources=[]
    for name,artifact,public in [('corner','bp_first_fresh_confirmation','bp-first-fresh-confirmation'),('uniform','bp_first_endpoint_shift','bp-first-endpoint-shift')]:
        directory=S.OUT/artifact;public_directory=S.ROOT/'results/research/work'/public
        audit=json.loads((directory/'audit.json').read_text());assert audit['status']=='passed'
        for filename in ['protocol.json','audit.json','datasets.json','sealed-replays.json']:evidence.append(directory/filename)
        receipts={}
        for mode in ['LP96','BP_geometry_LP96']:
            relative=Path('runs')/f'pair0-{mode}'/'result.json';file=directory/relative;published=public_directory/relative
            if not file.exists():file.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(published,file)
            assert S.sha(file)==S.sha(published);receipts[mode]=S.sha(file);evidence.append(file)
        sources.append({'name':name,'directory':str(directory.relative_to(S.ROOT)),
            'datasets':json.loads((directory/'datasets.json').read_text())['datasets'],'cohort_receipts_sha256':receipts})
    p={'status':'ready_for_parent_source_compatibility_review','approved_proposal_sha256':APPROVED,
       'created_utc':S.utc(),'wall_budget_seconds_per_gpu':240,'worker_rss_limit_gib':8,'host_barrier_timeout_seconds':60,
       'models':parent['models'],'sources':sources,'source_sha256':{**parent['source_sha256'],**{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in modules()}},
       'source_evidence_sha256':{**parent['source_evidence_sha256'],**{str(file.relative_to(S.ROOT)):S.sha(file) for file in evidence}},
       'fresh_inputs_drawn':0,'scientific_timing_pairs':0,'source_model_inputs_per_arm_per_gpu':384,
       'scope':'Only the approved384immutable H2bo/H2bp source model-inputs per actual arm per GPU. Full source arrays/subset trajectories mapped to original identities. Complete gate attempt retained, no new study draw/timing authorized by this packet.'}
    (DEST/'source-compatibility-attempt1').mkdir(exist_ok=False)
    S.save(target,p);print(json.dumps({'packet_sha256':S.sha(target),'source_bindings':len(p['source_sha256']),'evidence_bindings':len(p['source_evidence_sha256']),'fresh_inputs_drawn':0},indent=2))


if __name__=='__main__':prepare()
