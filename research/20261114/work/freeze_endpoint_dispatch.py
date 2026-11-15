"""H2br final source/evidence candidate; fresh draws require parent final review."""
import argparse
import ast
import copy
import json
from pathlib import Path
import study as S
DEST=S.OUT/'bp_first_endpoint_dispatch'
PP=S.HERE/'bp-first-endpoint-dispatch-protocol.json'
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


def freeze(preview=False):
    assert not PP.exists() and not (DEST/'protocol.json').exists()
    proposal=S.HERE/'bp-first-endpoint-dispatch-proposal.json';assert S.sha(proposal)==APPROVED;p=json.loads(proposal.read_text())
    packet_file=DEST/'source-compatibility-packet.json';packet=json.loads(packet_file.read_text());ph=S.sha(packet_file)
    for key in ['source_sha256','source_evidence_sha256']:
        for path,digest in packet[key].items():assert S.sha(S.ROOT/path)==digest,path
    registry=json.loads((DEST/'seed-registry.json').read_text());assert registry['reserved_seed_base']==2026111532 and registry['existing_protocol_references']==registry['fresh_inputs_drawn']==0
    assert len(registry['prior_data_manifests'])==29
    for item in registry['protocol_registry']+registry['prior_data_manifests']:assert S.sha(S.ROOT/item['path'])==item['sha256']
    evidence=[proposal,packet_file,DEST/'seed-registry.json',DEST/'initial-design-and-seed-registry.json']
    for name in ['policy-synthetic-gate.json','generator-synthetic-gate.json','accounting-synthetic-gate.json','compatibility-synthetic-gate.json','source-archive-preflight.json','source-cost-preflight.json','study-synthetic-gate.json']:
        file=DEST/name;gate=json.loads(file.read_text());assert gate['status']=='passed' and gate['fresh_inputs_drawn']==0
        for source,digest in gate['source_sha256'].items():assert S.sha(S.HERE/source)==digest,source
        evidence.append(file)
    for gpu in [0,1]:
        base=DEST/'source-compatibility-attempt1';gate=json.loads((base/f'gate-gpu{gpu}.json').read_text());worker=json.loads((base/f'gate-worker{gpu}.json').read_text())
        assert gate['status']=='passed' and worker['status']=='complete' and gate['protocol_sha256']==worker['protocol_sha256']==ph
        assert gate['source_model_inputs_per_arm']==384 and worker['worker_wall_seconds']<=240 and worker['peak_cpu_rss_kib']<=8*1024**2
        evidence.extend([base/f'gate-gpu{gpu}.json',base/f'gate-worker{gpu}.json'])
        for record in gate['receipts']:
            file=S.ROOT/record['result_path'];assert S.sha(file)==record['result_sha256'];evidence.append(file)
    evidence.extend([DEST/'source-compatibility-attempt1'/'live-gate.json',DEST/'source-compatibility-attempt1'/'launcher-gate.json'])
    parent=json.loads((S.OUT/'bp_first_endpoint_shift/protocol.json').read_text())
    schedule=copy.deepcopy(p['paired_schedule']);mapping={'LP96':'LP96','always_cascade':'BP_geometry_LP96','dispatcher':'endpoint_dispatch'}
    for block in schedule:block['order']=[mapping[mode] for mode in block['order']]
    assert len(schedule)==6 and len({tuple(b['order']) for b in schedule})==6
    for gpu in [0,1]:
        selected=[b for b in schedule if b['gpu']==gpu]
        for position in range(3):assert {b['order'][position] for b in selected}==set(mapping.values())
    cohort=copy.deepcopy(p['cohort']);cohort['prior_registry']=registry['prior_data_manifests']
    p.update(status='frozen_before_fresh_mixture_draws_and_timing',frozen_utc=S.utc(),approved_initial_proposal_sha256=APPROVED,
        compatibility_packet_sha256=ph,models=packet['models'],sources=packet['sources'],arms=parent['arms'],jobs=parent['jobs'],
        paired_schedule=schedule,budgets=p['proposed_budgets'],cohort=cohort,primary_accuracy_block=0,planned_fresh_jobs=48,
        primary_fresh_evaluations=49152,logical_arms=['BP_AdamW','LP96','cascade','dispatcher'],
        diagnostic_timing_scope='Per-component timings mean operation components of full cohorts; shared mixed-batch costs are not attributed to individual inputs or mixture strata.',
        inherited_job_gpu_field='Old source allocation metadata only. Actual fresh work is solely the six-block schedule; block0 accuracy is entirely GPU0.')
    p['source_sha256']={**packet['source_sha256'],**{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in modules()}}
    for seed in range(3):
        for arm in ['pl','old_binary']:
            source=S.OUT/'bp_first_endpoint_shift'/f'model-seed{seed}-{arm}.npz';target=DEST/source.name
            if not target.exists():
                with target.open('xb') as stream:stream.write(source.read_bytes())
            assert S.sha(source)==S.sha(target);evidence.append(target)
    p['source_evidence_sha256']={**packet['source_evidence_sha256'],**{str(file.relative_to(S.ROOT)):S.sha(file) for file in evidence}}
    if preview:
        p.update(status='candidate_awaiting_parent_final_review',frozen_utc=None);target=DEST/'candidate-protocol.json';assert not target.exists();S.save(target,p)
        print(json.dumps({'status':p['status'],'candidate_sha256':S.sha(target),'source_bindings':len(p['source_sha256']),'evidence_bindings':len(p['source_evidence_sha256']),'fresh_inputs_drawn':0},indent=2));return
    candidate=json.loads((DEST/'candidate-protocol.json').read_text())
    compare=copy.deepcopy(p);compare.update(status=candidate['status'],frozen_utc=None);assert compare==candidate,'Reviewed candidate changed'
    S.save(PP,p);S.save(DEST/'protocol.json',p);print(json.dumps({'status':p['status'],'protocol_sha256':S.sha(PP),'fresh_inputs_drawn':0},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');freeze(parser.parse_args().preview)
