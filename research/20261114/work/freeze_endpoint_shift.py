"""H2bp source freeze, only after final review; no new data is drawn here."""
import argparse
import ast
import copy
import json
import study as S
DEST=S.OUT/'bp_first_endpoint_shift'
PARENT=S.OUT/'bp_first_fresh_confirmation'
PP=S.HERE/'bp-first-endpoint-shift-protocol.json'


def modules():
    names={p.name for p in S.HERE.glob('*endpoint_shift*.py')};pending=list(names)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((S.HERE/name).read_text())):
            imports=([node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                file=(module or '').split('.')[0]+'.py'
                if file not in names and (S.HERE/file).exists():names.add(file);pending.append(file)
    return sorted(names)


def freeze(preview=False):
    assert not PP.exists() and not (DEST/'protocol.json').exists()
    proposal=S.HERE/'bp-first-endpoint-shift-proposal.json';p=json.loads(proposal.read_text())
    parent=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((PARENT/'audit.json').read_text())
    assert audit['status']=='passed' and audit['protocol_sha256']==S.sha(PARENT/'protocol.json')
    registry=json.loads((DEST/'seed-registry.json').read_text());gate=json.loads((DEST/'synthetic-gate.json').read_text());generator=json.loads((DEST/'generator-synthetic-gate.json').read_text())
    assert registry['status']==gate['status']==generator['status']=='passed'
    assert registry['reserved_seed_base']==2026111530 and registry['existing_protocol_references']==0
    assert registry['fresh_inputs_drawn']==gate['fresh_inputs_drawn']==generator['fresh_inputs_drawn']==0
    assert gate['complete_answer_segment_AST_identical_to_frozen_H2bn']
    for proof in [gate,generator]:
        for name,digest in proof['source_sha256'].items():assert S.sha(S.HERE/name)==digest,name
    for item in registry['protocol_registry']+registry['prior_data_manifests']:assert S.sha(S.ROOT/item['path'])==item['sha256']
    assert registry['source_sha256']==S.sha(S.HERE/'audit_endpoint_shift_seed_registry.py')
    cohort=copy.deepcopy(p['cohort']);cohort['prior_registry']=registry['prior_data_manifests']
    p.update(status='frozen_before_source_gates_and_uniform_endpoint_draws',frozen_utc=S.utc(),
        models=parent['models'],arms=parent['arms'],jobs=parent['jobs'],source_jobs=parent['source_jobs'],
        source_datasets=parent['source_datasets'],paired_schedule=copy.deepcopy(parent['paired_schedule']),
        budgets=p['proposed_budgets'],cohort=cohort,primary_accuracy_pair=0,planned_fresh_jobs=48,primary_fresh_evaluations=36864,
        parent_protocol_sha256=S.sha(PARENT/'protocol.json'),freeze_reads_source_outcome_arrays=False,
        inherited_job_gpu_field='H2bh source-allocation metadata only; actual fresh execution is governed solely by paired_schedule.')
    p['source_sha256']={**parent['source_sha256'],**{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in modules()}}
    evidence=[proposal,DEST/'approved-initial-design.json',DEST/'seed-registry.json',DEST/'synthetic-gate.json',DEST/'generator-synthetic-gate.json',PARENT/'protocol.json',PARENT/'audit.json',PARENT/'sealed-replays.json']
    for seed in range(3):
        for arm in ['pl','old_binary']:
            source=PARENT/f'model-seed{seed}-{arm}.npz';target=DEST/source.name
            if not target.exists():
                with target.open('xb') as stream:stream.write(source.read_bytes())
            assert S.sha(source)==S.sha(target);evidence.append(target)
    p['source_evidence_sha256']={**parent['source_evidence_sha256'],**{str(q.relative_to(S.ROOT)):S.sha(q) for q in evidence}}
    if preview:
        p.update(status='candidate_awaiting_parent_final_review',frozen_utc=None);target=DEST/'candidate-protocol.json'
        S.save(target,p);print(json.dumps({'status':p['status'],'candidate_sha256':S.sha(target),'fresh_inputs_drawn':0}));return
    S.save(PP,p);S.save(DEST/'protocol.json',p)
    print(json.dumps({'status':p['status'],'protocol_sha256':S.sha(PP),'fresh_inputs_drawn':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');freeze(parser.parse_args().preview)
