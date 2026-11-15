"""Freeze H2bo only after the final parent source review; never draws data."""
import argparse
import ast
import copy
import json
import study as S

DEST = S.OUT / 'bp_first_fresh_confirmation'
PARENT = S.OUT / 'bp_first_geometry_cascade'
SOURCE = S.OUT / 'loop80_confirmation'
PP = S.HERE / 'bp-first-fresh-confirmation-protocol.json'


def modules():
    names = {p.name for p in S.HERE.glob('*bp_first_fresh*.py')}
    pending = list(names)
    while pending:
        name = pending.pop()
        tree = ast.parse((S.HERE / name).read_text())
        for node in ast.walk(tree):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                file = (module or '').split('.')[0] + '.py'
                if file not in names and (S.HERE / file).exists():
                    names.add(file)
                    pending.append(file)
    return sorted(names)


def freeze(preview=False):
    assert not PP.exists() and not (DEST / 'protocol.json').exists()
    proposal = S.HERE / 'bp-first-fresh-confirmation-proposal.json'
    p = json.loads(proposal.read_text())
    parent = json.loads((PARENT / 'protocol.json').read_text())
    source = json.loads((SOURCE / 'protocol.json').read_text())
    audit = json.loads((PARENT / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['protocol_sha256'] == S.sha(PARENT / 'protocol.json')
    gate = json.loads((DEST / 'synthetic-gate.json').read_text())
    statistics = json.loads((DEST / 'statistics-gate.json').read_text())
    registry = json.loads((DEST / 'seed-registry.json').read_text())
    compatibility = json.loads((DEST / 'source-format-compatibility.json').read_text())
    assert compatibility['status'] == 'passed' and compatibility['fresh_inputs_drawn'] == 0
    assert compatibility['source_sha256'] == S.sha(S.HERE / 'audit_bp_first_fresh_states.py')
    assert compatibility['driver_sha256'] == S.sha(DEST / 'source_format_preflight.py')
    provenance = json.loads((DEST / 'source-format-provenance.json').read_text())
    assert provenance['status'] == 'passed' and provenance['files'] == 72
    assert gate['status'] == statistics['status'] == registry['status'] == 'passed'
    assert gate['synthetic_only'] and gate['fresh_inputs_drawn'] == registry['fresh_inputs_drawn'] == 0
    assert gate['complete_answer_segment_AST_identical_to_frozen_H2bn']
    assert not gate['fresh_outcomes_read'] and not statistics['fresh_outcomes_read']
    assert registry['reserved_seed_base'] == 2026111529 and registry['existing_protocol_references'] == 0
    for name, digest in gate['source_sha256'].items():
        assert S.sha(S.HERE / name) == digest, name
    for name, digest in statistics['source_sha256'].items():
        assert S.sha(S.HERE / name) == digest, name
    for item in registry['protocol_registry'] + registry['prior_data_manifests']:
        assert S.sha(S.ROOT / item['path']) == item['sha256']
    assert S.sha(S.HERE / 'audit_bp_first_fresh_seed_registry.py') == registry['source_sha256']
    p.update(status='frozen_before_source_gates_and_fresh_draws', frozen_utc=S.utc(),
             models=source['models'], arms=source['arms'], source_jobs=parent['source_jobs'],
             source_datasets=parent['datasets'], jobs=source['jobs'],
             budgets=p['proposed_budgets'], paired_schedule=copy.deepcopy(parent['paired_schedule']),
             cohort={'base_seed': 2026111529, 'n': 4096, 'size': 180, 'loop_probability': .8,
                     'prior_registry': registry['prior_data_manifests']},
             primary_accuracy_pair=0, planned_fresh_jobs=48, primary_fresh_evaluations=36864,
             parent_protocol_sha256=S.sha(PARENT / 'protocol.json'),
             freeze_reads_source_outcome_arrays=False)
    p['paired_schedule']['cohort_per_arm'] = 'All4096 fresh inputs and allthree fixed models in each arm; onlypair0 determines scientific accuracy.'
    p['source_sha256'] = {**source['source_sha256'], **parent['source_sha256'],
        **{str((S.HERE / name).relative_to(S.ROOT)): S.sha(S.HERE / name) for name in modules()}}
    evidence = [proposal, DEST / 'synthetic-gate.json', DEST / 'statistics-gate.json', DEST / 'seed-registry.json',
                PARENT / 'protocol.json', PARENT / 'audit.json', PARENT / 'sealed-replays.json',
                PARENT / 'synthetic-gate.json', PARENT / 'gate-gpu0.json', PARENT / 'gate-gpu1.json',
                DEST / 'source-format-compatibility.json', DEST / 'source-format-provenance.json',
                DEST / 'candidate-before-wall-unit-correction.json', DEST / 'source_format_preflight.py',
                DEST / 'source-format-compatibility-attempt1.json']
    for seed in range(3):
        for arm in ['pl', 'old_binary']:
            original = SOURCE / f'model-seed{seed}-{arm}.npz'
            copied = DEST / original.name
            if not copied.exists():
                with copied.open('xb') as stream:
                    stream.write(original.read_bytes())
            assert S.sha(copied) == S.sha(original)
            evidence.append(copied)
    p['source_evidence_sha256'] = {**parent['source_evidence_sha256'],
        **{str(path.relative_to(S.ROOT)): S.sha(path) for path in evidence}}
    if preview:
        p.update(status='candidate_awaiting_parent_final_review', frozen_utc=None)
        S.save(DEST / 'candidate-protocol.json', p)
        print(json.dumps({'status': p['status'], 'candidate_sha256': S.sha(DEST / 'candidate-protocol.json'),
                          'source_files': len(p['source_sha256']), 'fresh_inputs_drawn': 0}, indent=2))
        return
    S.save(PP, p)
    S.save(DEST / 'protocol.json', p)
    print(json.dumps({'status': p['status'], 'protocol_sha256': S.sha(PP),
                      'source_files': len(p['source_sha256']), 'fresh_inputs_drawn': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    freeze(parser.parse_args().preview)
