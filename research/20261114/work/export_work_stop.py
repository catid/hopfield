"""Publish preserved H2bt gates and unrun status; never launch an experiment."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/research/20261114/work'
RESULTS = ROOT / 'results/research/work'
DEST = OUT / 'bp_first_packed_controls'
PUBLIC = RESULTS / 'bp-first-packed-controls-unrun'
DESIGN = RESULTS / 'bp-first-packed-confirmation-unrun'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not PUBLIC.exists() and not DESIGN.exists()
    protocol = read(DEST / 'protocol.json')
    status = read(DEST / 'unrun-status.json')
    assert status['status'] == 'frozen_but_timing_unrun_at_user_stop'
    assert status['actual_timed_model_pipeline_copies'] == 0
    assert status['protocol_sha256'] == sha(DEST / 'protocol.json')
    assert not any(status['timing_artifacts_present'].values())
    assert not any((DEST / name).exists() for name in status['timing_artifacts_present'])
    prior = read(OUT / 'bp_first_packed_dispatch/publication-allowlist.json')
    shared = {'research/20261114/work/REPORT.md', 'results/research/work/summary.json',
              'results/research/work/manifest.json'}
    for name, digest in prior['sha256'].items():
        assert sha(ROOT / name) == digest, name
    verified_bindings = 0
    for key in ['source_sha256', 'source_evidence_sha256']:
        for name, digest in protocol[key].items():
            assert sha(ROOT / name) == digest, name
            verified_bindings += 1

    PUBLIC.mkdir()
    copied = {}

    def copy(source, relative):
        target = PUBLIC / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        copied[str(source.relative_to(ROOT))] = {
            'public_file': str(target.relative_to(RESULTS)), 'sha256': sha(target)}

    for path in sorted(DEST.glob('*.json')):
        copy(path, path.name)
    gate_copies = 0
    archive_references = {}
    for gpu in [0, 1]:
        gate = read(DEST / f'gate-gpu{gpu}.json')
        assert gate['status'] == 'passed' and gate['source_model_inputs_per_arm'] == 384
        assert len(gate['checks']) == 3
        for mode in ['LP96', 'BP_geometry_LP96', 'packed_dispatch']:
            folder = DEST / 'source-gates' / f'gpu{gpu}-{mode}'
            cohort = read(folder / 'result.json')
            check = next(row for row in gate['checks'] if row['mode'] == mode)
            assert check['status'] == 'passed' and check['model_pipeline_copies'] == 384
            assert check['cohort_receipt_sha256'] == sha(folder / 'result.json')
            copy(folder / 'result.json', str((folder / 'result.json').relative_to(DEST)))
            gate_copies += check['model_pipeline_copies']
            for row in cohort['rows']:
                if row['zero_work']:
                    continue
                archive_references[folder / row['file']] = row['file_sha256']
                for name, digest in row['trace_sha256'].items():
                    archive_references[folder / name] = digest
            for model in cohort['models']:
                for key in ['global_answer_file', 'queue_file']:
                    archive_references[folder / model[key]] = model[key + '_sha256']
    raw = {}
    for path in sorted((DEST / 'source-gates').rglob('*.npz')):
        digest = sha(path)
        assert archive_references.pop(path) == digest, str(path)
        raw[str(path.relative_to(DEST))] = {'sha256': digest, 'bytes': path.stat().st_size}
    assert not archive_references and gate_copies == 2304
    save(PUBLIC / 'retained-archive-manifest.json', {
        'status': 'all_original_gate_archive_hashes_verified',
        'retained_directory': str(DEST.relative_to(ROOT)),
        'retained_hosts': ['local workspace', 'work:/home/catid/hopfield'],
        'files': raw, 'files_count': len(raw), 'bytes': sum(r['bytes'] for r in raw.values()),
        'public_raw_archive_copies': 0,
        'scope': 'Original source-gate raw states and trajectories retained locally and on Work. '
                 'This concise public package is a receipt/hash record, not a portable numerical replay.'})
    markers = {str(path.relative_to(DEST)): path.read_text()
               for path in sorted((DEST / 'barriers').rglob('*.json'))}
    assert len(markers) == 540
    save(PUBLIC / 'barrier-markers.json', {'encoding': 'Original UTF-8 text, byte-exact', 'markers': markers})
    independent = ROOT / 'artifacts/research/20261114/loki/h2bt-independent-source-review.json'
    assert sha(independent) == '47757fb587d174dc01f49ca78a2b7d6381be17974b6bdc1fbf69f3548b1793fd'
    copy(independent, independent.name)
    save(PUBLIC / 'publication-evidence-map.json', {
        'copied_original_files': copied,
        'actual_source_gate_model_pipeline_copies': gate_copies,
        'actual_timed_model_pipeline_copies': 0,
        'fresh_inputs': 0, 'new_scientific_jobs': 0, 'new_scientific_evaluations': 0,
        'scope': 'Passed fixed-source gates only. Complete-cohort timing, full final audit, '
                 'and both cost conclusions are absent. Gate-local clocks are diagnostic evidence only.'})
    (PUBLIC / 'README.md').write_text(
        '# H2bt: frozen, timing unrun\n\n'
        'The user stopped this research round after the source gates and final protocol freeze. '
        '**No complete-cohort timing was launched, and neither 5% cost target has a result.**\n\n'
        'The [status receipt](unrun-status.json) binds the passed GPU gates and frozen protocol. '
        'Each GPU checked 384 model-inputs per actual arm, across LP96, the always-BP geometry '
        'cascade, and the packed dispatcher: 2,304 source-gate copies in total, zero fresh '
        'scientific observations, and zero timed copies. Gate-local costs are not endpoint results.\n\n'
        'The [evidence map](publication-evidence-map.json) connects unchanged receipts to their '
        'original locations. The [archive manifest](retained-archive-manifest.json) hashes all '
        'raw source-gate states and trajectories retained locally and on Work. Raw gate arrays '
        'are not included here; this package does not claim a portable numerical replay. '
        'The preceding H2bs measured result retains its separate complete public evidence.\n')

    DESIGN.mkdir()
    design_receipt = OUT / 'bp_first_packed_confirmation/initial-design.json'
    shutil.copyfile(design_receipt, DESIGN / design_receipt.name)
    design = {'hypothesis': 'H2bu', 'status': 'design_only_unrun_at_user_stop',
              'proposal': 'research/20261114/work/bp-first-packed-confirmation-proposal.json',
              'proposal_sha256': sha(HERE / 'bp-first-packed-confirmation-proposal.json'),
              'initial_design_sha256': sha(design_receipt), 'seed_reserved': False,
              'fresh_inputs_drawn': 0, 'changed_implementation': False,
              'new_scientific_jobs': 0, 'new_scientific_evaluations': 0,
              'scope': 'Unreviewed proposed fresh confirmation only; no protocol freeze, '
                       'seed reservation, dataset draw, fit, inference or endpoint result.'}
    save(DESIGN / 'status.json', design)
    report = HERE / 'REPORT.md'
    old = report.read_text()
    assert '### H2bt:' not in old and '### H2bu:' not in old
    report.write_text(old + '''

### H2bt: source gates passed; frozen timing experiment never launched

H2bt has **no complete-cohort timing or cost result**. Its [proposal](bp-first-packed-controls-proposal.json) would test separate ≥5% savings against newly executed LP96 and always-BP geometry cascade controls on the examined H2br cohort. The [protocol](bp-first-packed-controls-protocol.json) froze at **2026-11-15 17:10:40 UTC**, after the source gates passed. The user then stopped this research round before any timing warmup or timing worker launched. Neither cost target is supported or falsified.

The [unchanged gate receipts and stop evidence](../../../results/research/work/bp-first-packed-controls-unrun/unrun-status.json) preserve **384 model-inputs per actual arm per GPU**, three arms and two GPUs: **2,304 source-gate copies**, **zero timed copies**, and **zero new scientific observations**. The two source-gate workers finished in **24.673268 and 24.672912 seconds** within their 240-second caps, with **25 live process/device samples**, both exit codes zero, no forced signals and no independent CPU-audit overlap. Their local diagnostic clocks are not complete-cohort endpoint measurements. At the stop check, both Work GPUs were idle and no compute process remained.

The [concise public evidence package](../../../results/research/work/bp-first-packed-controls-unrun/README.md) includes the frozen packet, synthetic gates, all six actual cohort receipts, worker/live evidence and all 540 gate barrier markers. Its [retained-archive manifest](../../../results/research/work/bp-first-packed-controls-unrun/retained-archive-manifest.json) records the complete raw gate states/trajectories preserved locally and on Work. Those raw arrays are not included in this concise package; it does not claim a portable numerical replay. The six timing blocks, final seal, full final audit and both cost joins remain unrun. **H2bs is the last measured Work experiment**, and its 31.336402% gain remains limited to the original-order dispatcher control.

### H2bu: proposed fresh confirmation, design only

The [H2bu proposal](bp-first-packed-confirmation-proposal.json) was recorded before any H2bt timing result. It proposes a fresh 4,096-maze IID endpoint mixture with BP AdamW first, LP96 and always-cascade controls, fixed absolute 99.9% joint-success and −0.1 percentage-point noninferiority targets, and separate ≥5% cost targets. This is an **unreviewed, unrun design**, not a result: [no seed was reserved, no fresh input was drawn, and no changed implementation or inference was started](../../../results/research/work/bp-first-packed-confirmation-unrun/status.json).

Research is stopped at the user's request. H2bt gates and the H2bu design add **zero scientific jobs, zero run-arm observation rows and zero primary evaluations**. Final Work totals remain **2,314 planned jobs / 1,765,118 primary evaluations**, or **2,320 included jobs / 1,765,250 evaluations**, with **7,647 run-arm rows**.
''')
    summary = read(RESULTS / 'summary.json')
    assert 'bp_first_packed_controls' not in summary
    totals = {k: summary[k] for k in ['jobs', 'run_arm_rows', 'model_maze_evaluations',
                                    'primary_jobs', 'primary_model_maze_evaluations']}
    summary['bp_first_packed_controls'] = {
        'hypothesis': 'H2bt', 'status': status['status'], 'source_gate_copies': 2304,
        'actual_timed_model_pipeline_copies': 0, 'fresh_inputs': 0,
        'new_scientific_jobs': 0, 'new_scientific_evaluations': 0,
        'cost_endpoint_status': 'not_run', 'cost_decisions': None,
        'status_evidence': 'bp-first-packed-controls-unrun/unrun-status.json',
        'scope': 'Frozen protocol and passed source gates only; no complete-cohort cost result.'}
    summary['bp_first_packed_confirmation'] = design
    summary['protocols']['bp-first-packed-controls-protocol'] = {
        'sha256': sha(DEST / 'protocol.json'), 'frozen_utc': protocol['frozen_utc'],
        'hypothesis': 'H2bt', 'status': 'frozen_but_timing_unrun_at_user_stop'}
    summary['status'] = 'Stopped at user request; H2bs last measured result; H2bt timing unrun; H2bu design only'
    summary['updated_utc'] = '2026-11-15T17:11:48+00:00'
    assert totals == {k: summary[k] for k in totals}
    save(RESULTS / 'summary.json', summary)

    links = []
    for path in [report, PUBLIC / 'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            resolved = (path.parent / target.split('#')[0]).resolve()
            assert resolved.exists(), (str(path), target)
            links.append((str(path.relative_to(ROOT)), target))
    for name, digest in prior['sha256'].items():
        if name not in shared:
            assert sha(ROOT / name) == digest, name
    for key in ['source_sha256', 'source_evidence_sha256']:
        for name, digest in protocol[key].items():
            assert sha(ROOT / name) == digest, name
    save(PUBLIC / 'publication-verification.json', {
        'status': 'passed', 'kind': 'publication_hash_and_link_check_only',
        'H2bt_frozen_bindings_verified': verified_bindings,
        'raw_gate_archives_verified': len(raw), 'source_gate_copies': 2304,
        'gate_barrier_markers': len(markers), 'report_and_package_links_verified': len(links),
        'H2bs_prior_package_bytes_unchanged': True, 'scientific_totals_unchanged': totals,
        'actual_timed_H2bt_copies': 0, 'new_experiment_jobs_launched': 0,
        'publisher_sha256': sha(Path(__file__))})
    save(RESULTS / 'manifest.json', {
        'source_sha256': sha(Path(__file__)),
        'files': {str(path.relative_to(RESULTS)): sha(path)
                  for path in sorted(RESULTS.rglob('*'))
                  if path.is_file() and path != RESULTS / 'manifest.json'}})
    paths = set(prior['paths'])
    parent = read(OUT / 'bp_first_packed_dispatch/protocol.json')
    paths.update(set(protocol['source_sha256']) - set(parent['source_sha256']))
    paths.update(str(path.relative_to(ROOT)) for directory in [PUBLIC, DESIGN]
                 for path in directory.rglob('*') if path.is_file())
    paths.update('research/20261114/work/' + name for name in [
        'bp-first-packed-controls-proposal.json', 'bp-first-packed-controls-protocol.json',
        'bp-first-packed-confirmation-proposal.json', 'export_work_stop.py'])
    manifest = read(RESULTS / 'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(RESULTS / name) == digest, name
    allowlist = {
        'hypotheses': ['H2bs measured', 'H2bt frozen timing unrun', 'H2bu design only'],
        'status': 'stable_for_parent_final_publication',
        'supersedes_H2bs_allowlist_sha256': sha(OUT / 'bp_first_packed_dispatch/publication-allowlist.json'),
        'paths': sorted(paths), 'sha256': {name: sha(ROOT / name) for name in sorted(paths)},
        'global_manifest_hashes_verified': len(manifest['files']),
        'H2bt_frozen_bindings_verified': verified_bindings,
        'report_and_package_links_verified': len(links),
        'scope': 'Audited H2bs package unchanged; H2bt source, passed gate receipts and frozen-but-unrun '
                 'evidence; H2bu unrun proposal; shared Work report/summary/manifest. Raw H2bt gates '
                 'retained and hash-listed, not publicly copied. No timing result or new experiment.'}
    target = DEST / 'final-publication-allowlist.json'
    assert not target.exists()
    save(target, allowlist)
    print(json.dumps({'allowlist': str(target.relative_to(ROOT)), 'sha256': sha(target),
                      'paths': len(paths), 'new_public_files': sum(p.is_file() for d in [PUBLIC, DESIGN] for p in d.rglob('*')),
                      'global_manifest_hashes': len(manifest['files']), 'links': len(links),
                      'raw_gate_archives_verified': len(raw), 'raw_gate_bytes': sum(r['bytes'] for r in raw.values())}, indent=2))


if __name__ == '__main__':
    main()
