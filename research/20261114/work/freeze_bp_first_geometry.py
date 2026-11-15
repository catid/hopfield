"""H2bn freeze helper; run only after parent source/design review."""
import json
import study as S

DEST = S.OUT / 'bp_first_geometry_cascade'
SOURCE = S.OUT / 'loop80_confirmation'
PP = S.HERE / 'bp-first-geometry-cascade-protocol.json'


def freeze():
    assert not PP.exists() and not (DEST / 'protocol.json').exists()
    proposal = S.HERE / 'bp-first-geometry-cascade-proposal.json'
    p = json.loads(proposal.read_text())
    source = json.loads((SOURCE / 'protocol.json').read_text())
    gate_path = DEST / 'synthetic-gate.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'passed' and gate['synthetic_only']
    for key in ['both_archive_tasks_drained_before_answer', 'incomplete_worker_receipt_preserved',
                'existing_attempt_refused', 'owned_watchdog_signal_receipt', 'subset_trace_index_normalization']:
        assert gate[key]
    for key in ['host_barrier_waits_for_both_owned_workers', 'host_barrier_marker_overwrite_refused',
                'host_barrier_peer_failure_timeout_death_abort']:
        assert gate[key]
    for name, digest in gate['source_sha256'].items():
        assert S.sha(S.HERE / name) == digest, name
    jobs = []
    for job in source['jobs']:
        path = SOURCE / 'runs' / job['id']
        receipt = json.loads((path / 'result.json').read_text())
        assert receipt['status'] == 'complete'
        assert receipt['protocol_sha256'] == S.sha(SOURCE / 'protocol.json')
        assert S.sha(path / 'outputs.npz') == receipt['outputs_sha256']
        hashes = {f'batch{bi:02d}.npz': S.sha(path / f'batch{bi:02d}.npz') for bi in range(8)}
        for trace in receipt['trace_archives']:
            assert S.sha(path / trace['path']) == trace['sha256']
            hashes[trace['path']] = trace['sha256']
        assert len(hashes) == 56
        jobs.append({**job, 'outputs_sha256': receipt['outputs_sha256'],
                     'receipt_sha256': S.sha(path / 'result.json'), 'batch_sha256': hashes})
    names = ['bp_geometry_checks.py', 'bp_first_geometry_policy.py', 'bp_first_geometry_accounting.py',
             'bp_first_lp96_core.py', 'bp_first_geometry_batch.py', 'bp_first_geometry_parity.py',
             'bp_first_geometry_pipeline.py', 'bp_first_geometry_study.py', 'audit_bp_first_geometry.py',
             'bp_first_geometry_barrier.py',
             'preflight_bp_first_geometry.py', 'freeze_bp_first_geometry.py', 'seal_bp_first_geometry.py',
             'bounded_archive_writer.py', 'audit_connectivity_screen.py', 'verify_hard_negative_feasibility.py',
             'verify_training_feasibility.py']
    p.update(status='frozen_before_complete_source_gate_and_paired_replays', frozen_utc=S.utc(),
             source_jobs=jobs, models=source['models'],
             datasets=json.loads((SOURCE / 'datasets.json').read_text())['datasets'],
             freeze_reads_source_outcome_arrays=False)
    p['source_sha256'] = {**source['source_sha256'],
                         **{str((S.HERE / n).relative_to(S.ROOT)): S.sha(S.HERE / n) for n in names}}
    evidence = [proposal, gate_path, SOURCE / 'protocol.json', SOURCE / 'audit.json',
                SOURCE / 'datasets.json', SOURCE / 'model-extraction.json',
                S.ROOT / 'results/research/work/confirmation96/timing-accounting.json',
                S.ROOT / 'results/research/work/bp-geometry-decomposition/analysis.json',
                S.ROOT / 'results/research/work/bp-geometry-decomposition/protocol.json']
    evidence += [SOURCE / f'model-seed{seed}-{arm}.npz' for seed in range(3) for arm in ['pl', 'old_binary']]
    p['source_evidence_sha256'] = {str(path.relative_to(S.ROOT)): S.sha(path) for path in evidence}
    S.save(PP, p)
    S.save(DEST / 'protocol.json', p)
    print(json.dumps({'status': p['status'], 'frozen_utc': p['frozen_utc'],
                      'protocol_sha256': S.sha(PP), 'source_jobs': len(jobs)}, indent=2))


if __name__ == '__main__':
    freeze()
