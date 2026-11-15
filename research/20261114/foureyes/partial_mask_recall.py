"""H4au: incomplete, perfectly precise corruption masks on all H4at banks."""
from pathlib import Path
import argparse, datetime, json, os, resource, shutil, signal, subprocess, time
WALL_ORIGIN = time.monotonic()
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from corruption_censor import ROOT, HERE, PYTHON, FAMILIES, sha, read, save
PARENT = ROOT/'results/research/foureyes/corruption_censor'
PARENT_PROTOCOL = HERE/'corruption-censor.json'
OUT = ROOT/'artifacts/research/20261114/foureyes/partial_mask_recall'
PUBLIC = ROOT/'results/research/foureyes/partial_mask_recall'
PROTOCOL = HERE/'partial-mask-recall.json'
RECALLS = [50, 75, 90, 95, 99]


def verify(p, sources_only=False):
    for rel, digest in (p['source_sha256'] if sources_only else p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate = read(HERE/'partial-mask-recall-preflight.json')
    assert gate['status'] == 'passed' and not gate['study_partial_masks_or_scores_read']
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    seed_path = HERE/'partial-mask-recall-seed-audit.json'
    seeds = read(seed_path)
    assert seeds['status'] == 'passed' and sha(seed_path) == gate['seed_audit_sha256']
    for rel, digest in seeds['frozen_protocol_sha256'].items():
        assert sha(ROOT/rel) == digest
    parent = read(PARENT_PROTOCOL)
    design = read(HERE/'partial-mask-recall-proposal.json')
    assert design['recall_percent'] == RECALLS
    # Bind every public dependency of the complete H4at audit, excluding unused raw archives.
    inputs = {rel:digest for rel, digest in parent['input_sha256'].items() if not rel.startswith('artifacts/')}
    for path in [PARENT_PROTOCOL, HERE/'partial-mask-recall-proposal.json', HERE/'partial-mask-recall-preflight.json', seed_path,
                 *sorted(PARENT.rglob('*.npz')), *sorted(PARENT.rglob('*.json'))]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    p = dict(design, status='frozen_before_study_partial_mask_draws_scores_and_LPs',
             frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=parent['jobs'],
             source_sha256=gate['source_sha256'], input_sha256=inputs,
             solver=dict(parent['solver'], random_seed_base=design['solver_seed_base']))
    verify(p)
    save(PROTOCOL, p)
    print('FROZEN', sha(PROTOCOL), flush=True)


def worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from partial_mask_math import draws, revealed_mask, score
    from audit_partial_mask_recall import audit_bank
    p = read(PROTOCOL)
    verify(p, sources_only=True)
    ii, jj = np.triu_indices(81, 1)
    for index in range(worker_id, 256, 8):
        folder = OUT/f'b{index:03d}'
        folder.mkdir()
        with np.load(PARENT/f'b{index:03d}/models.npz', allow_pickle=False) as z:
            cells, noisy = z['cells'], z['noisy']
        raw = draws(p['thinning']['seed_base'], index)
        np.save(folder/'draws.npy', raw, allow_pickle=False)
        degree = p['jobs'][index]['inferred_degree']
        for arm, percent in enumerate(RECALLS):
            mask = revealed_mask(cells, raw, percent)
            counts = score(noisy, mask)
            log = folder/f'p{percent}.log'
            x, y, metadata = solve(81, degree, ii, jj, counts['cost'], p['solver_seed_base']-2026111502+5*index+arm, log)
            if metadata['solver_optimal'] and metadata['value_valid'] and metadata['dual_valid']:
                certificate, proof = attempt_certificate(81, degree, ii, jj, counts['cost'], x, y)
            else:
                certificate = dict(certified=False, reason='solver_status_or_validity')
                proof = {key:np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
            np.savez_compressed(folder/f'p{percent}.npz', mask=mask, **counts, primal_values=x, node_dual_values=y, **proof)
            save(folder/f'p{percent}.json', dict(index=index, recall_percent=percent, degree=degree,
                 protocol_sha256=sha(PROTOCOL), models_sha256=sha(folder/f'p{percent}.npz'),
                 log_sha256=sha(log), solver=metadata, certificate=certificate))
            save(folder/'progress.json', dict(completed_LPs=arm+1, pid=os.getpid()))
        save(folder/'audit.json', audit_bank(index, p))
        print('BANK_AUDITED', index, flush=True)


def export(p):
    from audit_partial_mask_recall import aggregate
    PUBLIC.mkdir()
    rows, files = [], {}
    for index in range(256):
        source = OUT/f'b{index:03d}'
        audit = read(source/'audit.json')
        assert audit['protocol_sha256'] == sha(PROTOCOL) and audit['auditor_sha256'] == sha(HERE/'audit_partial_mask_recall.py')
        for name, digest in audit['artifact_sha256'].items():
            assert sha(source/name) == digest
        rows.extend(audit['rows'])
        dest = PUBLIC/source.name
        dest.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json']:
            shutil.copyfile(source/name, dest/name)
            files[str((dest/name).relative_to(PUBLIC))] = sha(dest/name)
    save(PUBLIC/'summary.json', dict(status='validated', protocol_sha256=sha(PROTOCOL), rows=rows,
         **aggregate(rows), artifact_sha256=files, new_LPs=1280, new_labels=0, new_source_draws=0,
         inherited_certificates_checked=512, inherited_mutations_rejected=1536,
         mask_supervision='Perfect precision; five nested nominal recall arms. No detector trained or evaluated.',
         observation_provenance='Public H4at observations/masks and inherited complete archive audit. Original generation not rerun.'))


def launch():
    p = read(PROTOCOL)
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    children, logs, samples = [], [], []
    status, error = 'incomplete', None
    ticks = os.sysconf('SC_CLK_TCK')
    def cpu():
        total = sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)])
        for c in children:
            if c.poll() is None:
                try:
                    f = Path(f'/proc/{c.pid}/stat').read_text().split()
                    total += (int(f[13])+int(f[14]))/ticks
                except FileNotFoundError:
                    pass
        return total
    def monitor(phase):
        active = [c.pid for c in children if c.poll() is None]
        sample = dict(phase=phase, wall_seconds=time.monotonic()-WALL_ORIGIN, aggregate_cpu_seconds=cpu(),
                      active_pids=active, completed_LPs=sum(read(q)['completed_LPs'] for q in OUT.glob('b*/progress.json')),
                      audited_banks=len(list(OUT.glob('b*/audit.json'))))
        samples.append(sample)
        save(OUT/'progress.json', sample)
        assert sample['wall_seconds'] < p['budget']['wall_seconds'] and sample['aggregate_cpu_seconds'] < p['budget']['aggregate_cpu_seconds'], 'frozen_resource_ceiling'
    try:
        verify(p)
        monitor('hash_verification')
        for i in range(8):
            stream = (OUT/f'worker{i}.log').open('x')
            logs.append(stream)
            children.append(subprocess.Popen([str(PYTHON), str(Path(__file__)), 'worker', '--worker', str(i)],
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor('scores_LPs_and_audits')
            assert all(c.poll() in [None, 0] for c in children), 'worker_failure'
            time.sleep(.2)
        assert all(c.returncode == 0 for c in children)
        monitor('scores_LPs_and_audits')
        export(p)
        monitor('export')
        status = 'complete'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid, signal.SIGTERM)
                try:
                    c.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(c.pid, signal.SIGKILL)
                    c.wait()
        for stream in logs:
            stream.close()
        runtime = dict(status=status, error=error, protocol_sha256=sha(PROTOCOL), wall_seconds=time.monotonic()-WALL_ORIGIN,
                       aggregate_cpu_seconds=cpu(), returncodes=[c.returncode for c in children], samples=samples)
        save(OUT/'runtime.json', runtime)
        if PUBLIC.exists():
            save(PUBLIC/'runtime.json', runtime)
        print(json.dumps({k:v for k,v in runtime.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'worker', 'launch'])
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    worker(args.worker) if args.action == 'worker' else globals()[args.action]()
