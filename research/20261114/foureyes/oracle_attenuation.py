"""H4as: privileged corruption-mask attenuation at a fixed inferred degree."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
WALL_ORIGIN = time.monotonic()
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ripper'))
PARENT = ROOT / 'results/research/foureyes/prefix_budget/degree'
OUT = ROOT / 'artifacts/research/20261114/foureyes/oracle_attenuation'
PUBLIC = ROOT / 'results/research/foureyes/oracle_attenuation'
PROTOCOL = HERE / 'oracle-attenuation.json'
PYTHON = ROOT / 'artifacts/degree-lp-venv/bin/python'
SOLVER_SEED_BASE = 2029021502
FAMILIES = ['cycle81', 'power2', 'power3', 'cliques9']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(protocol, public=False, sources_only=False):
    unavailable = []
    hashes = protocol['source_sha256'] if sources_only else protocol['source_sha256'] | protocol['input_sha256']
    for rel, digest in hashes.items():
        path = ROOT / rel
        if public and rel.startswith('artifacts/') and not path.exists():
            unavailable.append(rel)
        else:
            assert sha(path) == digest, rel
    return unavailable


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate_path = HERE / 'oracle-attenuation-preflight.json'
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['study_masks_or_scores_read'] is False
    for rel, digest in gate['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    seed_path = HERE / 'oracle-attenuation-seed-audit.json'
    seed_audit = read(seed_path)
    assert seed_audit['status'] == 'passed' and sha(seed_path) == gate['seed_audit_sha256']
    for rel, digest in seed_audit['frozen_protocol_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    prior = read(PARENT / 'summary.json')
    parent_jobs = read(HERE/'prefix_budget.json')['jobs']
    jobs = []
    inputs = [gate_path, seed_path, HERE / 'oracle-attenuation-proposal.json',
              PARENT / 'summary.json', HERE / 'prefix_budget.json']
    for group in range(64):
        for fi, family in enumerate(FAMILIES):
            index = group*4 + fi
            pi = 12*group + 3*fi + 1
            old = prior['rows'][pi]
            assert old['family'] == family and old['rows'] == 600 and old['classification'] != 'unresolved'
            jobs.append(dict(index=index, group=group, family=family, parent_prefix_index=pi,
                             inferred_degree=old['selected_degree'], baseline_exact=old['classification']=='success',
                             archive=parent_jobs[pi]))
            source = ROOT/parent_jobs[pi]['source_path']
            assert sha(source) == parent_jobs[pi]['source_sha256']
            inputs.append(source)
            inputs += [PARENT / f'j{pi:03d}' / name for name in ['models.npz', 'receipt.json']]
    assert len(jobs) == 256 and sum(j['baseline_exact'] for j in jobs) == 129
    inputs += [ROOT / f'results/research/foureyes/{name}/controls/graphs.npz'
               for name in ['cycle_power', 'varying_degree']]
    inputs += [HERE/name for name in ['varying_degree.json', 'cycle_power.json']]
    parent = read(HERE / 'bootstrap-consensus.json')
    proposal = read(HERE / 'oracle-attenuation-proposal.json')
    protocol = dict(proposal, status='frozen_before_study_mask_reads_attenuation_scores_and_new_LPs',
                    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=jobs,
                    source_sha256=gate['source_sha256'],
                    input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    solver=dict(parent['solver'], random_seed_base=SOLVER_SEED_BASE),
                    solver_seed_rule='random_seed_base + bank index, 0..255')
    save(PROTOCOL, protocol)
    print('FROZEN', sha(PROTOCOL), flush=True)


def worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from audit_oracle_attenuation import audit_bank
    protocol = read(PROTOCOL)
    verify(protocol, sources_only=True)
    from oracle_attenuation_math import mask_from_indices, score
    ii, jj = np.triu_indices(81, 1)
    for index in range(worker_id, 256, 8):
        job = protocol['jobs'][index]
        directory = OUT / f'b{index:03d}'
        directory.mkdir()
        with np.load(PARENT / f"j{job['parent_prefix_index']:03d}/models.npz", allow_pickle=False) as z:
            counts = z['cost']
        archive = job['archive']
        assert sha(ROOT/archive['source_path']) == archive['source_sha256']
        with np.load(ROOT/archive['source_path'], allow_pickle=False) as z:
            cells = z['cells'][:600]
        mask = mask_from_indices(cells)
        result = score(counts, mask)
        log = directory / 'solver.log'
        if len(result['nonpositive_pairs']):
            log.write_text('LP skipped: nonpositive attenuation; whole bank unresolved.\n')
            metadata = dict(skipped=True, reason='nonpositive_attenuation')
            x, y = np.empty(0, np.float64), np.empty(0, np.float64)
            certificate = dict(certified=False, reason='nonpositive_attenuation')
            proof = {key: np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
        else:
            x, y, metadata = solve(81, job['inferred_degree'], ii, jj, result['cost'],
                                   SOLVER_SEED_BASE-2026111502+index, log)
            if metadata['solver_optimal'] and metadata['value_valid'] and metadata['dual_valid']:
                certificate, proof = attempt_certificate(81, job['inferred_degree'], ii, jj, result['cost'], x, y)
            else:
                certificate = dict(certified=False, reason='solver_status_or_validity')
                proof = {key: np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
        np.savez_compressed(directory/'models.npz', counts=counts, cells=cells, mask=mask, **result,
                            primal_values=x, node_dual_values=y, **proof)
        save(directory/'receipt.json', dict(index=index, degree=job['inferred_degree'],
             protocol_sha256=sha(PROTOCOL), models_sha256=sha(directory/'models.npz'),
             log_sha256=sha(log), solver=metadata, certificate=certificate))
        save(directory/'progress.json', dict(index=index, completed_LPs=int(not len(result['nonpositive_pairs'])), processed_banks=1, pid=os.getpid()))
        audited = audit_bank(index, protocol, public=False)
        save(directory/'audit.json', audited)
        print('BANK_AUDITED', index, flush=True)


def export(protocol):
    import shutil
    from audit_oracle_attenuation import aggregate
    PUBLIC.mkdir()
    rows = []
    files = {}
    for index in range(256):
        directory = OUT / f'b{index:03d}'
        audit = read(directory/'audit.json')
        assert audit['protocol_sha256'] == sha(PROTOCOL)
        assert audit['auditor_sha256'] == sha(HERE/'audit_oracle_attenuation.py')
        for name, digest in audit['artifact_sha256'].items():
            assert sha(directory/name) == digest
        rows.append(audit['row'])
        target = PUBLIC / directory.name
        target.mkdir()
        for name in ['models.npz', 'receipt.json', 'audit.json']:
            shutil.copyfile(directory/name, target/name)
            files[str((target/name).relative_to(PUBLIC))] = sha(target/name)
    save(PUBLIC/'summary.json', dict(status='validated', protocol_sha256=sha(PROTOCOL),
         rows=rows, **aggregate(rows), artifact_sha256=files, new_labels=0,
         new_source_draws=0, new_LPs=sum(r['LP_attempted'] for r in rows), privileged_location_indices=256*600*9, new_GPU_seconds=0,
         scope=protocol['scope'], counts_provenance='Inherited complete H4an count audit; all archived counts are hash-bound and all mask streams regenerated independently and every exact attenuation score independently recomputed. No clean digits or actual offsets enter the scorer.'))


def launch():
    protocol = read(PROTOCOL)
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    children = []
    logs = []
    samples = []
    status, error = 'failed', None
    ticks = os.sysconf('SC_CLK_TCK')
    def cpu():
        live = 0.
        for child in children:
            if child.poll() is None:
                try:
                    fields = Path(f'/proc/{child.pid}/stat').read_text().split()
                    live += (int(fields[13])+int(fields[14]))/ticks
                except FileNotFoundError:
                    pass
        return live + sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),
                                                          resource.getrusage(resource.RUSAGE_CHILDREN)])
    def monitor(phase):
        sample = dict(phase=phase, wall_seconds=time.monotonic()-WALL_ORIGIN,
                      aggregate_cpu_seconds=cpu(), completed_LPs=sum(read(q)['completed_LPs'] for q in OUT.glob('b*/progress.json')),
                      processed_banks=len(list(OUT.glob('b*/progress.json'))),
                      audited_banks=len(list(OUT.glob('b*/audit.json'))),
                      active_pids=[c.pid for c in children if c.poll() is None])
        samples.append(sample)
        save(OUT/'progress.json', sample)
        assert sample['wall_seconds'] < 90 and sample['aggregate_cpu_seconds'] < 300, 'frozen_resource_ceiling'
    try:
        verify(protocol)
        monitor('input_hashes')
        for worker_id in range(8):
            stream = (OUT/f'worker{worker_id}.log').open('x')
            logs.append(stream)
            children.append(subprocess.Popen([str(PYTHON), str(Path(__file__)), 'worker', '--worker', str(worker_id)],
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor('LP_and_audit')
            assert all(c.poll() in [None, 0] for c in children), 'worker_failure'
            time.sleep(.2)
        assert all(c.returncode == 0 for c in children)
        monitor('LP_and_audit')
        export(protocol)
        monitor('export')
        status = 'complete'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in logs:
            stream.close()
        result = dict(status=status, error=error, protocol_sha256=sha(PROTOCOL),
                      wall_seconds=time.monotonic()-WALL_ORIGIN, aggregate_cpu_seconds=cpu(),
                      returncodes=[c.returncode for c in children], samples=samples)
        save(OUT/'runtime.json', result)
        if PUBLIC.exists():
            save(PUBLIC/'runtime.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'launch', 'worker'])
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    worker(args.worker) if args.action == 'worker' else freeze() if args.action == 'freeze' else launch()
