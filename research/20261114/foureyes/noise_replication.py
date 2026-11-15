"""Independent corruption replicates; estimators receive counts, never graph truth."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/research/20261114/foureyes/noise_replication'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def claim(gpu, job_id=None, status=None):
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            item = next((j for j in jobs if j['status'] == 'pending'), None)
            if item is not None:
                item.update(status='running', gpu=gpu, pid=os.getpid(), started=time.time())
        else:
            item = next(j for j in jobs if j['seed'] == job_id)
            item.update(status=status, finished=time.time())
        write(OUT/'queue.json', jobs)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    assert platform.node() == 'foureyes'
    protocol = HERE/'noise_replication.json'
    p = json.loads(protocol.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in p['frozen_inputs'].items():
        assert digest(ROOT/name) == expected
    if args.initialize:
        assert not (OUT/'queue.json').exists()
        write(OUT/'queue.json', [dict(seed=s, status='pending') for s in p['source_seeds']])
        return
    lock = (ROOT/f'artifacts/gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi',f'--id={args.gpu}',
        '--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    owners = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True)
    assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    ii, jj = np.triu_indices(81,1)
    it = torch.as_tensor(ii, device='cuda')
    jt = torch.as_tensor(jj, device='cuda')

    def execute(job):
        start = time.monotonic()
        seed = job['seed']
        dest = OUT/f's{seed}'
        dest.mkdir(exist_ok=False)
        source = ROOT/f'artifacts/research/20261114/ripper/clues/completion_controls/s{seed}.npz'
        clues = ROOT/f'artifacts/research/20261114/ripper/clues/clue_inputs/s{seed}.npz'
        a, b = np.load(source), np.load(clues)
        clean_np = a['solutions'].astype(np.int64)
        free_np = ~b['given']
        assert np.array_equal(a['source_row'], b['source_row']) and free_np.sum(1).min() >= 9
        clean = torch.as_tensor(clean_np, device='cuda')
        free = torch.as_tensor(free_np, device='cuda')
        generator = torch.Generator(device='cuda').manual_seed(p['noise_rng_base'] + seed)
        stored_counts = {n: [] for n in p['sample_sizes']}
        stored_cells, stored_values = [], []
        thresholds = {n: [] for n in p['sample_sizes']}
        preflight = []
        tie_rows = 0
        for offset in range(0, p['draws'], p['batch_size']):
            assert time.monotonic()-start < p['per_job_wall_limit_s'], 'Prespecified compute envelope exceeded'
            batch = min(p['batch_size'], p['draws']-offset)
            keys = torch.rand((batch,3000,81), generator=generator, dtype=torch.float64, device='cuda')
            keys.masked_fill_(~free[None], float('inf'))
            values, indices = keys.topk(10, dim=2, largest=False, sorted=True)
            tied = (values.diff(dim=2) == 0).any(2)
            while bool(tied.any()):
                tie_rows += int(tied.sum())
                replacement = torch.rand((int(tied.sum()),81), generator=generator, dtype=torch.float64, device='cuda')
                replacement.masked_fill_(~free.expand(batch,-1,-1)[tied], float('inf'))
                keys[tied] = replacement
                values, indices = keys.topk(10, dim=2, largest=False, sorted=True)
                tied = (values.diff(dim=2) == 0).any(2)
            cells = indices[:,:,:9]
            original = clean.expand(batch,-1,-1)
            replacements = (original.gather(2,cells) + torch.randint(1,9,cells.shape,
                generator=generator,device='cuda')) % 9
            noisy = original.clone().scatter_(2,cells,replacements)
            assert (noisy != original).sum(2).eq(9).all()
            assert torch.equal(noisy[:,~free], original[:,~free])
            if offset == 0:
                sample = noisy[:2].cpu().numpy()
            for n in p['sample_sizes']:
                # Each dot product sums at most n exact 0/1 products (<2**24).
                encoded = torch.nn.functional.one_hot(noisy[:,:n],9).float()
                features = encoded.permute(0,2,1,3).reshape(batch,81,n*9)
                matrix = torch.bmm(features,features.transpose(1,2))
                assert torch.equal(matrix,matrix.round())
                counts = matrix[:,it,jt].to(torch.int16).cpu().numpy().astype(np.uint16)
                if offset == 0:
                    before = time.monotonic()
                    expected = np.stack([(row[:,ii] == row[:,jj]).sum(0) for row in sample[:,:n]])
                    assert np.array_equal(counts[:2],expected)
                    preflight.append(dict(n=n, draws=2, independent_cpu_count_seconds=time.monotonic()-before))
                stored_counts[n].append(counts)
                for row in counts:
                    unique = np.unique(row.astype(np.int64))
                    index = int(np.diff(unique).argmax()) if len(unique)>1 else None
                    thresholds[n].append(int((unique[index]+unique[index+1])//2) if index is not None else 0)
            stored_cells.append(cells.cpu().numpy().astype(np.uint8))
            stored_values.append(replacements.cpu().numpy().astype(np.uint8))
            write(dest/'progress.json', dict(seed=seed, draws_done=offset+batch,
                wall_s=time.monotonic()-start, pid=os.getpid(), gpu=args.gpu))
        torch.cuda.synchronize()
        np.savez_compressed(dest/'corruptions.npz', cells=np.concatenate(stored_cells), values=np.concatenate(stored_values))
        np.savez_compressed(dest/'counts.npz', **{f'n{n}':np.concatenate(stored_counts[n]) for n in p['sample_sizes']},
            **{f't{n}':np.asarray(thresholds[n],dtype=np.int64) for n in p['sample_sizes']})
        write(dest/'receipt.json', dict(status='complete', seed=seed, gpu=args.gpu, uuid=uuid,
            draws=p['draws'], sample_sizes=p['sample_sizes'], per_row_cells_changed=9,
            generation='Float64 uniform random priorities, reject any tied selection boundary; independent other-eight digit replacements.',
            resampled_tie_rows=tie_rows, cpu_preflight=preflight, wall_s=time.monotonic()-start,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), torch=torch.__version__, numpy=np.__version__,
            protocol_sha256=digest(protocol), source_sha256=digest(Path(__file__)),
            input_sha256={str(x.relative_to(ROOT)):digest(x) for x in (source,clues)},
            output_sha256={name:digest(dest/name) for name in ['corruptions.npz','counts.npz']}))
        print(json.dumps(dict(seed=seed, status='complete', draws=p['draws'], wall_s=time.monotonic()-start)),flush=True)

    while True:
        job = claim(args.gpu)
        if job is None:
            break
        try:
            execute(job)
        except BaseException:
            claim(args.gpu,job['seed'],'failed')
            raise
        claim(args.gpu,job['seed'],'complete')


if __name__ == '__main__':
    main()
