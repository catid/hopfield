"""Independent GPU matrix-count audit of every archived uniform-mask draw."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT,UP,HERE,save,sha,now

OUT = ROOT/'artifacts/research/20261114/ripper/masks'
PROTOCOL = HERE/'protocol-masks.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--seeds',type=int,nargs='+',required=True)
    a = ap.parse_args(); assert platform.node() == 'ripper'
    lock = (ROOT/f'artifacts/gpu{a.gpu}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    from clues_estimator import estimate
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    p = json.loads(PROTOCOL.read_text()); mp = ROOT/p['input_manifest']; assert sha(mp) == p['input_manifest_sha256']
    manifest = json.loads(mp.read_text()); pair = np.triu_indices(81,1)
    for seed in a.seeds:
        start = time.monotonic(); d = OUT/f's{seed}'; receipt = json.loads((d/'receipt.json').read_text())
        assert receipt['protocol_sha256'] == sha(PROTOCOL) and receipt['wrapper_sha256'] == sha(HERE/'mask_count.py')
        for name,h in receipt['output_sha256'].items(): assert sha(d/name) == h
        e = next(e for e in manifest['inputs'] if e['seed']==seed)
        sol = np.load(ROOT/e['completion_path'])['solutions']
        original = np.load(ROOT/e['clues_path']); counts = original['given'].sum(1)
        archive = np.load(d/'packed_masks.npy',mmap_mode='r'); draws = np.load(d/'draws.npz')
        first = np.load(d/'first_clues.npz'); assert np.array_equal(first['source_row'],original['source_row'])
        xsol = torch.nn.functional.one_hot(torch.as_tensor(sol.astype(np.int64),device='cuda'),9).float()
        rng = np.random.default_rng(20261120+seed)
        ranks = np.broadcast_to(np.arange(81,dtype=np.int16),(3000,81)).copy()
        cpu_checks = 0; matrix_checks = 0
        for lo in range(0,1024,16):
            hi = lo+16
            masks = np.unpackbits(archive[lo:hi],axis=2)[:,:,:81].astype(bool)
            assert np.array_equal(masks.sum(2),np.broadcast_to(counts,(16,3000)))
            for k in range(16):
                expected = rng.permuted(ranks,axis=1)<counts[:,None]
                assert np.array_equal(expected,masks[k])
            observed = xsol[None]*torch.as_tensor(masks,device='cuda')[:,:,:,None]
            for z,n in enumerate(p['sizes']):
                # Each feature is an observed (row,digit). Dot products count co-observed equality.
                features = observed[:,:n].permute(0,1,3,2).reshape(16,n*9,81)
                matrix = torch.bmm(features.transpose(1,2),features)
                equal = matrix[:,pair[0],pair[1]].cpu().numpy()
                assert np.array_equal(equal,equal.astype(np.int32))
                support = equal == 0
                saved = np.unpackbits(draws['packed_supports'][lo:hi,z],axis=1).astype(bool)
                assert np.array_equal(support,saved)
                assert np.array_equal(support.sum(1),draws['retained_edges'][lo:hi,z])
                for k in range(16):
                    assert hashlib.sha256(support[k].tobytes()).hexdigest() == receipt['support_bytes_sha256'][lo+k][z]
                    if lo+k in (0,1023):
                        values = np.where(masks[k,:n],sol[:n],-1).astype(np.int8)
                        cs,ce,_ = estimate(values,masks[k,:n])
                        assert np.array_equal(cs,support[k]) and np.array_equal(ce,equal[k])
                        cpu_checks += 1
                matrix_checks += 16
            if lo == 0:
                assert np.array_equal(first['given'],masks[0])
                assert np.array_equal(first['values'],np.where(masks[0],sol,-1))
            if lo%128 == 0 or hi == 1024:
                save(d/'audit_progress.json',{'utc':now(),'draws_verified':hi,'elapsed_s':time.monotonic()-start})
        # Independent missing-value invariance on the preselected inference model.
        baseline = np.load(d/'first_model.npz'); values = first['values'][:300].copy(); given = first['given'][:300]
        hidden = ~given; gates = 0
        for replacement in (-99,0,8):
            values[hidden] = replacement; cs,ce,_ = estimate(values,given)
            assert np.array_equal(cs,baseline['support']) and np.array_equal(ce,baseline['equal_count']); gates += 1
        values[hidden] = np.random.default_rng(20261122+seed).permutation(sol[:300][hidden])
        cs,ce,_ = estimate(values,given)
        assert np.array_equal(cs,baseline['support']) and np.array_equal(ce,baseline['equal_count']); gates += 1
        torch.cuda.synchronize()
        rec = {'status':'validated','utc':now(),'seed':seed,'gpu':a.gpu,'host':platform.node(),
            'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'count_receipt_sha256':sha(d/'receipt.json'),
            'matrix_prefix_checks':matrix_checks,'independent_cpu_dense_prefix_checks':cpu_checks,
            'full_rng_mask_replay_verified':True,'mask_count_preservation_verified':True,
            'hidden_value_invariance_checks':gates,'elapsed_s':time.monotonic()-start,
            'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
            'arithmetic':'FP32 dot products of 0/1 entries; sums are integers <=3000, exactly representable. TF32 disabled.'}
        save(d/'audit_counts.json',rec); print(json.dumps(rec),flush=True)


if __name__ == '__main__': main()
