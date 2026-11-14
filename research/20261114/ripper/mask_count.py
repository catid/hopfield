"""H1q uniform-mask draws; equality-list counting avoids all 3240 pairs per row."""
import argparse
import hashlib
import json
import time
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT = ROOT/'artifacts/research/20261114/ripper/masks'
PROTOCOL = HERE/'protocol-masks.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); a = ap.parse_args()
    assert np.__version__ == '2.3.4'
    p = json.loads(PROTOCOL.read_text()); mp = ROOT/p['input_manifest']
    assert sha(mp) == p['input_manifest_sha256']; manifest = json.loads(mp.read_text())
    e = next(e for e in manifest['inputs'] if e['seed']==a.seed)
    for key in ('clues','completion'): assert sha(ROOT/e[key+'_path']) == e[key+'_sha256']
    original = np.load(ROOT/e['clues_path']); complete = np.load(ROOT/e['completion_path'])
    sol = complete['solutions']; source = complete['source_row']; counts = original['given'].sum(1)
    assert np.array_equal(source,original['source_row']) and len(sol)==3000
    dest = OUT/f's{a.seed}'; dest.mkdir(parents=True,exist_ok=False)
    ordered = np.argsort(sol,axis=1,kind='stable').reshape(3000,9,9)
    u,v = np.triu_indices(9,1)
    left = ordered[:,:,u].reshape(3000,324); right = ordered[:,:,v].reshape(3000,324)
    ii = np.minimum(left,right); jj = np.maximum(left,right)
    edge = ii*(161-ii)//2 + jj-ii-1
    rows = np.arange(3000)[:,None]
    assert np.all(sol[rows,left] == sol[rows,right])
    ranks = np.broadcast_to(np.arange(81,dtype=np.int16),(3000,81)).copy()
    rng = np.random.default_rng(20261120+a.seed)
    masks = np.lib.format.open_memmap(dest/'packed_masks.npy',mode='w+',dtype=np.uint8,shape=(1024,3000,11))
    packed = np.empty((1024,3,405),np.uint8); retained = np.empty((1024,3),np.int32)
    support_hash = []; timing = {'mask_generation_s':0.,'equality_gather_s':0.,'counting_s':0.}
    begin = time.monotonic()
    for draw in range(1024):
        t = time.monotonic(); given = rng.permuted(ranks,axis=1)<counts[:,None]
        assert np.array_equal(given.sum(1),counts)
        timing['mask_generation_s'] += time.monotonic()-t
        masks[draw] = np.packbits(given,axis=1)
        t = time.monotonic(); exposed = given[rows,left]&given[rows,right]
        timing['equality_gather_s'] += time.monotonic()-t
        hashes = []; first = None
        t = time.monotonic()
        for z,n in enumerate(p['sizes']):
            equal = np.bincount(edge[:n][exposed[:n]],minlength=3240).astype(np.int32)
            support = equal == 0; retained[draw,z] = support.sum(); packed[draw,z] = np.packbits(support)
            hashes.append(hashlib.sha256(support.tobytes()).hexdigest())
            if n == 300: first = (support,equal)
        timing['counting_s'] += time.monotonic()-t
        support_hash.append(hashes)
        if draw == 0:
            np.savez_compressed(dest/'first_clues.npz',values=np.where(given,sol,-1).astype(np.int8),given=given,source_row=source)
            np.savez_compressed(dest/'first_model.npz',support=first[0],equal_count=first[1])
            save(dest/'first_stream.json',{'status':'prepared','utc':now(),'seed':a.seed,'draw':0,
                'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
                'clues_path':str((dest/'first_clues.npz').relative_to(ROOT)),'clues_sha256':sha(dest/'first_clues.npz'),
                'model_path':str((dest/'first_model.npz').relative_to(ROOT)),'model_sha256':sha(dest/'first_model.npz'),
                'support_bytes_sha256':hashes[0],'n':300,'source_completion_sha256':e['completion_sha256']})
        if draw%64 == 0 or draw == 1023:
            save(dest/'progress.json',{'utc':now(),'seed':a.seed,'completed_draws':draw+1,'total_draws':1024,
                'elapsed_s':time.monotonic()-begin,'phase_s':timing})
    masks.flush(); del masks
    np.savez_compressed(dest/'draws.npz',retained_edges=retained,packed_supports=packed)
    save(dest/'receipt.json',{'status':'complete','utc':now(),'seed':a.seed,'draws':1024,
        'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'input_manifest_sha256':sha(mp),
        'numpy':np.__version__,'algorithm':'Observe only the 324 equal-digit pairs in each valid complete board, then count exposed pairs. No peer graph.',
        'output_sha256':{n:sha(dest/n) for n in ['packed_masks.npy','draws.npz','first_clues.npz','first_model.npz','first_stream.json']},
        'support_bytes_sha256':support_hash,'elapsed_s':time.monotonic()-begin,'phase_s':timing})
    print(json.dumps({'seed':a.seed,'draws':1024,'elapsed_s':time.monotonic()-begin,'phase_s':timing}),flush=True)


if __name__ == '__main__': main()
