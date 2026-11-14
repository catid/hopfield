"""Fresh frozen streams of the unchanged official Sudoku augmentation function."""
import ast
import concurrent.futures
import json
import os
import platform
import time
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/coverage'


def sample(base_index):
    p=json.loads((HERE/'protocol-coverage.json').read_text());boards=np.load(OUT/'bases.npz')['solutions'];board=boards[base_index].reshape(9,9).astype(np.int64)+1
    tree=ast.parse((OUT/'builder.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shuffle_sudoku')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(OUT/'builder.py'),'exec'),ns);shuffle=ns['shuffle_sudoku']
    pair=np.triu_indices(81,1);flat=board.ravel();start_seen=flat[pair[0]]==flat[pair[1]]
    cells=np.arange(81);r,c=cells//9,cells%9;peers=((r[:,None]==r[None])|(c[:,None]==c[None])|((r[:,None]//3==r[None]//3)&(c[:,None]//3==c[None]//3)))[pair]
    variants=p['variants_grid'];extra=np.empty((256,len(variants)),np.int32);hits=np.zeros(2,np.int64);missing=0;beg=time.perf_counter()
    for stream in range(256):
        rng_seed=2026111500+10000*base_index+stream;np.random.seed(rng_seed)
        seen=start_seen.copy();extra[stream,0]=int((~seen&~peers).sum());col=1
        for v in range(1,128):
            _,sol=shuffle(board,board);flat=sol.ravel()
            hits += [flat[0]==flat[12],flat[0]==flat[30]]
            seen |= flat[pair[0]]==flat[pair[1]]
            if col<len(variants) and v==variants[col]:
                support=~seen;missing+=int((~support&peers).sum());extra[stream,col]=int((support&~peers).sum());col+=1
        assert col==len(variants)
    dest=OUT/'samples'/f'base{base_index:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'counts.npz',extra_edges=extra,representative_pair_equalities=hits)
    save(dest/'receipt.json',{'status':'complete','base_index':base_index,'utc':now(),'pid':os.getpid(),'host':platform.node(),'numpy':np.__version__,'wall_s':time.perf_counter()-beg,'draws':256*127,'missing_true_peer_events':missing,'counts_sha256':sha(dest/'counts.npz'),'wrapper_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol-coverage.json')})
    print(json.dumps({'base':base_index,'status':'complete','wall_s':time.perf_counter()-beg}),flush=True)


def main():
    assert platform.node()=='ripper';assert np.__version__=='2.3.4'
    p=json.loads((HERE/'protocol-coverage.json').read_text())
    for name,h in p['inputs'].items():assert sha(OUT/name)==h
    with concurrent.futures.ProcessPoolExecutor(8) as pool:list(pool.map(sample,range(32)))


if __name__=='__main__':main()
