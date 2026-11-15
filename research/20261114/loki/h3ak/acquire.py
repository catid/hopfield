"""Fixed random positive acquisition; no N15 arrays, scores or geometry."""
import itertools
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen

def main():
    p=frozen();assert not(OUT/'source-positive-seal.json').exists();population=np.load(ROOT/p['acquisition']['population_path']);assert population.shape==(1932,169);records=[]
    for bank,seed in enumerate(p['acquisition']['seeds']):
        indices=np.random.default_rng(seed).choice(1932,64,replace=False);selected=population[indices];assert len(np.unique(selected,axis=0))==64 and(selected.sum(1)==13).all()
        # Source-only positive label gate, separate from the repair process.
        for board in selected:
            for a,b in itertools.combinations(np.flatnonzero(board),2):
                ar,ac=divmod(int(a),13);br,bc=divmod(int(b),13);assert ar!=br and ac!=bc and abs(ar-br)!=abs(ac-bc)and(abs(ar-br),abs(ac-bc))not in[(1,3),(3,1)]
        folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);np.save(folder/'positive-boards.npy',selected);np.save(folder/'positive-population-indices.npy',indices);records.append(dict(bank=bank,seed=seed,selected_positive_count=64,indices=indices.tolist(),boards_sha256=sha(folder/'positive-boards.npy'),indices_sha256=sha(folder/'positive-population-indices.npy'),source_validity_independently_gated=True))
    overlap=[[len(set(a['indices'])&set(b['indices']))for b in records]for a in records];save(OUT/'source-positive-seal.json',dict(status='passed',utc=now(),banks=records,overlap_counts=overlap,population_sha256=sha(ROOT/p['acquisition']['population_path']),no_N15_data_or_geometry_accessed=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
