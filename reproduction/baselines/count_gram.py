"""Official full operation count with the main-table sample count for each model."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(Path.cwd()),str(ROOT/'code/gram'),str(ROOT/'code/trm'),str(ROOT/'code/flops')]
import flops as G
from trm_bench import bench_flops_full
if __name__=='__main__':
 out,samples,*ov=sys.argv[1:]
 n=int(samples)
 bench_flops_full(out,ov,finish=lambda rec,outputs,batch,model:G.selection(rec,outputs,batch,model,n=n))
