"""CPU-only verification and timing inference for frozen H1d."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now

BASE=ROOT/'artifacts/research/20261114/ripper/sparse'
PUBLIC=ROOT/'results/research/ripper'


def main():
    pre=[];hashes={};cells=[];timings=[];alltimes=[]
    for k in (10,17,21):
        for s in range(6):
            f=BASE/f'preflight-k{k}-s{s}-n256-c64.json';r=json.loads(f.read_text())
            assert r['status']=='passed' and r['backend_sha256']==sha(HERE/'sparse_backend.py')
            assert len(r['checks'])==65 and all(c['state_fields']==25 and not c['mismatches'] for c in r['checks'])
            hashes[str(f.relative_to(ROOT))]=sha(f);pre.append(r)
    for k in (10,17,21):
        for n in (128,512,2048):
            f=BASE/f'benchmark/k{k}-n{n}/receipt.json';r=json.loads(f.read_text())
            assert r['status']=='complete' and r['protocol_sha256']==sha(HERE/'protocol-sparse.json')
            assert r['backend_sha256']==sha(HERE/'sparse_backend.py') and r['wrapper_sha256']==sha(HERE/'sparse_benchmark.py')
            assert len(r['runs'])==14 and len(r['pairs'])==7
            assert all(p['exact_state_and_boards'] and not p['mismatches'] for p in r['pairs'])
            row={'k':k,'n':n,'gpu':r['cell']['gpu'],'repeats':6};t=[]
            for b in ('general','sparse'):
                rs=sorted((x for x in r['runs'] if x['backend']==b and x['repeat']>=0),key=lambda x:x['repeat'])
                assert [x['repeat'] for x in rs]==list(range(6));t.append([x['inference_s'] for x in rs])
                for x in rs:timings.append({'k':k,'n':n,'gpu':r['cell']['gpu'],**x})
                for key in ('inference_s','complete_s','setup_s','moves_gpu_s'):
                    row[f'{b}_{key}']=float(np.median([x[key] for x in rs]))
            for key in ('inference_s','complete_s','moves_gpu_s'):
                row[key+'_speedup']=row['general_'+key]/row['sparse_'+key]
            cells.append(row);alltimes.append(t);hashes[str(f.relative_to(ROOT))]=sha(f)
    t=np.array(alltimes);rng=np.random.default_rng(20261114);boot=[]
    for _ in range(5000):
        ids=rng.integers(0,9,9);ratios=[]
        for i in ids:
            ix=rng.integers(0,6,6);v=t[i][:,ix]
            ratios.append(np.median(v[0])/np.median(v[1]))
        boot.append(float(np.exp(np.log(ratios).mean())))
    effect=float(np.exp(np.log([c['inference_s_speedup'] for c in cells]).mean()));ci=np.quantile(boot,[.025,.975]).tolist()
    decision='supported_for_frozen_protocol' if ci[0]>=1.5 else 'falsified_for_frozen_protocol' if ci[1]<1.5 else 'inconclusive'
    rec={'status':'validated','utc':now(),'primary_geomean_warm_inference_speedup':effect,'primary_ci95':ci,'decision':decision,'complete_geomean_speedup':float(np.exp(np.log([c['complete_s_speedup'] for c in cells]).mean())),'kernel_call_geomean_speedup':float(np.exp(np.log([c['moves_gpu_s_speedup'] for c in cells]).mean())),'preflight_models':18,'preflight_state_comparisons':1170,'state_fields':25,'preflight_puzzles_per_model':256,'preflight_checks_per_model':64,'timed_cells':9,'exact_final_state_and_board_pairs':63,'cells':cells,'input_sha256':hashes,'limitations':['Bounded measured exact parity, not universal mathematical floating-point transition proof.','RTX PRO6000 Blackwell Max-Q, sameGPU paired timings; does not establish crosshardware speed.','One trainedmodel per k in timing matrix; sixseeds each were included in paritygate.','Data/model loading excluded; setup and completeconstructor+inference included separately.','CUDAevent call durations include devicepreparation and sparse/general move kernel; refresh and bookkeeping are in wall inference timing.','Bootstrap interval describes cell/repeat variability, not performance across allmodels/hardware.','Stock SparseSearch rejects variabledegree graphs; derivative uses unique zero padding,NOBIAS=False,hscale1,GeneralSearch refresh.'],'new_inference':False}
    save(BASE/'audit.json',rec);save(PUBLIC/'sparse.json',rec)
    with (PUBLIC/'sparse_timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
    p=PUBLIC/'summary.json';summary=json.loads(p.read_text());summary['sparse_backend']=rec;save(p,summary)
    lines=['# H1d: exact sparse backend and measured speed','',f'Warm inference geometric-mean speedup: **{effect:.3f}×**, timing bootstrap95% interval **[{ci[0]:.3f}, {ci[1]:.3f}]×**. The frozen ≥1.5× hypothesis is **{decision}**. Constructor plus inference geometric-mean speedup is **{rec["complete_geomean_speedup"]:.3f}×**.','',
           'The18models passed1,170 complete25field state comparisons plus best/certificate board arrays. Each used256development puzzles and64checks of4096moves. All63 paired warmup/timed full searches matched final state and boards exactly. This is extensive bounded empirical parity, not a universal floating-point proof.','',
           '| k | Puzzles | General inference (s) | Sparse inference (s) | Inference speedup | Complete speedup |','|---:|---:|---:|---:|---:|---:|']
    for c in cells:lines.append(f'| {c["k"]} | {c["n"]} | {c["general_inference_s"]:.4f} | {c["sparse_inference_s"]:.4f} | {c["inference_s_speedup"]:.3f}× | {c["complete_s_speedup"]:.3f}× |')
    lines+=['','Stock SparseSearch fails its uniform-degree assertion on these learned graphs. The derivative pads with unique zero-weight neighbors, forces bias-aware arithmetic with unit hscale, preserves floating fields and the original GeneralSearch refresh, and changes no model energy/focus offsets. No one-level integer approximation is used.','',
            'Timing uses the same exclusive GPU for paired backends, alternating order across six repetitions after one warmup per backend. Each search uses262144moves per chain and fourchains. Reports include constructor/conversion setup separately. Loading input files and constructing the shared dense S tensor are outside timing. Capture, compaction and periodic refresh are included. CUDAevent move-call timings include device preparation.','',
            '[Frozen protocol](protocol-sparse.json) · [Backend](sparse_backend.py) · [Preflight](sparse_preflight.py) · [Benchmark](sparse_benchmark.py) · [Public evidence](../../../results/research/ripper/sparse.json)','']
    (HERE/'REPORT-sparse.md').write_text('\n'.join(lines))
    print(json.dumps({k:rec[k] for k in ('primary_geomean_warm_inference_speedup','primary_ci95','complete_geomean_speedup','decision')}))


if __name__=='__main__':main()
