"""H1ar: isolate equality-null calibration on unchanged acquired evidence."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import argparse,csv,hashlib,json,os,resource,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/research/20261114/ripper/permuted_noise';OUT=ROOT/'artifacts/research/20261114/ripper/null_replay'
PUBLIC=ROOT/'results/research/ripper';PROTOCOL=HERE/'protocol-null-replay-corrected.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def verify():
    p=json.loads(PROTOCOL.read_text())
    for path,digest in (p['code_sha256']|p['input_sha256']).items():assert sha(ROOT/path)==digest,path
    return p

def freeze():
    assert not PROTOCOL.exists()and not OUT.exists()
    code=[HERE/n for n in ['null_replay_study_corrected.py','null_replay_policy.py','audit_null_replay_corrected.py']]
    inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json',HERE/'protocol-null-replay.json',HERE/'null-replay-pre-run-correction.json',HERE/'null_replay_study.py',HERE/'audit_null_replay.py',HERE/'verify_null_replay_correction.py']
    for b in range(32):inputs += [PARENT/f's{b}'/n for n in ['observations.npz','fit.npz','receipt.json','calibration.json']]
    p=dict(id='H1ar-fixed-evidence-equality-null-comparison',frozen_utc=datetime.now(timezone.utc).isoformat(),banks=32,rows=1110,
        scope='Post-hoc mechanism replay on all32 unchanged H1ap observed masks/labels; no new acquisition, calibration, source rows, puzzle inputs or GPU work.',
        nulls=dict(calibrated='42288940519/64000000000000',exact_law='9/1600',nominal_iid='311/25600'),q1='1/9',boundary=648000,
        semantics='First threshold crossing excludes a pair permanently. Same original1110 row masks in every replay; no counterfactual adaptive re-selection. Retain unexposed pairs. Final-only threshold supports are descriptive and must not replace sequential decisions.',
        empirical_primary='At least29/32 exact-law supports retain all810 true edges. Supports iff all_true_retaining_count>=29, otherwise falsifies. Exact graph recovery, extra edges, nominal iid and final-only contrasts descriptive.',
        conditional_argument='Under supplied board-permutation law only, mask is fixed before independent current event/permutation. Given history/current source board, each queried true pair equals with probability9/1600; unqueried increment1. E[increment]=p*(q1/q0)+(1-p)*((1-q1)/(1-q0)), equal1 for exact q0=p and less than1 for nominaliid q0>p. This conditional martingale/supermartingale argument does not imply robustness to arbitrary correlated noise or a fresh empirical confirmation.',
        audit='Exact rational-power thresholds for each integer exposure0..1110; independent Decimal80 crossing/predecessor checks and Fraction multiplier expectation. Independently reconstruct every cumulative joint/equal state, every sequential first crossing and final-only support. Calibrated rule must exactly reproduce all archived first crossings/counts/supports. No banks filtered.',
        budget=dict(aggregate_cpu_seconds=120,wall_seconds=60,max_cpu_workers=8,gpu_seconds=0),
        code_sha256={str(f.relative_to(ROOT)):sha(f)for f in code},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in inputs})
    save(PROTOCOL,p);print(json.dumps({'protocol_sha256':sha(PROTOCOL)}))

def prepare():
    import numpy as np
    from null_replay_policy import thresholds
    p=verify();assert not OUT.exists();OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'thresholds.npz',**{a:thresholds(Fraction(q))for a,q in p['nulls'].items()})
    from audit_null_replay_corrected import thresholds_audit
    thresholds_audit()

def bank(seed):
    import numpy as np
    p=verify();assert 0<=seed<32;d=PARENT/f's{seed}';obs=np.load(d/'observations.npz',allow_pickle=False)
    masks,values=obs['given'],obs['values'];threshold=np.stack([np.load(OUT/'thresholds.npz')[a]for a in p['nulls']]);i,j=np.triu_indices(81,1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);first=np.full((3,3240),-1,np.int32);digests=[]
    for row in range(1110):
        cells=np.flatnonzero(masks[row]);view=np.ix_(cells,cells);joint[view]+=1;labels=values[row,cells];equal[view]+=(labels[:,None]==labels[None,:])
        m,c=joint[i,j],equal[i,j];digests.append(hashlib.sha256(m.tobytes()+c.tobytes()).hexdigest())
        for index in range(3):first[index,(first[index]<0)&(c>=threshold[index,m])]=row+1
    np.savez_compressed(OUT/f's{seed}.npz',joint_count=m,equal_count=c,first_crossing=first,support=first<0,final_only_support=c[None,:]<threshold[:,m],count_sha256=np.array(digests))
    from audit_null_replay_corrected import audit_bank
    a=audit_bank(seed);print(json.dumps(dict(seed=seed,status=a['status'])),flush=True)

def summary():
    import numpy as np
    p=verify();rows=[];hashes={};models=[]
    for seed in range(32):
        path=OUT/f's{seed}-audit.json';a=json.loads(path.read_text());assert a['status']=='validated'and a['protocol_sha256']==sha(PROTOCOL)and a['auditor_sha256']==sha(HERE/'audit_null_replay_corrected.py')
        assert a['model_sha256']==sha(OUT/f's{seed}.npz');rows+=a['rows'];hashes[str(path.relative_to(ROOT))]=sha(path)
        with np.load(OUT/f's{seed}.npz',allow_pickle=False)as z:models.append(z['support'])
    groups=[]
    for arm in p['nulls']:
        part=[r for r in rows if r['arm']==arm]
        groups.append(dict(arm=arm,**{k:sum(r[k]for r in part)for k in ['missing_true_edges','extra_edges','all_true_retained','exact_graph','final_only_reinstated_pairs','final_only_missing_true','final_only_extras']}))
    exact=next(g for g in groups if g['arm']=='exact_law');primary=dict(target='At least29/32 exact-law banks retain every true edge.',all_true_retaining_banks=exact['all_true_retained'],decision='supported_finite_bank_target'if exact['all_true_retained']>=29 else'falsified_finite_bank_target')
    PUBLIC.mkdir(parents=True,exist_ok=True);np.savez_compressed(PUBLIC/'null_replay_supports.npz',banks=np.arange(32),arms=np.array(list(p['nulls'])),supports=np.stack(models))
    ta=json.loads((OUT/'threshold-audit.json').read_text())
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),scope=p['scope'],primary=primary,grouped=groups,rows=rows,threshold_audit=ta,
        cumulative_pair_states_checked=115084800,calibrated_first_crossing_paths_matched=103680,artifact_sha256=hashes,support_export_sha256=sha(PUBLIC/'null_replay_supports.npz'),new_queries=0,new_calibrations=0,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0)
    save(OUT/'audit.json',result);save(PUBLIC/'null_replay.json',result)
    with(PUBLIC/'null_replay_per_bank.csv').open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1ar: known equality nulls on unchanged permutation-noise evidence','',f'The target of at least 29 exact-law graphs retaining every true edge is **{primary["decision"]}**: {exact["all_true_retained"]}/32 banks retain all true edges.','',
        '| Equality null | All true edges retained | Exact graphs | Missing true edges | Extras | Final-only reinstated pairs |','|---|---:|---:|---:|---:|---:|',
        '| BP AdamW: outside this source-only study; not run | — | — | — | — | — |']
    for g in groups:lines.append(f'| {g["arm"]} | {g["all_true_retained"]}/32 | {g["exact_graph"]}/32 | {g["missing_true_edges"]} | {g["extra_edges"]} | {g["final_only_reinstated_pairs"]} |')
    lines+=['','All 32 H1ap acquired masks and observed labels are unchanged. The replay changes only the equality-null probability: archived calibrated q0 = 42288940519/64000000000000; exact permutation-law q0 = 9/1600; or nominal iid eta = 0.05, giving q0 = 311/25600. The alternative remains 1/9 and the crossing bound 648000. These known-law values are supplied oracle information, not newly estimated calibration.','',
        'Pairs are removed permanently at the first crossing. Final-only threshold differences are reported separately; they do not reinstate sequentially excluded pairs. Every unexposed pair remains retained. The original acquisition masks are replayed rather than reselected using counterfactual exclusions.','',
        'For this supplied corruption law, the current mask is fixed before the independent board permutation. Conditional on history and the current clean source board, a queried true pair is equal with probability p = 9/1600. Its LR multiplier has expectation p(q1/q0) + (1-p)(1-q1)/(1-q0). This is exactly one at q0 = p and below one at nominal iid q0 > p; an unqueried pair has multiplier one. This is a conditional martingale/supermartingale statement for this law and predictable masks, not a universal correlated-noise guarantee.','',
        f'Thresholds use exact rational powers, independently checked by 3,333 Decimal80 endpoint/predecessor tests. Independent reconstruction checked all 115,084,800 cumulative pair states and matched all 103,680 archived calibrated first-crossing paths. Every bank and outcome is retained.','',
        'This post-hoc mechanism comparison uses no new observations, calibration fits, source boards, puzzle inputs or GPU work. Exact graph recovery and extra-edge costs are descriptive; the primary concerns true-edge retention only.','',
        '[Protocol](protocol-null-replay-corrected.json) · [Independent replay](audit_null_replay_corrected.py) · [Results](../../../results/research/ripper/null_replay.json)','']
    (HERE/'REPORT-null-replay.md').write_text('\n'.join(lines));print(json.dumps(dict(primary=primary,grouped=groups)),flush=True)

def launch():
    verify();execution=OUT.parent/'null_replay_execution';execution.mkdir(exist_ok=False);start=time.monotonic();active={};lock=threading.Lock();finished=threading.Event();stop=threading.Event();completed=[];ticks=os.sysconf('SC_CLK_TCK')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def cpu_seconds():
        a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);total=a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
        with lock:pids=list(active)
        for pid in pids:
            try:f=Path(f'/proc/{pid}/stat').read_text().split();total+=(int(f[13])+int(f[14]))/ticks
            except FileNotFoundError:pass
        return total
    def monitor():
        while not finished.is_set():
            wall=time.monotonic()-start;cpu=cpu_seconds()
            with lock:processes=list(active.values());count=len(completed)
            rec=dict(wall_s=wall,cpu_s=cpu,active_pids=[p.pid for p in processes],completed_commands=count);save(execution/'progress.json',rec)
            if wall>=60 or cpu>=120:
                stop.set();save(execution/'budget_exhausted.json',rec)
                for p in processes:
                    if p.poll()is None:p.terminate()
                return
            finished.wait(.1)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def run(action,seed=None):
        assert not stop.is_set();args=[sys.executable,str(Path(__file__)),action]+([]if seed is None else['--seed',str(seed)])
        with(execution/f'{action}-{seed}.log').open('w')as f:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            with lock:active[process.pid]=process
            code=process.wait()
            with lock:active.pop(process.pid,None);completed.append(dict(action=action,seed=seed,returncode=code))
            assert code==0 and not stop.is_set(),(action,seed,code)
    try:
        run('prepare')
        with ThreadPoolExecutor(max_workers=8)as workers:list(workers.map(lambda seed:run('bank',seed),range(32)))
        run('summary');wall=time.monotonic()-start;cpu=cpu_seconds();assert wall<60 and cpu<120 and not stop.is_set()
        runtime=dict(status='validated',wall_s=wall,aggregate_cpu_s=cpu,wall_ceiling_s=60,aggregate_cpu_ceiling_s=120,max_cpu_workers=8,gpu_seconds=0,commands=completed,wrapper_sha256=sha(__file__))
        result=json.loads((OUT/'audit.json').read_text());result['runtime']=runtime;save(OUT/'runtime.json',runtime);save(PUBLIC/'null_replay_runtime.json',runtime);save(OUT/'audit.json',result);save(PUBLIC/'null_replay.json',result)
        with(HERE/'REPORT-null-replay.md').open('a')as f:f.write(f'\nExecution used {cpu:.4f} CPU-seconds and {wall:.4f} wall seconds within the frozen 120 CPU-second / 60-second ceilings.\n')
        print(json.dumps(dict(primary=result['primary'],grouped=result['grouped'],wall_s=wall,aggregate_cpu_s=cpu)),flush=True)
    finally:
        finished.set();watcher.join(timeout=1)
        with lock:processes=list(active.values())
        for p in processes:
            if p.poll()is None:p.terminate()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','bank','summary','launch']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.action=='bank':bank(args.seed)
    else:globals()[args.action]()
