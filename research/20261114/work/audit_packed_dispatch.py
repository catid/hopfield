"""H2bs complete-source preservation and independently reconstructed paired cost."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from audit_packed_dispatch_cohort import verify,sha


def main(root,directory,workers):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());ph=sha(directory/'protocol.json');seal=json.loads((directory/'sealed-replays.json').read_text())
    assert seal['status']=='complete' and seal['protocol_sha256']==ph and seal['endpoint_aggregates_joined'] is False and len(seal['receipt_sha256'])==35
    for name,digest in seal['receipt_sha256'].items():assert sha(directory/name)==digest
    for key in ['source_sha256','source_evidence_sha256']:
        for name,digest in p[key].items():assert sha(root/name)==digest,name
    assert workers==p['budgets']['independent_final_audit_processes']==16 and p['budgets']['independent_final_audit_wall_seconds']==900
    packet=json.loads((directory/'source-packet.json').read_text());packet_hash=sha(directory/'source-packet.json');partials=directory/'audit-partials';partials.mkdir(exist_ok=False)
    tasks=[]
    for gpu in [0,1]:
        for kind,command,protocol,protocol_hash,ids in [('source-gates','gate',packet,packet_hash,np.r_[np.arange(64),np.arange(3840,3904)]),('warmup','timing',p,ph,np.arange(32))]:
            worker=json.loads((directory/f'{command}-worker{gpu}.json').read_text());assert worker['status']=='complete' and worker['worker_wall_seconds']<=(240 if command=='gate' else 2400) and worker['peak_cpu_rss_kib']<=8*1024**2
            for mode in ['original_order','packed']:tasks.append((f'{kind}-gpu{gpu}-{mode}',kind,(root,directory/kind/f'gpu{gpu}-{mode}',protocol,protocol_hash,ids,directory,command)))
    for pair in range(6):
        for mode in ['original_order','packed']:tasks.append((f'pair{pair}-{mode}','timing',(root,directory/'runs'/f'pair{pair}-{mode}',p,ph,np.arange(4096),directory,'timing')))
    results=[];pool=ProcessPoolExecutor(max_workers=workers)
    try:
        futures={pool.submit(verify,*args):(name,kind) for name,kind,args in tasks}
        for future in as_completed(futures):
            name,kind=futures[future];record=future.result();record.update(name=name,kind=kind);results.append(record)
            (partials/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'event':'audit_progress','completed':len(results),'of':20,'name':name}),flush=True)
    except BaseException:
        for child in list((getattr(pool,'_processes',None) or {}).values()):
            if child.is_alive():child.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    timed=[r for r in results if r['kind']=='timing'];assert len(timed)==12 and sum(r['model_pipeline_copies'] for r in timed)==147456
    pairs=[];values=[]
    for pair in range(6):
        bymode={r['mode']:r for r in timed if r['pair']==pair};assert set(bymode)=={'original_order','packed'}
        a,b=[bymode[mode]['complete_answer_seconds'] for mode in ['original_order','packed']];saving=Fraction(1)-Fraction(b)/Fraction(a);values.append(saving)
        pairs.append({'pair':pair,'original_order_seconds':a,'packed_seconds':b,'saving':float(saving)})
    values.sort();median=(values[2]+values[3])/2;elapsed=time.perf_counter()-started;assert elapsed<=900
    result={'status':'passed','hypothesis':'H2bs','protocol_sha256':ph,'seal_sha256':sha(directory/'sealed-replays.json'),'source_audit_sha256':p['source']['audit_sha256'],
        'source_preservation':'All per-input final outputs, source model selections, original margins, hidden steps/couplings and normalized internal LP transcripts preserved across all actual copies.',
        'cost_endpoint':{'pairs':pairs,'median_saving':float(median),'exact_median_fraction':str(median),'minimum_saving':.10,'decision':'supported' if median>=Fraction(1,10) else 'falsified'},
        'timed_model_pipeline_copies':147456,'fresh_scientific_inputs':0,'new_scientific_jobs':0,'new_scientific_evaluations':0,'source_gate_and_warmup_copies':sum(r['model_pipeline_copies'] for r in results if r['kind']!='timing'),
        'all_candidate_score_comparisons':sum(r['candidate_score_comparisons'] for r in results),'all_active_trace_states':sum(r['active_trace_states'] for r in results),'all_barrier_events':sum(r['barrier_events'] for r in results),
        'cohorts':sorted(results,key=lambda r:r['name']),'workers':workers,'seconds':elapsed,'auditor_sha256':sha(Path(__file__)),
        'scope':'Performance-only offline source-cohort experiment under a new shared complete-cost harness. No fresh accuracy observations, population interval, online latency or hardware-general claim.'}
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('directory',type=Path);parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    progress=args.directory/'audit-worker.json'
    if progress.exists() or (args.directory/'audit.json').exists():raise FileExistsError('Refusing an existing audit attempt')
    started=time.perf_counter();state={'status':'running','pid':os.getpid(),'wall_budget_seconds':900,'workers':args.workers};progress.write_text(json.dumps(state,indent=2)+'\n')
    def alarm(signum,frame):raise TimeoutError('Frozen independent audit budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,900)
    try:main(args.root,args.directory,args.workers)
    except BaseException as error:
        state.update(status='incomplete',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started,partial_receipts_retained=True);progress.write_text(json.dumps(state,indent=2)+'\n');raise
    else:state.update(status='complete',wall_seconds=time.perf_counter()-started);progress.write_text(json.dumps(state,indent=2)+'\n')
    finally:signal.setitimer(signal.ITIMER_REAL,0)
