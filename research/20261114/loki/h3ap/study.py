"""Verify conditional safety construction, then test exact finite input capacity."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=cpu();wall=time.perf_counter();infos=[];rows=[]
    def expired(*args):raise TimeoutError('H3ap120CPU/180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        subprocess.run([sys.executable,str(SRC/'source.py')],check=True);source=read(OUT/'source-outcomes.json');frozen(True);helper=load(SRC.parent/'h3ao/evaluate.py','h3ap_complete_helpers');reference=load(SRC.parent/'h3ad/generate.py','h3ap_ordinary_reference');ordinary=reference.queens(13);assert len(ordinary)==73712;queenF=helper.graph(13);binary=OUT/'reverse_literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,name):
            inp=OUT/f'{name}-F.bin';out=OUT/f'{name}-columns.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert len(columns)==int(result.stdout);return columns
        assert np.array_equal(ordinary,enumerate_F(queenF,'ordinary'));references={}
        for family,cfg in enumerate(p['families']):
            truth=np.load(ROOT/cfg['target_path']);Ftrue=helper.graph(13,tuple(cfg['leaper']));assert np.array_equal(truth,reference.filter_leaper(ordinary,tuple(cfg['leaper'])))and np.array_equal(truth,enumerate_F(Ftrue,f'family{family}'));h=helper.hist(truth);keys=np.flatnonzero(h);folder=OUT/f'family{family}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'input-universe.npz',keys=keys,counts=h[keys]);np.save(folder/'true-columns.npy',truth);np.save(folder/'true-F.npy',Ftrue);references[family]=(truth,Ftrue,keys,{row.tobytes()for row in truth})
        full={r['group']:np.load(OUT/f"support{r['group']}/F.npz")['F']for r in source['supports']};assert all((F>=queenF).all()for F in full.values())
        for row in source['rows']:
            if row['arm']!='published_filtered':assert(full[row['group']]>=references[row['family']][1]).all(),'Conditional safety construction failed; no independent safety-success claim'
        save(OUT/'construction-safety-seal.json',dict(utc=now(),status='verified_construction_condition',all_unfiltered_supports_contain_true_F=True,all_supports_contain_ordinary_queen_F=True,interpretation=p['safety_theorem'],not_an_independent_empirical_safety_hypothesis=True))
        cells=13*np.arange(13)[None,:]+ordinary.astype(np.int64);ii,jj=np.triu_indices(13,1);ai,aj=np.triu_indices(169,1)
        for record in source['supports']:
            group=record['group'];family=record['family'];F=full[group];truth,Ftrue,keys,truekeys=references[family];columns=ordinary[F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,enumerate_F(F,f'support{group}'));valid=np.array([row.tobytes()in truekeys for row in columns]);h=helper.hist(columns[valid]);flags=h[keys]>0;folder=OUT/f'support{group}';np.savez_compressed(folder/'capacity.npz',columns=columns,valid_state_mask=valid,key_capacity_bits=np.packbits(flags,bitorder='little'));info=dict(group=group,family=family,complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(flags.sum()),capacity_pct=100*float(flags.mean()),meets_95_percent=20*int(flags.sum())>=19*len(keys),true_state_retention_pct=100*float(valid.sum()/len(truth)),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),capacity_sha256=sha(folder/'capacity.npz'));infos.append(info)
        prior=read(ROOT/p['evaluation']['published_filtered_summary'])
        for src in source['rows']:
            info=infos[src['group']];row=dict(src,**{k:v for k,v in info.items()if k not in('group','family')},decision=('supported'if info['meets_95_percent']else'falsified')if src['primary']else'descriptive')
            if src['arm']=='published_filtered':
                old=next(r for r in prior['rows']if r['index']==src['index']and r['dose']==16);assert all(old[k]==row[k]for k in('capacity_pct','covered_keys','invalid_zero_F_states','false_forbidden_edges','missed_true_edges'))
            else:assert row['invalid_zero_F_states']==row['missed_true_edges']==0
            rows.append(row);print('ARM',src['index'],src['arm'],row['capacity_pct'],row['false_forbidden_edges'],row['invalid_zero_F_states'],row['decision'],flush=True)
        primary=[r for r in rows if r['primary']];assert len(primary)==8
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['meets_95_percent']for r in primary)else'falsified',rows=rows,primary=primary,support_results=infos,safety='verified_conditional_construction_not_independent_empirical_endpoint',source_outcomes_sha256=sha(OUT/'source-outcomes.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_supports=infos,completed_rows=rows,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),partial_results_usable=False));raise
if __name__=='__main__':main()
