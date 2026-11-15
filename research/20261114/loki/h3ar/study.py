"""Post-selection complete N13 validation of every likelihood maximizer."""
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
    p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();start=cpu();wall=time.perf_counter();infos=[];rows=[]
    def expired(*args):raise TimeoutError('H3ar120CPU/180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        subprocess.run([sys.executable,str(SRC/'source.py')],check=True);selection=read(OUT/'source-selection.json');frozen(True);helper=load(SRC.parent/'h3ao/evaluate.py','h3ar_target_helpers');ordinary=load(SRC.parent/'h3ad/generate.py','h3ar_target_ordinary').queens(13);assert len(ordinary)==73712;binary=OUT/'source_reverse'
        def enumerated(F,name):
            inp=OUT/f'{name}F.bin';out=OUT/f'{name}columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert len(columns)==int(r.stdout);return columns
        queenF=helper.graph(13);assert np.array_equal(ordinary,enumerated(queenF,'ordinary'));references={}
        for family,cfg in enumerate(p['families']):
            Ftrue=helper.graph(13,tuple(cfg['leaper']));truth=np.load(ROOT/cfg['target_path']);assert np.array_equal(truth,enumerated(Ftrue,f'family{family}'));h=helper.hist(truth);keys=np.flatnonzero(h);folder=OUT/f'family{family}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'input-universe.npz',keys=keys,counts=h[keys]);np.save(folder/'true-F.npy',Ftrue);np.save(folder/'true-columns.npy',truth);references[family]=(truth,Ftrue,keys,{r.tobytes()for r in truth})
        full={r['group']:np.load(OUT/f"support{r['group']}/model.npz")['F']for r in selection['supports']};assert all((F>=queenF).all()for F in full.values());cells=13*np.arange(13)[None,:]+ordinary.astype(np.int64);ii,jj=np.triu_indices(13,1);ai,aj=np.triu_indices(169,1)
        for record in selection['supports']:
            group=record['group'];F=full[group];truth,Ftrue,keys,truekeys=references[record['family']];columns=ordinary[F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,enumerated(F,f'support{group}'));valid=np.array([r.tobytes()in truekeys for r in columns]);h=helper.hist(columns[valid]);flags=h[keys]>0;folder=OUT/f'support{group}';np.savez_compressed(folder/'capacity.npz',columns=columns,valid_state_mask=valid,key_capacity_bits=np.packbits(flags,bitorder='little'));info=dict(group=group,family=record['family'],complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(flags.sum()),capacity_pct=100*float(flags.mean()),meets_95_percent=20*int(flags.sum())>=19*len(keys),exact_graph=bool(np.array_equal(F,Ftrue)),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),capacity_sha256=sha(folder/'capacity.npz'));infos.append(info)
        decisions=[]
        for bank in selection['banks']:
            members=[dict(candidate=r['candidate'],group=r['group'],arm='ML_maximizer')for r in bank['maximizers']]+[dict(candidate=None,group=bank['consistency_control_group'],arm='consistency_envelope')]
            for member in members:
                info=infos[member['group']];row=dict(index=bank['index'],family=bank['family'],bank=bank['bank'],**member,**{k:v for k,v in info.items()if k not in('group','family')},primary=member['arm']=='ML_maximizer');row['decision']=('supported'if row['meets_95_percent']and row['invalid_zero_F_states']==0 else'falsified')if row['primary']else'descriptive';rows.append(row)
            primary=[r for r in rows if r['index']==bank['index']and r['primary']];decision='supported'if all(r['decision']=='supported'for r in primary)else'falsified';decisions.append(dict(index=bank['index'],family=bank['family'],maximizers=len(primary),unique=bank['unique_maximum'],decision=decision,all_maximizers_exact_graph=all(r['exact_graph']for r in primary),maximizer_and_consistency_control_F_identical=all(np.array_equal(full[r['group']],full[bank['consistency_control_group']])for r in primary)));print('BANK',bank['index'],decision,bank['unique_maximum'],[(r['capacity_pct'],r['invalid_zero_F_states'],r['exact_graph'])for r in primary],flush=True)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in decisions)else'falsified',rows=rows,bank_decisions=decisions,support_results=infos,source_selection_sha256=sha(OUT/'source-selection.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_supports=infos,completed_rows=rows,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),partial_results_usable=False));raise
if __name__=='__main__':main()
