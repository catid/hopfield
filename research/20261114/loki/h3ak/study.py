"""Sequential source-only repair seal, complete compilation, then seen-universe evaluation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
from compiler import build_validate,compile_supports,reverse

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

def histogram(columns):
    hist=np.zeros(226**3,np.int64);indices={k:np.array(list(itertools.combinations(range(15),k)))for k in [1,2,3]}
    for lo in range(0,len(columns),4096):
        cells=15*np.arange(15)[None,:]+columns[lo:lo+4096];pieces=[]
        for k in [1,2,3]:
            keys=((cells[:,indices[k]]+1)*np.array([226**(2-j)for j in range(k)],np.int64)).sum(-1);pieces.append(keys.reshape(-1))
            if lo==0:
                literal=np.array([sum((int(c)+1)*226**(2-j)for j,c in enumerate(subset))for row in cells[:17]for subset in itertools.combinations(row,k)]);assert np.array_equal(literal,keys[:17].reshape(-1))
        hist+=np.bincount(np.concatenate(pieces),minlength=226**3)
    assert hist.sum()==575*len(columns);return hist

def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start_cpu=cpu();rows=[];repair=None;compiled=[];validation=None
    def expired(*args):raise TimeoutError('H3ak120CPU-second/180wall-second positive-witness stage')
    signal.signal(signal.SIGUSR1,expired)
    try:
        for name in ['acquire','repair']:subprocess.run([sys.executable,str(SRC/f'{name}.py')],check=True)
        repair=read(OUT/'repair-seal.json');assert repair['status']=='complete_source_only_repair'and repair['artifact_read_allowlist_enforced'];validation=build_validate(p,cpu);compiled=compile_supports(repair)
        # The first target/reference integrity reads occur only after source repair and all learned enumerations seal.
        frozen('target');save(OUT/'progress.json',dict(stage='post_seal_true_reference',utc=now()));ref=ROOT/p['evaluation']['reference_root'];Ftrue=np.load(ref/'truth/F.npy');truthcache=np.load(ref/'truth/cache.npz');truecols=truthcache['columns'];out=OUT/'verifier-io/true-reference.columns.bin';run=subprocess.run([str(OUT/'truth-camel'),'15',str(out)],capture_output=True,text=True,check=True);assert int(run.stdout)==len(truecols)and np.array_equal(np.fromfile(out,np.uint8).reshape(-1,15),truecols)and np.array_equal(reverse(Ftrue,15,'true-reference'),truecols);u=np.load(ref/'input-universe.npz');eligible=u['eligible_keys'];clues=u['eligible_clues'];truehist=histogram(truecols);assert np.array_equal(np.flatnonzero(truehist),u['all_keys'])and np.array_equal(truehist[eligible],u['eligible_true_counts']);baseline=read(ref/'summary.json');key=lambda a:a.astype(np.int64)@np.array([15**j for j in range(14,-1,-1)],np.int64);truthkeys=key(truecols);results=[]
        for comp in compiled:
            group=comp['group'];save(OUT/'progress.json',dict(stage='exact_capacity',group=group,utc=now()));folder=OUT/f'support{group}';columns=np.load(folder/'cache.npz')['columns'];valid=np.isin(key(columns),truthkeys,assume_unique=True);hist=histogram(columns[valid]);assert(hist<=truehist).all();cap=hist[eligible];np.savez_compressed(folder/'capacity.npz',eligible_keys=eligible,compatible_valid_zero_F=cap,valid_state_mask=valid);np.savez_compressed(folder/'key-histogram.npz',keys=np.flatnonzero(hist),counts=hist[hist>0]);m=np.load(OUT/f'b{comp["representative_bank"]}/d{comp["representative_dose"]}/model.npz');i,j=np.triu_indices(225,1);actual=m['F'][i,j].astype(bool);truth=Ftrue[i,j].astype(bool);results.append(dict(group=group,complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truecols),valid_state_retention_pct=100*float(valid.sum()/len(truecols)),false_forbidden_edges=int((actual&~truth).sum()),missed_true_edges=int((~actual&truth).sum()),eligible_inputs=len(eligible),inputs_with_valid_zero_F_completion=int((cap>0).sum()),capacity_pct=100*float((cap>0).mean()),per_clue={str(k):dict(eligible_inputs=int((clues==k).sum()),inputs_with_valid_zero_F_completion=int(((clues==k)&(cap>0)).sum()),capacity_pct=100*float((cap[clues==k]>0).mean()))for k in [1,2,3]}));print('CAPACITY',group,results[-1]['capacity_pct'],flush=True)
        for r in repair['rows']:
            info=results[r['support_group']];base=baseline['rows'][r['bank']];rows.append(dict(bank=r['bank'],dose=r['dose'],method='BP AdamW positive-witness deletion',primary=r['dose']==64,decision='supported'if info['capacity_pct']>=95 and info['invalid_zero_F_states']==0 else'falsified',support_group=r['support_group'],removed_finite_offsets=len(r['removed_finite_offsets']),removed_target_pairs=r['removed_target_pairs'],baseline_H3aj_capacity_pct=base['capacity_pct'],capacity_gain_pp=info['capacity_pct']-base['capacity_pct'],**info))
        for bank in range(4):
            family=sorted([r for r in rows if r['bank']==bank],key=lambda r:r['dose']);assert all(a['capacity_pct']<=b['capacity_pct']and a['valid_zero_F_states']<=b['valid_zero_F_states']for a,b in zip(family[:-1],family[1:]));assert family[0]['capacity_pct']>=baseline['rows'][bank]['capacity_pct']
        primary=[r for r in rows if r['primary']];decision='supported'if all(r['decision']=='supported'for r in primary)else'falsified';save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,rows=rows,primary=primary,support_results=results,repair_sha256=sha(OUT/'repair-seal.json'),source_positive_seal_sha256=sha(OUT/'source-positive-seal.json'),compiled_supports=compiled,compiler_validation=validation,reference_true_states=len(truecols),same_seen_eligible_inputs=len(eligible),seen_universe_sha256=sha(ref/'input-universe.npz'),stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(e),completed_rows=rows,repair_sha256=sha(OUT/'repair-seal.json')if(OUT/'repair-seal.json').exists()else None,compiled_supports=compiled,compiler_validation=validation,stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,partial_universe_usable=False,protocol_sha256=sha(SRC/'protocol.json')));raise
if __name__=='__main__':main()
