"""Exact finite-universe capacity of all four fixed incomplete-bank N15 transfers."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
from enumerators import eligibility,forward
from transfer import make
from verification import prepare,reverse,truth,graph,counts,support_hash

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

def seal(columns,F,folder):
    assert len(np.unique(columns,axis=0))==len(columns)and not counts(columns,F).any();cells=15*np.arange(15)[None,:]+columns;boards=np.zeros((len(columns),225),bool);boards[np.arange(len(columns))[:,None],cells]=True;index=np.packbits(boards.T,axis=1,bitorder='little');np.savez_compressed(folder/'cache.npz',columns=columns,boards=boards,index=index)

def histograms(columns):
    base=226;hist=np.zeros(base**3,np.int64);ixs={k:np.array(list(itertools.combinations(range(15),k)))for k in [1,2,3]}
    for lo in range(0,len(columns),4096):
        cells=15*np.arange(15)[None,:]+columns[lo:lo+4096];pieces=[]
        for k in [1,2,3]:
            ix=ixs[k];powers=np.array([base**(2-j)for j in range(k)],np.int64);encoded=((cells[:,ix]+1)*powers).sum(-1);pieces.append(encoded.reshape(-1))
            if lo==0:
                literal=np.array([sum((int(c)+1)*base**(2-j)for j,c in enumerate(subset))for row in cells[:17]for subset in itertools.combinations(row,k)]);assert np.array_equal(literal,encoded[:17].reshape(-1))
        hist+=np.bincount(np.concatenate(pieces),minlength=base**3)
    assert hist.sum()==len(columns)*575;return hist

def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start_cpu=cpu();records=[];models=[];supports=[];mapping=[];verification=None;seal_record=None
    def expired(*args):raise TimeoutError('H3aj120aggregateCPU-second/180wall-second full-universe gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        save(OUT/'progress.json',dict(stage='build_and_validation',utc=now()));verification=prepare(p,cpu)
        # Every fit is transferred independently; only equal full supports share enumeration.
        for bank,cfg in enumerate(p['models']):
            folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);m,receipt=make(cfg,15);proof=eligibility(m);np.savez_compressed(folder/'model.npz',**m);save(folder/'transfer.json',receipt);key=support_hash(m['F']);matches=[i for i,s in enumerate(supports)if s['sha256']==key]
            if matches:group=matches[0];assert np.array_equal(m['F'],models[supports[group]['representative_bank']]['F'])
            else:group=len(supports);supports.append(dict(group=group,representative_bank=bank,sha256=key,banks=[]))
            supports[group]['banks'].append(bank);mapping.append(group);models.append(m)
        for item in supports:
            group=item['group'];bank=item['representative_bank'];folder=OUT/f'support{group}';folder.mkdir(exist_ok=True);F=models[bank]['F'];started=time.perf_counter();save(OUT/'progress.json',dict(stage='learned_forward',group=group,utc=now()));a=forward(F,15);np.save(folder/'unverified-forward-columns.npy',a);save(OUT/'progress.json',dict(stage='learned_reverse',group=group,utc=now()));b=reverse(F,15,f'N15-support{group}');assert np.array_equal(a,b);seal(a,F,folder);item.update(states=len(a),compile_wall_seconds=time.perf_counter()-started,cache_sha256=sha(folder/'cache.npz'));save(folder/'compilation.json',dict(item,status='complete_verified',utc=now(),no_supplied_truth_input=True,protocol_sha256=sha(SRC/'protocol.json')));print('LEARNED_COMPLETE',group,len(a),item['compile_wall_seconds'],flush=True)
        folder=OUT/'truth';folder.mkdir(exist_ok=True);save(OUT/'progress.json',dict(stage='true_universe',utc=now()));started=time.perf_counter();Ftrue=graph(15);a=forward(Ftrue,15);b=reverse(Ftrue,15,'N15-truth');c=truth(15,'N15-truth');assert np.array_equal(a,b)and np.array_equal(a,c);seal(a,Ftrue,folder);np.save(folder/'F.npy',Ftrue);truth_columns=a;truth_record=dict(status='complete_independently_verified',states=len(a),wall_seconds=time.perf_counter()-started,cache_sha256=sha(folder/'cache.npz'),graph_sha256=sha(folder/'F.npy'),general_forward_compiled_literal_and_direct_geometry_identical=True)
        seal_record=dict(status='all_complete',utc=now(),supports=supports,bank_to_support=mapping,true_universe=truth_record,protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'enumeration-seal.json',seal_record)
        save(OUT/'progress.json',dict(stage='full_key_histograms',utc=now()));hist_true=histograms(truth_columns);allkeys=np.flatnonzero(hist_true);excluded=np.unique(np.concatenate([np.load(ROOT/path)['test_keys']for path in p['exclusion']['prior_N15_data_paths']]));eligible=np.setdiff1d(allkeys,excluded);assert len(excluded)==8192;clues=sum((eligible//226**power%226)>0 for power in [2,1,0]);np.savez_compressed(OUT/'input-universe.npz',all_keys=allkeys,all_true_counts=hist_true[allkeys],excluded_keys=excluded,eligible_keys=eligible,eligible_true_counts=hist_true[eligible],eligible_clues=clues);encoded_truth=truth_columns.astype(np.int64)@np.array([15**power for power in range(14,-1,-1)],np.int64);support_results=[]
        for item in supports:
            group=item['group'];bank=item['representative_bank'];F=models[bank]['F'];columns=np.load(OUT/f'support{group}/cache.npz')['columns'];encoded=columns.astype(np.int64)@np.array([15**power for power in range(14,-1,-1)],np.int64);valid=np.isin(encoded,encoded_truth,assume_unique=True);truth_counts=counts(columns,Ftrue);assert np.array_equal(valid,truth_counts==0);valid_columns=columns[valid];hist=histograms(valid_columns);assert(hist<=hist_true).all();capacity=hist[eligible];np.savez_compressed(OUT/f'support{group}/capacity.npz',eligible_keys=eligible,compatible_true_zero_F=capacity,valid_state_mask=valid);np.savez_compressed(OUT/f'support{group}/key-histogram.npz',keys=np.flatnonzero(hist),counts=hist[hist>0]);i,j=np.triu_indices(225,1);actual=F[i,j].astype(bool);gt=Ftrue[i,j].astype(bool);info=dict(group=group,banks=item['banks'],complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),complete_true_states=len(truth_columns),valid_state_retention_pct=100*float(valid.sum()/len(truth_columns)),false_forbidden_edges=int((actual&~gt).sum()),missed_true_edges=int((~actual&gt).sum()),eligible_inputs=len(eligible),inputs_with_valid_zero_F_completion=int((capacity>0).sum()),capacity_pct=100*float((capacity>0).mean()),compatible_true_completions=int(hist_true[eligible].sum()),compatible_valid_zero_F_completions=int(capacity.sum()),per_clue={str(k):dict(eligible_inputs=int((clues==k).sum()),inputs_with_valid_zero_F_completion=int(((clues==k)&(capacity>0)).sum()),capacity_pct=100*float((capacity[clues==k]>0).mean()))for k in [1,2,3]});support_results.append(info);print('CAPACITY',info,flush=True)
        for bank,cfg in enumerate(p['models']):
            info=support_results[mapping[bank]];decision='supported'if info['capacity_pct']>=95 and info['invalid_zero_F_states']==0 else'falsified';records.append(dict(bank=bank,method='BP AdamW',source_bank_seed=2026112700+bank,source_patterns=128,source_population=176,source_model_sha256=sha(ROOT/cfg['inherited_n13_model_path']),target_model_sha256=sha(OUT/f'b{bank}/model.npz'),support_group=mapping[bank],decision=decision,**info))
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in records)else'falsified',rows=records,support_results=support_results,verification=verification,enumeration_seal=seal_record,universe=dict(total_true_input_keys=len(allkeys),all_prior_N15_exclusion_keys=len(excluded),excluded_true_keys=int(np.isin(allkeys,excluded).sum()),remaining_eligible_keys=len(eligible),true_states=len(truth_columns)),stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(e),completed_rows=records,supports=supports,verification=verification,enumeration_seal=seal_record,partial_universe_usable=False,stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise
if __name__=='__main__':main()
