"""Exact conditional hypothetical acquisition reliability with full target capacity."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import math
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
from distribution import mobius,grouped_dp,validate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();start=cpu();wall=time.perf_counter();rows=[];infos=[]
    def expired(*_):raise TimeoutError('H3av120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        validation=validate();subprocess.run([sys.executable,str(SRC/'source.py')],check=True);selection=read(OUT/'source-selection.json');frozen(True);helper=load(SRC.parent/'h3ao/evaluate.py','h3av_target_helpers');rule=load(SRC.parent/'h3at/source.py','h3av_incidence_helpers');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];ordinary=load(SRC.parent/'h3ad/generate.py','h3av_ordinary').queens(13);assert len(ordinary)==73712;binary=OUT/'reverse'
        def enumerate_F(F,n,label):
            inp=OUT/f'{label}-F.bin';out=OUT/f'{label}-cols.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        queenF=helper.graph(13);assert np.array_equal(ordinary,enumerate_F(queenF,13,'ordinary'));references={};populations={}
        for family,cfg in enumerate(p['families']):
            population=np.load(ROOT/cfg['source_path']);assert np.array_equal(population,enumerate_F(rule.rule_graph(12,[tuple(cfg['leaper'])]),12,f'f{family}source'));populations[family]=population;Ftrue=helper.graph(13,tuple(cfg['leaper']));truth=np.load(ROOT/cfg['target_path']);assert np.array_equal(truth,enumerate_F(Ftrue,13,f'f{family}truth'));h=helper.hist(truth);keys=np.flatnonzero(h);folder=OUT/f'family{family}';folder.mkdir(exist_ok=True);np.save(folder/'source-population.npy',population);np.save(folder/'true-columns.npy',truth);np.save(folder/'true-F.npy',Ftrue);np.savez_compressed(folder/'input-universe.npz',keys=keys,counts=h[keys]);references[family]=(truth,Ftrue,keys,set(map(bytes,truth)))
        cells=13*np.arange(13)[None,:]+ordinary.astype(np.int64);ii,jj=np.triu_indices(13,1);ai,aj=np.triu_indices(169,1)
        for record in selection['supports']:
            group=record['group'];folder=OUT/f'support{group}';F=np.load(folder/'model.npz')['F'];assert (F>=queenF).all();truth,Ftrue,keys,truekeys=references[record['family']];columns=ordinary[F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,enumerate_F(F,13,f'support{group}'));valid=np.array([bytes(row) in truekeys for row in columns]);flags=helper.hist(columns[valid])[keys]>0;np.savez_compressed(folder/'capacity.npz',columns=columns,valid_state_mask=valid,key_capacity_bits=np.packbits(flags,bitorder='little'));info=dict(group=group,family=record['family'],complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(flags.sum()),capacity_pct=100*float(flags.mean()),meets_95_percent=20*int(flags.sum())>=19*len(keys),exact_graph=bool(np.array_equal(F,Ftrue)),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),capacity_sha256=sha(folder/'capacity.npz'));infos.append(info)
        for bank in selection['banks']:
            index=bank['index'];family=bank['family'];folder=OUT/f'b{index}';initial=np.load(folder/'initial-evidence.npz')['columns'];seen=set(map(bytes,initial));population=populations[family];assert seen<=set(map(bytes,population));complement=np.array([row for row in population if bytes(row) not in seen],np.uint8);assert len(complement)==len(population)-16;orbitsub=[orbits[j] for j in bank['unwitnessed_orbit_indices']];hit=rule.incidence(complement,orbitsub);masks=(hit.astype(np.int64)*(1<<np.arange(len(orbitsub)))).sum(1);frequency=np.bincount(masks,minlength=1<<len(orbitsub));mass=mobius(frequency.tolist(),16);dp,operations=grouped_dp(frequency.tolist(),16);assert mass==[dp[16].get(mask,0) for mask in range(len(mass))];total=math.comb(len(complement),16);outcomes=[];success=0;exact=0;uniqueexact=0
            for pattern in bank['patterns']:
                mask=pattern['mask'];members=[infos[r['group']] for r in pattern['maximizers']];safe_capacity=all(r['invalid_zero_F_states']==0 and r['meets_95_percent'] for r in members);exactgraph=all(r['exact_graph'] for r in members);weight=mass[mask]
                if weight:assert not pattern['ineligible32_candidates'],'Positive mass cannot contain32 distinct boards in a smaller candidate'
                success+=weight*safe_capacity;exact+=weight*exactgraph;uniqueexact+=weight*(exactgraph and pattern['unique']);outcomes.append(dict(mask=mask,subset_count=str(weight),probability_numerator=str(weight),probability_denominator=str(total),primary_success=safe_capacity,all_maximizers_exact_graph=exactgraph,unique=pattern['unique'],selected_candidates=[r['candidate'] for r in pattern['maximizers']],selected_groups=[r['group'] for r in pattern['maximizers']]))
            np.savez_compressed(folder/'acquisition-evidence.npz',complement_columns=complement,board_masks=masks,frequency=frequency);save(folder/'exact-distribution.json',dict(index=index,family=family,initial_labels=16,hypothetical_additional_labels=16,complement_size=len(complement),total_subsets=str(total),outcomes=outcomes,all_masks_retained_including_zero_mass=True,mobius_and_grouped_DP_all_masses_identical=True,grouped_DP_operations=operations,acquisition_evidence_sha256=sha(folder/'acquisition-evidence.npz')));passed=20*success>=19*total;gcd=math.gcd(success,total);row=dict(index=index,family=family,bank=bank['bank'],complement_size=len(complement),initial_labels=16,hypothetical_additional_labels=16,new_observed_labels=0,total_subsets=str(total),successful_subsets=str(success),reduced_success_numerator=str(success//gcd),reduced_success_denominator=str(total//gcd),success_probability=success/total,success_pct=100*success/total,all_maximizers_exact_graph_subsets=str(exact),all_maximizers_exact_graph_probability=exact/total,unique_exact_graph_subsets=str(uniqueexact),unique_exact_graph_probability=uniqueexact/total,abstract_masks=len(mass),positive_mass_masks=sum(value>0 for value in mass),decision='supported' if passed else 'falsified',distribution_sha256=sha(folder/'exact-distribution.json'));rows.append(row);print('BANK_RELIABILITY',index,row['success_pct'],row['decision'],row['unique_exact_graph_probability'],flush=True)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported' if all(r['decision']=='supported' for r in rows) else 'falsified',rows=rows,support_results=infos,validation=validation,source_selection_sha256=sha(OUT/'source-selection.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_rows=rows,completed_supports=infos,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
