"""Complete state-set separation and exact pairwise positive-label distinguishability."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools
import math
import resource
import signal
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();rows=[]
    def expired(*_):raise TimeoutError('H3ay120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3ay_rule_helpers');ordinaryhelper=load(SRC.parent/'h3ad/generate.py','h3ay_ordinary');census=load(SRC.parent/'h3ao/evaluate.py','h3ay_census');binary=OUT/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True);populations={};graphs={}
        def enumerate_F(F,n,label):
            inp=OUT/f'{label}-F.bin';out=OUT/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        for n in (12,13):
            ordinary=ordinaryhelper.queens(n);assert len(ordinary)==(14200 if n==12 else 73712);assert np.array_equal(ordinary,enumerate_F(rule.rule_graph(n,[]),n,f'ordinary{n}'));cells=n*np.arange(n)[None,:]+ordinary.astype(np.int64);a,b=np.triu_indices(n,1)
            for cfg in p['families']:
                name=cfg['name'];F=rule.rule_graph(n,[tuple(o) for o in cfg['orbits']]);columns=ordinary[F[cells[:,a],cells[:,b]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,enumerate_F(F,n,f'{name}N{n}'));folder=OUT/name;folder.mkdir(exist_ok=True);np.save(folder/f'N{n}-columns.npy',columns);np.save(folder/f'N{n}-F.npy',F);populations[name,n]=columns;graphs[name,n]=F
                if n==13:
                    h=census.hist(columns);keys=np.flatnonzero(h);np.savez_compressed(folder/'N13-clue-census.npz',keys=keys,counts=h[keys]);print('N13_POPULATION',name,len(columns),len(keys),flush=True)
        A12=populations['A',12];B12=populations['B',12];assert np.array_equal(A12,B12) and len(A12)==120;np.save(OUT/'N12-shared-columns.npy',A12)
        A=populations['A',13];B=populations['B',13];As=set(map(bytes,A));Bs=set(map(bytes,B));intersection=np.array([row for row in A if bytes(row) in Bs],np.uint8).reshape(-1,13);np.save(OUT/'N13-intersection.npy',intersection);difference={name:np.array([row for row in pop if bytes(row) not in other],np.uint8).reshape(-1,13) for name,pop,other in [('A',A,Bs),('B',B,As)]};witnesses=[]
        for name,other in [('A','B'),('B','A')]:
            pop=populations[name,13];delta=difference[name];np.save(OUT/f'N13-{name}-only.npy',delta);M=len(pop);I=len(intersection);eligible=M>=16;total=math.comb(M,16) if eligible else None;fail=math.comb(I,16) if eligible else None;success=total-fail if eligible else None;passed=eligible and len(delta)>0 and 20*success>=19*total
            row=dict(true_family=name,other_family=other,N13_population=M,intersection_states=I,directed_difference_states=len(delta),directed_difference_nonempty=len(delta)>0,dose16_eligible=eligible,total_subsets=str(total) if eligible else None,indistinguishable_subsets=str(fail) if eligible else None,distinguishing_subsets=str(success) if eligible else None,distinguishing_probability=success/total if eligible else None,distinguishing_pct=100*success/total if eligible else None,probability_at_least95=bool(eligible and 20*success>=19*total),decision='supported' if passed else 'falsified' if eligible else 'ineligible_population_fewer_than16',population_sha256=sha(OUT/name/'N13-columns.npy'),difference_sha256=sha(OUT/f'N13-{name}-only.npy'));rows.append(row)
            if len(delta):
                board=delta[0];cells=13*np.arange(13)+board;violations=[dict(row_a=int(a),col_a=int(board[a]),row_b=int(b),col_b=int(board[b]),absolute_displacement=[b-a,abs(int(board[b])-int(board[a]))]) for a,b in itertools.combinations(range(13),2) if graphs[other,13][cells[a],cells[b]]];assert violations and not any(graphs[name,13][cells[a],cells[b]] for a,b in itertools.combinations(range(13),2));witnesses.append(dict(valid_under=name,invalid_under=other,columns_zero_based=board.tolist(),columns_one_based=(board+1).tolist(),selection='Lexicographically first full board in the directed set difference.',all_violating_occupied_pairs_under_other=violations))
            print('DIRECTION',name,row['directed_difference_states'],row['distinguishing_pct'],row['decision'],flush=True)
        save(OUT/'distinguishing-witnesses.json',dict(witnesses=witnesses,none_if_empty_directed_difference=True));decision='inconclusive_ineligible_population' if not all(r['dose16_eligible'] for r in rows) else 'supported' if all(r['decision']=='supported' for r in rows) else 'falsified';save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,rows=rows,N12_populations_identical=True,N12_shared_states=120,N12_shared_sha256=sha(OUT/'N12-shared-columns.npy'),N13_intersection_states=len(intersection),N13_symmetric_difference_states=sum(len(d) for d in difference.values()),N13_intersection_sha256=sha(OUT/'N13-intersection.npy'),witnesses_sha256=sha(OUT/'distinguishing-witnesses.json'),clue_censuses=[dict(family=name,states=len(populations[name,13]),distinct_keys=len(np.load(OUT/name/'N13-clue-census.npz')['keys']),total_occurrences=377*len(populations[name,13]),sha256=sha(OUT/name/'N13-clue-census.npz')) for name in ('A','B')],no_new_observed_labels=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_rows=rows,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
