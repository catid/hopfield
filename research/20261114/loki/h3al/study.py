"""Exact reliability evaluation of the fixed positive-witness repair algorithm."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from fractions import Fraction
import itertools
import math
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

def exact_key_union(columns):
    covered=np.zeros(226**3,bool);pairs=0
    for k in [1,2,3]:
        ix=np.array(list(itertools.combinations(range(15),k)));powers=np.array([226**(2-j)for j in range(k)],np.int64)
        for lo in range(0,len(columns),2048):
            cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];keys=((cells[:,ix]+1)*powers).sum(-1);covered[keys.reshape(-1)]=True;pairs+=keys.size
            if lo==0:
                literal=np.array([sum((int(c)+1)*226**(2-j)for j,c in enumerate(subset))for row in cells[:17]for subset in itertools.combinations(row,k)]);assert np.array_equal(literal,keys[:17].reshape(-1))
    assert pairs==len(columns)*575;return covered

def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start_cpu=cpu();support_results=[];rows=[];source=None
    def expired(*args):raise TimeoutError('H3al120aggregateCPU-second/180wall-second exact probability gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        save(OUT/'progress.json',dict(stage='exact_source_union_distribution',utc=now()));subprocess.run([sys.executable,str(SRC/'source_outcomes.py')],check=True);source=read(OUT/'source-outcomes.json');assert source['status']=='complete_source_only'and source['class_complexity_gate_passed'];frozen('target');ref=ROOT/p['evaluation']['reference_root'];Ftrue=np.load(ref/'truth/F.npy');columns=np.load(ref/'truth/cache.npz')['columns'];u=np.load(ref/'input-universe.npz');keys=u['eligible_keys'];clues=u['eligible_clues'];assert len(columns)==26960 and len(keys)==831605;truecovered=exact_key_union(columns);assert np.array_equal(np.flatnonzero(truecovered),u['all_keys'])and truecovered[keys].all();i,j=np.triu_indices(225,1)
        for a,b in zip(i,j):
            ar,ac=divmod(int(a),15);br,bc=divmod(int(b),15);actual=(ar==br or ac==bc or abs(ar-br)==abs(ac-bc)or(abs(ar-br),abs(ac-bc))in[(1,3),(3,1)]);assert Ftrue[a,b]==Ftrue[b,a]==actual
        cells=15*np.arange(15)[None,:]+columns;ii,jj=np.triu_indices(15,1)
        # Verify every repaired support is a true-graph superset before filtering any outcome.
        for support in source['supports']:
            F=np.load(OUT/f'support{support["group"]}/F.npz')['F'];assert(F>=Ftrue).all(),'Filtering would not be complete: repaired support misses a true constraint'
        save(OUT/'filtering-completeness-seal.json',dict(status='passed',utc=now(),all_repaired_F_contain_true_F=True,supports=len(source['supports']),proof='Every zero-F repaired cardinality15 board obeys every true constraint; the supplied complete true universe therefore contains every repaired zero state. Filtering this universe by exact F counts is complete.',reference_cache_sha256=sha(ref/'truth/cache.npz'),reference_universe_sha256=sha(ref/'input-universe.npz')))
        for support in source['supports']:
            group=support['group'];save(OUT/'progress.json',dict(stage='evaluate_fixed_support',group=group,utc=now()));folder=OUT/f'support{group}';F=np.load(folder/'F.npz')['F'];valid=F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0;covered=exact_key_union(columns[valid]);flags=covered[keys];num=int(flags.sum());np.savez_compressed(folder/'capacity.npz',valid_state_bits=np.packbits(valid,bitorder='little'),eligible_capacity_bits=np.packbits(flags,bitorder='little'));info=dict(group=group,valid_zero_F_states=int(valid.sum()),true_states=len(columns),valid_state_retention_pct=100*float(valid.mean()),eligible_inputs=len(keys),covered_eligible_inputs=num,capacity_pct=100*float(flags.mean()),meets_95_percent=(20*num>=19*len(keys)),false_forbidden_edges=int(((F[i,j]>0)&(Ftrue[i,j]==0)).sum()),missed_true_edges=0,F_contains_true_F=True,capacity_sha256=sha(folder/'capacity.npz'),per_clue={str(k):dict(eligible_inputs=int((clues==k).sum()),covered_eligible_inputs=int(flags[clues==k].sum()))for k in [1,2,3]});support_results.append(info);print('OUTCOME',group,num,info['capacity_pct'],flush=True)
        for bank in source['banks']:
            mapping=bank['mask_to_support']
            for dose in p['doses']:
                mass=list(map(int,bank['draw_masses'][str(dose)]));den=int(bank['total_masses'][str(dose)]);assert len(mass)==len(mapping)and sum(mass)==den==math.comb(1932,dose);good=sum(weight for mask,weight in enumerate(mass)if support_results[mapping[mask]]['meets_95_percent']);expected=sum(weight*support_results[mapping[mask]]['covered_eligible_inputs']for mask,weight in enumerate(mass));fullgraph=sum(weight for mask,weight in enumerate(mass)if support_results[mapping[mask]]['false_forbidden_edges']==0);prob=Fraction(good,den);rows.append(dict(bank=bank['bank'],method='BP AdamW fixed positive-witness repair',dose=dose,primary=dose==64,decision=('supported'if 10*good>=9*den else'falsified')if dose==64 else'descriptive',class_count=bank['class_count'],all_union_masks=len(mapping),positive_mass_masks=sum(weight>0 for weight in mass),favorable_draw_subsets=str(good),total_draw_subsets=str(den),success_probability_numerator=str(prob.numerator),success_probability_denominator=str(prob.denominator),success_probability_pct=100*float(prob),expected_capacity_pct=100*float(Fraction(expected,den*len(keys))),exact_graph_probability_pct=100*float(Fraction(fullgraph,den)),unchanged_support_probability_pct=100*float(Fraction(mass[0],den))))
        primary=[r for r in rows if r['primary']];joint=Fraction(1)
        for r in primary:joint*=Fraction(int(r['success_probability_numerator']),int(r['success_probability_denominator']))
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in primary)else'falsified',rows=rows,primary=primary,support_results=support_results,source_outcomes_sha256=sha(OUT/'source-outcomes.json'),unique_full_supports=len(support_results),total_bank_masks=source['total_bank_masks'],all_integer_masses_nonnegative_and_normalized=True,independent_stream_joint=dict(numerator=str(joint.numerator),denominator=str(joint.denominator),probability_pct=100*float(joint),interpretation='Descriptive product only for independent acquisition streams conditional on these fixed models and finite source population.'),stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(e),completed_support_results=support_results,completed_rows=rows,source_outcomes_sha256=sha(OUT/'source-outcomes.json')if(OUT/'source-outcomes.json').exists()else None,stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),partial_results_usable=False));raise
if __name__=='__main__':main()
