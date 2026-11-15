"""Paired exact reliability at32 raw positives: unchanged repair versus D4 views."""
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
from common import ROOT,SRC,OUT,frozen,read,save,sha,now


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def keys_for(columns):
    covered=np.zeros(226**3,bool)
    cells=15*np.arange(15)[None,:]+columns.astype(np.int64)
    for k in (1,2,3):
        indices=np.array(list(itertools.combinations(range(15),k)))
        powers=np.array([226**(2-j)for j in range(k)],np.int64)
        for lo in range(0,len(columns),2048):
            keys=((cells[lo:lo+2048,indices]+1)*powers).sum(-1)
            covered[keys.reshape(-1)]=True
    return covered


def main():
    p=frozen();assert not(OUT/'study-result.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    wall=time.perf_counter();start_cpu=cpu();support_results=[];rows=[]
    def expired(*args):raise TimeoutError('H3am120CPU-second/180wall-second gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        save(OUT/'progress.json',dict(stage='source_D4_validation_and_paired_exact_distribution',utc=now()))
        subprocess.run([sys.executable,str(SRC/'source_outcomes.py')],check=True)
        source=read(OUT/'source-outcomes.json');assert source['status']=='complete_source_only'
        frozen('target');ref=ROOT/p['evaluation']['reference_root'];Ftrue=np.load(ref/'truth/F.npy');columns=np.load(ref/'truth/cache.npz')['columns'];u=np.load(ref/'input-universe.npz');keys=u['eligible_keys'];clues=u['eligible_clues'];assert len(columns)==26960 and len(keys)==831605
        assert np.array_equal(np.flatnonzero(keys_for(columns)),u['all_keys'])
        full={r['group']:np.load(OUT/f"support{r['group']}/F.npz")['F']for r in source['supports']}
        assert all((F>=Ftrue).all()for F in full.values()),'Completeness gate failed: no outcome may be filtered'
        save(OUT/'filtering-completeness-seal.json',dict(status='passed',utc=now(),all_repaired_F_contain_true_F=True,supports=len(full),reference_cache_sha256=sha(ref/'truth/cache.npz'),reference_universe_sha256=sha(ref/'input-universe.npz')))
        cells=15*np.arange(15)[None,:]+columns.astype(np.int64);ii,jj=np.triu_indices(15,1);ai,aj=np.triu_indices(225,1)
        for group,F in full.items():
            save(OUT/'progress.json',dict(stage='complete_capacity',group=group,utc=now()))
            valid=F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0;flags=keys_for(columns[valid])[keys];covered=int(flags.sum());folder=OUT/f'support{group}'
            np.savez_compressed(folder/'capacity.npz',valid_state_bits=np.packbits(valid,bitorder='little'),eligible_capacity_bits=np.packbits(flags,bitorder='little'))
            support_results.append(dict(group=group,valid_zero_F_states=int(valid.sum()),covered_eligible_inputs=covered,capacity_pct=100*float(flags.mean()),state_retention_pct=100*float(valid.mean()),meets_95_percent=20*covered>=19*len(keys),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=0,F_contains_true_F=True,capacity_sha256=sha(folder/'capacity.npz'),per_clue={str(k):dict(eligible_inputs=int((clues==k).sum()),covered_eligible_inputs=int(flags[clues==k].sum()))for k in (1,2,3)}))
        previous=read(ROOT/p['evaluation']['unaugmented_reference_summary'])
        for b in source['banks']:
            m=b['class_count'];mapping=b['mask_to_support'];mass=list(map(int,b['draw_masses']));den=math.comb(1932,32);assert sum(mass)==int(b['total_draws'])==den
            categories={key:0 for key in ('neither','raw_only','augmented_only','both')};raw_expected=aug_expected=raw_exact=aug_exact=0
            for pair,weight in enumerate(mass):
                rawmask=pair&((1<<m)-1);augmask=pair>>m;raw=support_results[mapping[rawmask]];aug=support_results[mapping[augmask]]
                if weight:
                    assert (rawmask|augmask)==augmask and (full[mapping[augmask]]<=full[mapping[rawmask]]).all()
                    assert aug['covered_eligible_inputs']>=raw['covered_eligible_inputs']
                category=('both'if aug['meets_95_percent']else'raw_only')if raw['meets_95_percent']else('augmented_only'if aug['meets_95_percent']else'neither')
                categories[category]+=weight;raw_expected+=weight*raw['covered_eligible_inputs'];aug_expected+=weight*aug['covered_eligible_inputs'];raw_exact+=weight*(raw['false_forbidden_edges']==0);aug_exact+=weight*(aug['false_forbidden_edges']==0)
            assert categories['raw_only']==0
            rawgood=categories['both']+categories['raw_only'];auggood=categories['both']+categories['augmented_only'];gain=auggood-rawgood;rawprob=Fraction(rawgood,den);augprob=Fraction(auggood,den);gainprob=Fraction(gain,den)
            prior=next(r for r in previous['rows']if r['bank']==b['bank']and r['dose']==32)
            assert rawprob==Fraction(int(prior['success_probability_numerator']),int(prior['success_probability_denominator']))
            row=dict(bank=b['bank'],method='BP AdamW fixed fit; paired source-only D4 repair',raw_positive_labels=32,raw_attempted_views=32,augmented_attempted_views=256,all_paired_masks=len(mass),positive_mass_pairs=sum(x>0 for x in mass),raw_success_probability_pct=100*float(rawprob),augmented_success_probability_pct=100*float(augprob),gain_percentage_points=100*float(gainprob),raw_favorable_draws=str(rawgood),augmented_favorable_draws=str(auggood),gain_draws=str(gain),total_draws=str(den),paired_categories={k:str(v)for k,v in categories.items()},raw_probability=dict(numerator=str(rawprob.numerator),denominator=str(rawprob.denominator)),augmented_probability=dict(numerator=str(augprob.numerator),denominator=str(augprob.denominator)),gain_probability=dict(numerator=str(gainprob.numerator),denominator=str(gainprob.denominator)),augmented_reliability_at_least90_percent=10*auggood>=9*den,gain_at_least10_percentage_points=10*gain>=den,raw_expected_capacity_pct=100*float(Fraction(raw_expected,den*len(keys))),augmented_expected_capacity_pct=100*float(Fraction(aug_expected,den*len(keys))),raw_exact_graph_probability_pct=100*float(Fraction(raw_exact,den)),augmented_exact_graph_probability_pct=100*float(Fraction(aug_exact,den)),unaugmented_probability_exactly_matches_H3al=True)
            row['decision']='supported'if row['augmented_reliability_at_least90_percent']and row['gain_at_least10_percentage_points']else'falsified';rows.append(row);print('BANK',b['bank'],row['raw_success_probability_pct'],row['augmented_success_probability_pct'],row['gain_percentage_points'],row['decision'],flush=True)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in rows)else'falsified',rows=rows,support_results=support_results,total_paired_bank_masks=source['total_paired_bank_masks'],distinct_full_supports=len(full),source_outcomes_sha256=sha(OUT/'source-outcomes.json'),stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_support_results=support_results,completed_rows=rows,stage_cpu_seconds=cpu()-start_cpu,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),partial_results_usable=False));raise


if __name__=='__main__':main()
