"""Descriptive complete-state check after the unchanged H3y structural result."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import datetime as dt
import itertools
from pathlib import Path
import signal
import socket
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,save,read,bit_enumeration,line_enumeration
PUBLIC=ROOT/'results/research/loki/h3y'

def freeze():
    assert not(SRC/'census-protocol.json').exists()
    summary=read(PUBLIC/'summary.json');case=summary['case_diagnostics']['f4-n10'];assert summary['decision']=='falsified'and len(case['all_minima'])==1 and case['all_minima'][0]['missed_pairs']==288
    save(SRC/'census-protocol.json',dict(id='H3y-descriptive-N11-complete-state-census',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='descriptive_followup_after_structural_outcome',case='f4-n10',n=11,
        question='Does the unique source minimum with288missed N11attack pairs admit any invalid complete q11zero-conflict state,or is its discrepancy only at graph level?',
        method='Enumerate the entire N11selected-rule and true-rule solution sets with constrained attack-bitmask recursion. Independently enumerate ordinary queens through line-coordinate-set recursion and literally pair-filter every board under both rules. Compare full sorted sets. Save every selected-but-invalid board and its exact selected/true conflict counts.',
        decision='Any selected-but-invalid complete board disproves soundness of the selected zero-conflict certificate atq11. If none,report graph mismatch but no full-state counterexample atq11. This is not a search-success-rate estimate.',
        cpu_seconds=30,wall_seconds=30,gpu_search_jobs=0,changes_to_H3y_primary='none',
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in [Path(__file__),SRC/'common.py']},
        structural_summary_sha256=sha(PUBLIC/'summary.json'),structural_protocol_sha256=sha(SRC/'protocol.json')))
    print('CENSUS_FROZEN',sha(SRC/'census-protocol.json'),flush=True)

def run():
    assert socket.gethostname()=='loki';assert not(PUBLIC/'q11-census.json').exists();p=read(SRC/'census-protocol.json')
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h
    assert sha(PUBLIC/'summary.json')==p['structural_summary_sha256'];s=read(PUBLIC/'summary.json');d=s['case_diagnostics'][p['case']]
    start=time.perf_counter();cpu=time.process_time()
    def expired(*args):raise TimeoutError('Frozen30second descriptive census budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,30)
    try:
        n=11;source_lines=[tuple(v)for v in d['source_result']['inferred_lines']];selected=[tuple(v)for v in d['all_minima'][0]['generators']]
        truth=[tuple(v)for v in s['families']['f4']];queen_lines=[(0,1),(1,-1),(1,0),(1,1)]
        chosen_boards=bit_enumeration(n,source_lines,selected);true_boards=bit_enumeration(n,queen_lines,truth)
        ordinary=line_enumeration(n,queen_lines)
        def conflicts(cols,lines,leapers):
            return sum(any(a*(t-r)+b*(int(cols[t])-int(cols[r]))==0 for a,b in lines)or tuple(sorted((t-r,abs(int(cols[t])-int(cols[r])))))in leapers for r,t in itertools.combinations(range(n),2))
        selected_counts=np.array([conflicts(c,source_lines,selected)for c in ordinary]);true_counts=np.array([conflicts(c,queen_lines,truth)for c in ordinary])
        assert np.array_equal(chosen_boards,ordinary[selected_counts==0]);assert np.array_equal(true_boards,ordinary[true_counts==0])
        bad=(selected_counts==0)&(true_counts>0);witnesses=ordinary[bad];assert all(conflicts(c,source_lines,selected)==0 and conflicts(c,queen_lines,truth)>0 for c in witnesses)
        np.savez_compressed(PUBLIC/'q11-census-boards.npz',ordinary_columns=ordinary,selected_columns=chosen_boards,true_columns=true_boards,
                            invalid_selected_columns=witnesses,ordinary_selected_conflicts=selected_counts,ordinary_true_conflicts=true_counts)
        elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu;assert elapsed<30 and cpu_elapsed<30
        result=dict(status='validated',analysis='descriptive_after_frozen_H3y_result',case=p['case'],n=n,
            ordinary_queen_boards=len(ordinary),selected_rule_complete_boards=len(chosen_boards),true_rule_complete_boards=len(true_boards),
            selected_but_invalid_complete_boards=len(witnesses),true_but_selected_excluded_boards=int(((true_counts==0)&(selected_counts>0)).sum()),
            selected_generator_orbits=selected,true_generator_orbits=truth,first_invalid_columns=witnesses[0].astype(int).tolist()if len(witnesses)else None,
            first_invalid_selected_conflicts=0 if len(witnesses)else None,first_invalid_true_conflicts=int(true_counts[bad][0])if len(witnesses)else None,
            every_invalid_witness_verified=True,both_complete_sets_independently_equal=True,zero_conflict_certificate_unsound=len(witnesses)>0,
            wall_seconds=elapsed,cpu_seconds=cpu_elapsed,gpu_search_jobs=0,original_H3y_primary_unchanged=True,
            census_protocol_sha256=sha(SRC/'census-protocol.json'),structural_summary_sha256=p['structural_summary_sha256'],
            exported_boards_sha256=sha(PUBLIC/'q11-census-boards.npz'))
        save(PUBLIC/'q11-census.json',result);print(result,flush=True)
    except BaseException as e:save(PUBLIC/'q11-census-failure.json',dict(status='incomplete',error=repr(e),seconds=time.perf_counter()-start));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','run']);args=a.parse_args()
    if args.mode=='freeze':freeze()
    else:run()
