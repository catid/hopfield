"""Validate all N-Queens test solution counts with one shared enumeration."""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
repo = ROOT / 'upstream'
sys.path[:0]=[str(repo),str(repo/'experiments')]
from cram.nqueens_data import load_split,completions
out=(ROOT / 'artifacts/multi_solution')
out.mkdir(parents=True, exist_ok=True)
records=[]
for n in (8,10):
 inputs,labels,stored=load_split(n,'test')
 unique,inverse=np.unique(inputs,axis=0,return_inverse=True)
 inverse=inverse.reshape(-1)
 start=time.perf_counter()
 all_boards=np.array(completions(np.zeros(n*n,dtype=bool),n),dtype=bool)
 enumeration_seconds=time.perf_counter()-start
 start=time.perf_counter()
 counts=np.array([np.count_nonzero(all_boards[:,row].all(axis=1)) for row in unique],dtype=np.int64)
 count_seconds=time.perf_counter()-start
 assert np.array_equal(counts[inverse],stored.reshape(-1))
 assert np.array_equal(np.bincount(inverse),counts)
 start=time.perf_counter()
 reference=np.array([len(completions(row,n)) for row in unique[:64]])
 reference_seconds=time.perf_counter()-start
 assert np.array_equal(counts[:64],reference)
 start=time.perf_counter()
 fast_sample=np.array([np.count_nonzero(all_boards[:,row].all(axis=1)) for row in unique[:64]])
 fast_sample_seconds=time.perf_counter()-start
 assert np.array_equal(reference,fast_sample)
 start=time.perf_counter()
 all_reference=np.array([len(completions(row,n)) for row in unique])
 reference_all_seconds=time.perf_counter()-start
 assert np.array_equal(counts,all_reference)
 np.savez_compressed(out/f'nq{n}-solution-counts.npz',unique=unique,inverse=inverse,counts=counts)
 record={'n':n,'test_rows':len(inputs),'distinct_inputs':len(unique),'all_board_solutions':len(all_boards),'all_stored_counts_match':True,'input_multiplicity_equals_solution_count':True,'enumeration_seconds':enumeration_seconds,'all_input_count_seconds':count_seconds,'reference_64_inputs_seconds':reference_seconds,'shared_64_inputs_seconds':enumeration_seconds+fast_sample_seconds,'reference_sample_size':64,'reference_vs_shared_including_enumeration_speedup':reference_seconds/(enumeration_seconds+fast_sample_seconds),'scope':'Exact count algorithm only; no model fitting, inference or hyperparameter selection.'}
 record.update(reference_all_inputs_seconds=reference_all_seconds,
               all_reference_counts_match=True,
               full_count_speedup=reference_all_seconds/(enumeration_seconds+count_seconds))
 records.append(record)
 print(json.dumps(record),flush=True)
(out/'nq-metric-count-validation.json').write_text(json.dumps(records,indent=2)+'\n')
