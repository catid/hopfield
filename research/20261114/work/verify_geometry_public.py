"""Independent public H2bd geometry, sample selection, graph and timing checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_large_cycle_geometry import geometry
from verify_hard_negative_feasibility import valid_shortest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();fixtures=directory/'public-fixtures';manifest=json.loads((fixtures/'manifest.json').read_text());assert all(sha(fixtures/k)==h for k,h in manifest['files'].items());d=fixtures/'geometry';m=json.loads((d/'manifest.json').read_text());expected=json.loads((d/'expected-geometry.json').read_text());all_hashes=json.loads((d/'all-unique-state-hashes.json').read_text())['state_hashes'];assert len(all_hashes)==12220 and all_hashes==sorted(set(all_hashes));keys=[];index=0;candidate_count=0
    for row in m['fixtures']:
        path=d/row['file'];assert sha(path)==row['sha256']
        with np.load(path) as z:
            for key,t,b in zip(row['state_hashes'],z['tokens'],z['boards']):
                digest=hashlib.sha256(b'H2bd-geometry-v1'+np.ascontiguousarray(t,dtype=np.uint8).tobytes()+np.ascontiguousarray(b,dtype=np.uint8).tobytes()).hexdigest();assert digest==key==expected[index]['state_hash'];cells,stats=geometry(t,b,256);assert [list(c) for c in cells]==expected[index]['candidates'] and stats==expected[index]['geometry'];keys.append(key);index+=1;candidate_count+=len(cells)
    assert index==1024 and keys==all_hashes[:1024];source=json.loads((fixtures/'source-cohort.json').read_text());assert sha(fixtures/'source-cohort.npz')==source['cohort_npz_sha256']
    with np.load(fixtures/'source-cohort.npz') as z:tokens=z['tokens'];labels=z['labels'];reference={k:z[k] for k in ['predictions','iterations','stopped','finite','native_cap','block_cap']}
    for i in range(256):
        assert valid_shortest(tokens[i],labels[i])
        for seed in range(3):assert valid_shortest(tokens[i],reference['predictions'][seed,i])
    a=json.loads((directory/'audit.json').read_text());ratios=[];states=0
    for row in a['policy']['rows']:
        ratio=row['wall_seconds']['legacy']/row['wall_seconds']['fast'];assert abs(ratio-row['paired_wall_speedup'])<1e-12;ratios.append(ratio)
        for receipt in row['receipts'].values():
            path=directory/receipt['outputs_file'];assert sha(path)==receipt['outputs_sha256'] and receipt['trace_hashes']==source['trace_hashes']
            with np.load(path) as z:
                for key,expected_array in reference.items():assert np.array_equal(z[key],expected_array)
            states+=768
    median=float(np.median(ratios));assert median==a['policy']['median_paired_wall_speedup'] and a['policy']['decision']==('supported' if median>=1.1 else 'falsified');micro=a['geometry'];ratios2=[r['timing']['legacy']['process_seconds']/r['timing']['fast']['process_seconds'] for r in micro['rows']];assert all(abs(x-r['paired_process_speedup'])<1e-12 for x,r in zip(ratios2,micro['rows']));assert float(np.median(ratios2))==micro['median_paired_process_speedup'] and micro['decision']==('supported' if np.median(ratios2)>=2 else 'falsified')
    return {'status':'passed','geometry_states':1024,'candidate_tuples':candidate_count,'complete_hash_selection_verified':True,'unique_hash_universe':12220,'source_shortest_labels':256,'source_model_boards':768,'timing_model_boards_compared':states,'all_trace_hashes_match_source':True,'geometry_median':micro['median_paired_process_speedup'],'policy_median':median,'manifest_sha256':sha(fixtures/'manifest.json'),'audit_sha256':sha(directory/'audit.json'),'source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started,'scope':'Independent NumPy geometry and shortest paths, fixed hash selection, all6144 repeated final boards/counters/flags and trace hashes, separate timing arithmetic. No Torch or production decoders; no new accuracy claim.'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);print(json.dumps(verify(a.parse_args().directory),indent=2))
