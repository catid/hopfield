"""Independent public H2bi graph, full replay state and timing/receipt checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_hard_negative_feasibility import valid_shortest
from audit_connectivity_screen import path_valid


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());assert a['protocol_sha256']==sha(directory/'protocol.json');fixtures=directory/'public-fixtures';manifest=json.loads((fixtures/'manifest.json').read_text());assert all(sha(fixtures/name)==h for name,h in manifest['files'].items());assert sha(fixtures/'source-cohort.npz')==manifest['source_cohort_sha256'];expected=p['source_cohort'];assert manifest['source_protocol_sha256']==expected['source_protocol_sha256'] and manifest['source_audit_sha256']==expected['source_audit_sha256']
    with np.load(fixtures/'source-cohort.npz') as z:tokens=z['tokens'];labels=z['labels'];reference={k:z[k] for k in z.files if k not in ['tokens','labels']}
    assert tokens.shape==labels.shape==(256,180,180) and reference['predictions'].shape==(3,8,256,180,180);graph=np.zeros((3,8,256),bool);cache={}
    for i in range(256):
        assert valid_shortest(tokens[i],labels[i]);shortest_cells=int(labels[i].sum())
        for seed in range(3):
            for v in range(8):
                board=reference['predictions'][seed,v,i];key=(i,np.packbits(board).tobytes())
                if key not in cache:cache[key]=path_valid(tokens[i],board) and int(board.sum())==shortest_cells
                graph[seed,v,i]=cache[key]
    success=graph&reference['finite']&reference['stopped']&(reference['minimum_flip_margin64']>=-1e-6)&reference['hidden_good'][:,None,:];joint={arm:int(success[:,v].all(0).sum()) for v,arm in enumerate(p['source_internal_arms'])};assert joint==expected['source_joint_success_by_internal_arm']==a['source_joint_success_by_internal_arm'];seen=set();states=0
    for row in manifest['rows']:
        key=(row['repetition'],row['mode']);assert key not in seen;seen.add(key)
        with np.load(fixtures/row['file']) as z:assert set(z.files)==set(reference) and all(np.array_equal(z[k],reference[k]) for k in reference)
        assert row['full_output_sha256_by_job']=={j['id']:j['files_sha256']['outputs.npz'] for j in expected['jobs']};states+=6144
    assert seen=={(rep,mode) for rep in range(4) for mode in ['serialized','async']};ratios=[];full_hashes=0
    for gpu in [0,1]:
        gate=json.loads((directory/f'source-gate-gpu{gpu}.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==a['protocol_sha256'];assert sha(directory/f'source-gate-gpu{gpu}.json')==a['mandatory_gate_sha256'][f'source-gate-gpu{gpu}.json'];b=json.loads((directory/f'benchmark-gpu{gpu}.json').read_text());assert b['status']=='passed' and b['worker']['worker_seconds']<=240
        for pair in b['rows']:
            assert pair['order']==p['paired_order'][pair['repetition']];durations={}
            for mode in ['serialized','async']:
                r=pair['runs'][mode]['pipeline'];durations[mode]=r['seconds'];assert r['all_writes_closed_hashed_and_joined'] and r['complete_artifact_files']==174;assert r['writer_metrics']['maximum_pending']<=(2 if mode=='async' else 0) and r['writer_metrics']['maximum_rss_kib']<=8*1024**2
                recorded={(Path(f['path']).parent.name,Path(f['path']).name):f['sha256'] for f in r['files']};wanted={(j['id'],name):h for j in expected['jobs'] for name,h in j['files_sha256'].items()};assert recorded==wanted and len(recorded)==171;full_hashes+=len(recorded)
            ratio=durations['serialized']/durations['async'];row=next(r for r in a['rows'] if r['repetition']==pair['repetition']);assert ratio==pair['paired_pipeline_speedup']==row['paired_pipeline_speedup'] and durations==row['seconds'];ratios.append(ratio)
    median=float(np.median(ratios));assert median==a['median_paired_complete_pipeline_speedup'] and a['decision']==('supported' if median>=1.15 else 'falsified')
    return {'status':'passed','shortest_labels':256,'source_model_inputs':768,'source_internal_states':6144,'all_timed_state_copies_compared':states,'all_timed_reported_state_copies':24576,'recorded_full_archive_hashes_compared':full_hashes,'source_joint_success_by_internal_arm':joint,'median_paired_complete_pipeline_speedup':median,'decision':a['decision'],'manifest_sha256':sha(fixtures/'manifest.json'),'audit_sha256':sha(directory/'audit.json'),'source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started,'scope':'Independent shortest-label and graph classification on all 6,144 source prefix states, all 49,152 timed board/counter/flag/margin copies and 1,368 recorded full-file hashes; source-gate bindings, resource bounds and timing decisions. Full raw scientific NPZ byte comparison ran separately on Work. This public checker does not recreate unpublished couplings or trace archives. Zero fresh accuracy evidence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);print(json.dumps(verify(p.parse_args().directory),indent=2))
