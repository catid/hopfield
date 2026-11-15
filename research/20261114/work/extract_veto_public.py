"""Self-contained H2bf geometry, cohort and timing-output certificates."""
import hashlib
import json
import shutil
import numpy as np
import study as S
import fast_npz as Z
import connected_block_moves as OLD
from veto_speed_study import DEST,GATE,SOURCE


def main():
    target=DEST/'public-fixtures';target.mkdir(exist_ok=True);geo=target/'geometry';geo.mkdir(exist_ok=True);manifest=json.loads((GATE/'geometry-fixtures/manifest.json').read_text());expected=[]
    for row in manifest['fixtures']:
        path=GATE/'geometry-fixtures'/row['file'];assert S.sha(path)==row['sha256'];shutil.copyfile(path,geo/path.name)
        with np.load(path) as z:
            for digest,t,b in zip(row['state_hashes'],z['tokens'],z['boards']):
                cells,stats=OLD.candidates(t,b);expected.append({'state_hash':digest,'candidates':[list(c) for c in cells],'geometry':stats})
    S.save(geo/'manifest.json',manifest);S.save(geo/'expected-geometry.json',expected);all_hashes=set()
    for path in (GATE/'archive-rows').glob('*.json'):all_hashes.update(json.loads(path.read_text())['unique_state_references'])
    assert len(all_hashes)==25232 and [r['state_hash'] for r in expected]==sorted(all_hashes)[:1024];S.save(geo/'all-unique-state-hashes.json',{'domain':'H2bf-geometry-v1','state_hashes':sorted(all_hashes),'archive_gate_sha256':S.sha(GATE/'archive-gate.json')})
    ds=json.loads((SOURCE/'datasets.json').read_text())['datasets'][0]
    with np.load(S.ROOT/ds['path']) as z:arrays={'tokens':z['tokens'],'labels':z['labels']}
    saved={k:[] for k in ['predictions','iterations','stopped','finite','native_cap','block_cap']};source_rows=[];trace_hashes=[]
    for seed in range(3):
        run=SOURCE/'runs'/f'shard00-s{seed}';receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256'];source_rows.append({'seed':seed,'source_outputs_sha256':receipt['outputs_sha256']})
        with np.load(run/'outputs.npz') as z:
            for key in saved:saved[key].append(z['block_native_cap' if key=='native_cap' else key][3 if key in ['native_cap','block_cap'] else 5])
        for item in receipt['trace_archives']:
            with np.load(run/item['path']) as z:states=z['states_before'];indices=z['valid_prefix_indices'];payload=json.loads(z['trace_utf8'].tobytes())
            assert S.sha(run/item['path'])==item['sha256'];digest=hashlib.sha256(states.tobytes()+indices.tobytes()+json.dumps({k:payload[k] for k in ['trace','initial_terminal_checks','final_terminal_checks']},sort_keys=True,separators=(',',':')).encode()).hexdigest();trace_hashes.append({'seed':seed,'offset':item['offset'],'arm':item['arm'],'sha256':digest})
    arrays.update({k:np.stack(v) for k,v in saved.items()});Z.save(target/'source-cohort.npz',**arrays);S.save(target/'source-cohort.json',{'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'dataset':ds,'model_rows':source_rows,'trace_hashes':trace_hashes,'cohort_npz_sha256':S.sha(target/'source-cohort.npz'),'independent_inputs':256,'model_inputs':768,'scope':'Previously audited H2be shard00 reused for performance measurements, not fresh accuracy evidence.'});S.save(target/'manifest.json',{'files':{str(p.relative_to(target)):S.sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p!=target/'manifest.json'},'source_sha256':S.sha(__file__)})


if __name__=='__main__':main()
