"""H2bq post-audit public rows, every error/route, and fixed control states."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
import study as S
import fast_npz as Z

DEST=S.OUT/'bp_first_far_placement'
PUBLIC=DEST/'public-staging'


def extract():
    started=time.perf_counter()
    audit=json.loads((DEST/'audit.json').read_text())
    assert audit['status']=='passed'
    with (DEST/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==36864
    fixed={(shard*256,seed) for shard in range(16) for seed in range(3)}
    routed={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['arm']=='cascade' and r['routed']=='1'}
    failures={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['success']=='0'}
    longer={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['BP_geometry_valid_longer']=='1'}
    selected=fixed|routed|failures|longer
    PUBLIC.mkdir(exist_ok=False);fixtures=PUBLIC/'state-fixtures';fixtures.mkdir()
    for name in ['protocol.json','synthetic-gate.json','generator-synthetic-gate.json','seed-registry.json','sealed-replays.json',
                 'audit.json','audit-worker.json','per-model-input.csv','per-input.csv','datasets.json','generation-worker.json',
                 'independent-input-audit.json','input-audit-worker.json','live-gate.json','live-timing.json','launcher-gate.json',
                 'launcher-timing.json','distance-strata.json','approved-initial-design.json']:
        shutil.copyfile(DEST/name,PUBLIC/name)
    for seed in range(3):
        for arm in ['pl','old_binary']:
            name=f'model-seed{seed}-{arm}.npz';shutil.copyfile(DEST/name,fixtures/name)
    for gpu in [0,1]:
        for name in [f'gate-worker{gpu}.json',f'timing-worker{gpu}.json',f'gate-gpu{gpu}.json',f'timing-gpu{gpu}.json',f'pairs-gpu{gpu}.json']:
            shutil.copyfile(DEST/name,PUBLIC/name)
    cohorts=[]
    for folder in ['source-gates','warmup','runs']:
        for source in sorted((DEST/folder).glob('*/result.json')):
            relative=source.relative_to(DEST);target=PUBLIC/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target);cohorts.append({'file':str(relative),'sha256':S.sha(source)})
    markers={str(q.relative_to(DEST)):q.read_text() for q in sorted((DEST/'barriers').rglob('*.json'))}
    S.save(PUBLIC/'barrier-markers.json',{'encoding':'Original UTF-8 text, byte-exact','markers':markers})
    timed={}
    for pair in range(4):
        for mode in ['LP96','BP_geometry_LP96']:
            r=json.loads((DEST/'runs'/f'pair{pair}-{mode}'/'result.json').read_text())
            timed[pair,mode]={(row['job_id'],row['batch']):row for row in r['rows']}
    @lru_cache(maxsize=8)
    def batch(path,digest):
        assert S.sha(path)==digest
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files if k.startswith(('answer_','bp_')) or k in ['couplings','hidden_steps']}
    manifests=json.loads((DEST/'datasets.json').read_text())['datasets']
    @lru_cache(maxsize=1)
    def dataset(shard):
        ds=manifests[shard];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path'],allow_pickle=False) as z:return {k:z[k] for k in ['tokens','labels','indices','shortest_edges','original_endpoints','sampled_endpoints','endpoint_orientation','opened_loops','eligible_walls']}
    records=[]
    for index,seed in sorted(selected,key=lambda case:(case[1],case[0])):
        shard,local=divmod(index,256);bi,within=divmod(local,32);job=f'shard{shard:02d}-s{seed}'
        ds=dataset(shard);arrays={k:v[local:local+1] for k,v in ds.items()}
        for mode,prefix in [('LP96','control'),('BP_geometry_LP96','candidate')]:
            fields={};provenance=[]
            for pair in range(4):
                row=timed[pair,mode][job,bi];file=DEST/'runs'/f'pair{pair}-{mode}'/row['file']
                data=batch(file,row['file_sha256'])
                if pair==0:
                    if 'couplings' in arrays:assert np.array_equal(arrays['couplings'],data['couplings'][within:within+1],equal_nan=True)
                    else:arrays['couplings']=data['couplings'][within:within+1]
                for key,values in data.items():
                    if key!='couplings':fields.setdefault(key,[]).append(values[within])
                provenance.append({'pair':pair,'batch_file':str(file.relative_to(DEST)),'batch_sha256':row['file_sha256']})
            arrays.update({prefix+'_'+key:np.stack(values) for key,values in fields.items()})
            arrays[prefix+'_provenance_utf8']=np.frombuffer(json.dumps(provenance).encode(),np.uint8)
        name=f'i{index}-s{seed}.npz';Z.save(fixtures/name,**arrays)
        records.append({'file':name,'sha256':S.sha(fixtures/name),'input_index':index,'model_seed':seed,
            'fixed_control':(index,seed) in fixed,'actually_routed':(index,seed) in routed,
            'any_arm_failure':(index,seed) in failures,'BP_valid_longer':(index,seed) in longer,
            'dataset':manifests[shard]})
    parameters={f'model-seed{seed}-{arm}.npz':S.sha(fixtures/f'model-seed{seed}-{arm}.npz') for seed in range(3) for arm in ['pl','old_binary']}
    S.save(fixtures/'manifest.json',{'protocol_sha256':S.sha(DEST/'protocol.json'),'audit_sha256':S.sha(DEST/'audit.json'),
        'fixed_controls':len(fixed),'complete_routed_cases':len(routed),'complete_any_arm_failure_cases':len(failures),
        'complete_BP_valid_longer_cases':len(longer),'parameters_sha256':parameters,'records':records,
        'selection':'Union of first input per shard/model, every actual route, every failure in any logical arm and every BP valid-longer case. Every selected case retains actual four-pair final states and executed BP prefixes, inputs, labels and couplings.',
        'scope':'Portable final-state and numerical/graph checks. Complete internal candidate/trajectory checks remain in the independent audit and hash-bound Work archives.'})
    S.save(PUBLIC/'cohort-receipts.json',{'receipts':cohorts,'extractor_sha256':S.sha(Path(__file__)),
        'seconds':time.perf_counter()-started,'barrier_marker_count':len(markers),'cache':batch.cache_info()._asdict(),'post_run_cache_order':'Seed-major cases keep all8paired batches resident between nearby cases; same selected set and actual bytes, no inference change.'})
    print(json.dumps({'status':'extracted','cases':len(records),'routed':len(routed),'any_arm_failures':len(failures),'valid_longer':len(longer)},indent=2))


if __name__=='__main__':extract()
