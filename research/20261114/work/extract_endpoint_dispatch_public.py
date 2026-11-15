"""H2br complete scalar evidence and all rescue/error/stratum-control fixtures."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
import study as S
import fast_npz as Z
DEST=S.OUT/'bp_first_endpoint_dispatch'
PUBLIC=DEST/'public-staging'
MODES=['LP96','BP_geometry_LP96','endpoint_dispatch']


def extract():
    started=time.perf_counter();audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed' and audit['accuracy_block']==0
    with (DEST/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==49152
    fixed=set()
    for shard in range(16):
        for seed in range(3):
            for high in ['0','1']:
                group=[int(r['input_index']) for r in rows if r['arm']=='dispatcher' and int(r['input_index'])//256==shard and int(r['model_seed'])==seed and r['dispatch_high']==high]
                if group:fixed.add((min(group),seed))
    rescues={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['arm'] in ['cascade','dispatcher'] and r['routed']=='1'}
    failures={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['success']=='0'}
    longer={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['BP_geometry_valid_longer']=='1'}
    selected=fixed|rescues|failures|longer;bycase={(int(r['input_index']),int(r['model_seed']),r['arm']):r for r in rows}
    PUBLIC.mkdir(exist_ok=False);fixtures=PUBLIC/'state-fixtures';fixtures.mkdir()
    names=['protocol.json','candidate-protocol.json','initial-design-and-seed-registry.json','seed-registry.json','policy-synthetic-gate.json','generator-synthetic-gate.json','accounting-synthetic-gate.json','compatibility-synthetic-gate.json','source-archive-preflight.json','source-cost-preflight.json','study-synthetic-gate.json','sealed-replays.json','audit.json','audit-worker.json','per-model-input.csv','per-input.csv','datasets.json','generation-worker.json','independent-input-audit.json','input-audit-worker.json','live-timing.json','launcher-timing.json','descriptive-strata-and-costs.json']
    seal=json.loads((DEST/'sealed-replays.json').read_text())
    names+=list(seal['cohort_and_worker_receipt_sha256'])
    names+=['source-compatibility-attempt1/live-gate.json','source-compatibility-attempt1/launcher-gate.json']
    for name in sorted(set(names)):
        target=PUBLIC/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/name,target)
    for seed in range(3):
        for arm in ['pl','old_binary']:
            name=f'model-seed{seed}-{arm}.npz';shutil.copyfile(DEST/name,fixtures/name)
    markers={str(q.relative_to(DEST)):q.read_text() for base in [DEST/'barriers',DEST/'source-compatibility-attempt1'/'barriers'] for q in sorted(base.rglob('*.json'))}
    S.save(PUBLIC/'barrier-markers.json',{'encoding':'Original UTF-8 text, byte-exact','markers':markers})
    timed={}
    for block in range(6):
        for mode in MODES:
            r=json.loads((DEST/'runs'/f'block{block}-{mode}'/'result.json').read_text());timed[block,mode]={(row['job_id'],row['batch']):row for row in r['rows']}
    @lru_cache(maxsize=18)
    def batch(path,digest,include_bp):
        assert S.sha(path)==digest
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files if k.startswith('answer_') or (include_bp and k.startswith('bp_')) or k in ['couplings','hidden_steps']}
    manifests=json.loads((DEST/'datasets.json').read_text())['datasets']
    @lru_cache(maxsize=1)
    def dataset(shard):
        ds=manifests[shard];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path'],allow_pickle=False) as z:return {k:z[k] for k in ['tokens','labels','indices','shortest_edges','original_endpoints','sampled_endpoints','opened_loops','eligible_walls','endpoint_orientation','mixture_component']}
    endpoint_rows=[]
    for shard in range(16):
        ds=dataset(shard)
        for local,index in enumerate(ds['indices']):
            pair=ds['sampled_endpoints'][local]
            endpoint_rows.append({'input_index':int(index),'start':int(pair[0]),'goal':int(pair[1]),'mixture_component':int(ds['mixture_component'][local]),'endpoint_orientation':int(ds['endpoint_orientation'][local])})
    with (PUBLIC/'input-endpoint-records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(endpoint_rows[0]));writer.writeheader();writer.writerows(endpoint_rows)
    records=[]
    for index,seed in sorted(selected,key=lambda case:(case[1],case[0])):
        shard,local=divmod(index,256);bi,within=divmod(local,32);job=f'shard{shard:02d}-s{seed}';ds=dataset(shard);arrays={k:v[local:local+1] for k,v in ds.items()}
        for mode,prefix in [('LP96','control'),('BP_geometry_LP96','cascade'),('endpoint_dispatch','dispatch')]:
            fields={};provenance=[]
            for block in range(6):
                row=timed[block,mode][job,bi];file=DEST/'runs'/f'block{block}-{mode}'/row['file'];data=batch(file,row['file_sha256'],mode=='BP_geometry_LP96')
                if block==0:
                    if 'couplings' in arrays:assert np.array_equal(arrays['couplings'],data['couplings'][within:within+1],equal_nan=True)
                    else:arrays['couplings']=data['couplings'][within:within+1]
                for key,values in data.items():
                    if key!='couplings':fields.setdefault(key,[]).append(values[within])
                provenance.append({'block':block,'batch_file':str(file.relative_to(DEST)),'batch_sha256':row['file_sha256']})
            arrays.update({prefix+'_'+key:np.stack(values) for key,values in fields.items()});arrays[prefix+'_provenance_utf8']=np.frombuffer(json.dumps(provenance).encode(),np.uint8)
        name=f'i{index}-s{seed}.npz';Z.save(fixtures/name,**arrays);scalar=bycase[index,seed,'dispatcher']
        records.append({'file':name,'sha256':S.sha(fixtures/name),'input_index':index,'model_seed':seed,'fixed_control':(index,seed) in fixed,'actually_rescued':(index,seed) in rescues,'any_arm_failure':(index,seed) in failures,'BP_valid_longer':(index,seed) in longer,'dispatch_high':int(scalar['dispatch_high']),'mixture_component':int(scalar['mixture_component']),'dataset':manifests[shard]})
    parameters={f'model-seed{seed}-{arm}.npz':S.sha(fixtures/f'model-seed{seed}-{arm}.npz') for seed in range(3) for arm in ['pl','old_binary']}
    S.save(fixtures/'manifest.json',{'protocol_sha256':S.sha(DEST/'protocol.json'),'audit_sha256':S.sha(DEST/'audit.json'),'fixed_controls':len(fixed),'complete_rescued_cases':len(rescues),'complete_any_arm_failure_cases':len(failures),'complete_BP_valid_longer_cases':len(longer),'parameters_sha256':parameters,'records':records,
        'selection':'Every BP-to-LP rescue in either cascade, every arm error, every underlying BP valid-longer case and first input per shard/model in every nonempty token dispatch group. Every case retains actual six-block BP-prefix, LP96, cascade and dispatcher final states.',
        'scope':'Portable final-state numerical/graph/policy checks; full internal trajectories and dispatcher partial BP-prefix archives remain independently audited and hash-bound on Work.'})
    S.save(PUBLIC/'cohort-receipts.json',{'extractor_sha256':S.sha(Path(__file__)),'seconds':time.perf_counter()-started,'barrier_marker_count':len(markers),'cache':batch.cache_info()._asdict(),'scope':'Post-run extraction only; bounded18-batch cache and seed-major fixture order do not change inference or selected cases.'})
    print(json.dumps({'status':'extracted','cases':len(records),'fixed_controls':len(fixed),'rescues':len(rescues),'any_arm_failures':len(failures),'valid_longer':len(longer)},indent=2))


if __name__=='__main__':extract()
