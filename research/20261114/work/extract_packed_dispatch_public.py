"""H2bs post-audit complete final-board evidence and selected factor fixtures."""
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
import study as S
import fast_npz as Z
DEST=S.OUT/'bp_first_packed_dispatch'
PUBLIC=DEST/'public-staging'


def main():
    started=time.perf_counter();audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed' and audit['timed_model_pipeline_copies']==147456
    p=json.loads((DEST/'protocol.json').read_text());source=S.ROOT/p['source']['directory'];source_audit=json.loads((source/'audit.json').read_text());assert source_audit['status']=='passed' and S.sha(source/'audit.json')==p['source']['audit_sha256']
    PUBLIC.mkdir(exist_ok=False);context=PUBLIC/'source-context';context.mkdir();fixtures=PUBLIC/'state-fixtures';fixtures.mkdir()
    seal=json.loads((DEST/'sealed-replays.json').read_text());assert len(seal['receipt_sha256'])==35
    names=['protocol.json','source-packet.json','sealed-replays.json','audit.json','audit-worker.json','live-gate.json','launcher-gate.json','live-timing.json','launcher-timing.json']+list(seal['receipt_sha256'])
    for name in sorted(set(names)):
        out=PUBLIC/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/name,out)
    for name in ['protocol.json','audit.json','sealed-replays.json','independent-input-audit.json','datasets.json','per-model-input.csv','per-input.csv']:
        shutil.copyfile(source/name,context/name)
    datasets=json.loads((source/'datasets.json').read_text())['datasets'];(context/'datasets').mkdir()
    for ds in datasets:
        file=S.ROOT/ds['path'];assert S.sha(file)==ds['sha256'];shutil.copyfile(file,context/'datasets'/file.name)
    source_rows={}
    for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
        file=source/'runs'/f'block0-{mode}'/'result.json';assert S.sha(file)==p['source']['cohort_receipts_sha256'][mode]
        target=context/'runs'/f'block0-{mode}'/'result.json';target.parent.mkdir(parents=True);shutil.copyfile(file,target)
        for row in json.loads(file.read_text())['rows']:source_rows[row['seed'],row['shard'],row['batch'],mode]=row
    partial=set();copies=[]
    for pair in range(6):
        for mode in ['original_order','packed']:
            folder=DEST/'runs'/f'pair{pair}-{mode}';r=json.loads((folder/'result.json').read_text());target=PUBLIC/'runs'/folder.name
            for model in r['models']:
                for key in ['global_answer_file','queue_file']:
                    file=folder/model[key];assert S.sha(file)==model[key+'_sha256'];shutil.copyfile(file,target/file.name)
                copies.append({'pair':pair,'mode':mode,'model_seed':model['seed'],'file':str((target/model['global_answer_file']).relative_to(PUBLIC)),'sha256':model['global_answer_file_sha256']})
            for row in r['rows']:
                if not row['zero_work'] and len(row['indices'])<32:partial.update((int(i),row['seed']) for i in row['indices'])
    (context/'source-final-answers').mkdir();source_globals={}
    for seed in range(3):
        lists={};provenance=[]
        for shard in range(16):
            for bi in range(8):
                row=source_rows[seed,shard,bi,'endpoint_dispatch'];file=source/'runs'/'block0-endpoint_dispatch'/row['file'];assert S.sha(file)==row['file_sha256']
                with np.load(file,allow_pickle=False) as z:
                    for key in z.files:
                        if key.startswith('answer_'):lists.setdefault(key[7:],[]).append(z[key])
                provenance.append({'shard':shard,'batch':bi,'file':str(file.relative_to(source)),'sha256':row['file_sha256']})
        arrays={name:np.concatenate(parts,axis=0) for name,parts in lists.items()};arrays['global_indices']=np.arange(4096);arrays['model_seed']=np.full(4096,seed,np.int64)
        file=context/'source-final-answers'/f'seed{seed}.npz';Z.save(file,**arrays);source_globals[str(seed)]={'file':str(file.relative_to(PUBLIC)),'sha256':S.sha(file),'source_batches':provenance};del arrays,lists
    source_selection=json.loads((source/'public-staging/state-fixtures/manifest.json').read_text());fixed_source={(r['input_index'],r['model_seed']) for r in source_selection['records']};selection=fixed_source|partial
    @lru_cache(maxsize=3)
    def actual(seed,shard,bi):
        row=source_rows[seed,shard,bi,'endpoint_dispatch'];file=source/'runs'/'block0-endpoint_dispatch'/row['file'];assert S.sha(file)==row['file_sha256']
        with np.load(file,allow_pickle=False) as z:return {k:z[k] for k in ['couplings']}
    @lru_cache(maxsize=1)
    def data(shard):
        with np.load(S.ROOT/datasets[shard]['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
    records=[]
    for index,seed in sorted(selection,key=lambda item:(item[1],item[0])):
        shard,local=divmod(index,256);bi,within=divmod(local,32);ds=data(shard);arrays={key:value[local:local+1] for key,value in ds.items()};arrays['couplings']=actual(seed,shard,bi)['couplings'][within:within+1]
        file=fixtures/f'i{index}-s{seed}.npz';Z.save(file,**arrays);records.append({'input_index':index,'model_seed':seed,'file':file.name,'sha256':S.sha(file),'H2br_fixed_selection':(index,seed) in fixed_source,'actual_partial_batch_input':(index,seed) in partial})
    params={}
    for seed in range(3):
        for arm in ['pl','old_binary']:
            file=source/f'model-seed{seed}-{arm}.npz';shutil.copyfile(file,fixtures/file.name);params[file.name]=S.sha(file)
    markers={str(file.relative_to(DEST)):file.read_text() for file in sorted((DEST/'barriers').rglob('*.json'))}
    S.save(PUBLIC/'barrier-markers.json',{'encoding':'Original UTF-8 text, byte-exact','markers':markers})
    S.save(fixtures/'manifest.json',{'records':records,'parameters_sha256':params,'source_fixed_selection':len(fixed_source),'all_actual_partial_batch_inputs':len(partial),'selection':'All234 H2br rescue/error/fixed-dispatch-control cases plus every source model-input actually occurring in a true partial timing batch. All repeated final boards are additionally retained for every input.','scope':'Direct-factor fixtures supplement full final-board/scalar source-preservation evidence; full internal trajectories remain sealed and independently audited on Work.'})
    S.save(PUBLIC/'evidence-manifest.json',{'protocol_sha256':S.sha(DEST/'protocol.json'),'audit_sha256':S.sha(DEST/'audit.json'),'source_globals':source_globals,'timed_globals':copies,'actual_timed_model_pipeline_copies':147456,'fresh_scientific_observations':0,'fixture_cases':len(records),'barrier_markers':len(markers),'extractor_sha256':S.sha(Path(__file__)),'seconds':time.perf_counter()-started,'scope':'Complete final boards/scalars for all147456timing copies, source-derived4096-input arrays per model, original data and selected independent factor fixtures. Original source and internal trajectory archives stay hash-bound on Work; no new scientific observations.'})
    print(json.dumps({'status':'extracted','full_timing_copies':147456,'fixture_cases':len(records),'partial_inputs':len(partial),'seconds':time.perf_counter()-started},indent=2))


if __name__=='__main__':main()
