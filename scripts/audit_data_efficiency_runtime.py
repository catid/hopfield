"""Read-only data-efficiency preflight and explicitly labeled workload projections."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shutil,statistics,subprocess,time
ROOT=Path(__file__).resolve().parents[1];start=time.perf_counter()
path=ROOT/'scripts/collect_data_efficiency.py'
spec=importlib.util.spec_from_file_location('dataeff_collector_preflight',path);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
hashes=C.Hashes();config=C.read(C.CONFIG)
assert len(config['jobs'])==102 and len({j['id'] for j in config['jobs']})==102
assert sorted((j['k'],j['seed']) for j in config['jobs'])==[(k,s) for k in C.KS for s in range(6)]
assert config['protocol']['variants']==99 and config['protocol']['updates']==32 and config['protocol']['warm_fit_repeats']==7
for name,digest in config['source_sha256'].items():assert hashes.sha(Path(config['source'])/name)==digest
ref_path=C.DATA/'sudoku-extreme-1k-aug-1000/build_provenance.json';ref=C.read(ref_path)
assert ref['seed']==42 and len(ref['selected_train_rows'])==1000 and hashes.sha(ref_path)==config['reference_provenance_sha256']
verified=[];unique_data={}
for job in config['jobs']:
 key=(job['k'],job['seed'] if job['k']!=1000 else 0)
 if key not in unique_data:unique_data[key]=C.dataset(job,hashes,ref)
 verified.append(dict(job,data=unique_data[key]))
test_path=ROOT/'upstream/data/full-test/full-all.npz'
test_pin=next(r for r in C.read(ROOT/'artifacts/deployment/data_manifest.json') if r['path']=='full-test/full-all.npz')
assert hashes.sha(test_path)==test_pin['sha256']
pub_path=ROOT/'upstream/results/sudoku-data-eff-shared-v99.json';published=C.read(pub_path)['rows']
lookup={(r['k'],r['seed']):r for r in published};assert len(lookup)==102
calibrations=[]
for p in sorted((ROOT/'upstream/experiments/runs/CRAM-S/full/dev').glob('*.json')):
 d=C.read(p);calibrations.append({'path':str(p.relative_to(ROOT)),'sha256':hashes.sha(p),'puzzles':d['signature']['n'],'D':d['signature']['D'],'seconds':d['per_puzzle']['seconds'],'mean_legacy_proxy_flop':d['summary']['mean_flop'],'legacy_proxy_operations_per_second':d['summary']['mean_flop']*d['signature']['n']/d['per_puzzle']['seconds']})
lo=min(r['legacy_proxy_operations_per_second'] for r in calibrations);hi=max(r['legacy_proxy_operations_per_second'] for r in calibrations)
estimates=[]
for job in config['jobs']:
 p=lookup[(job['k'],job['seed'])];test=p.get('test10k');ops=test['mean_flop']*test['n'] if test else None
 estimates.append(dict(job,published_cold_fit_s=p['fit_gpu_cold_s'],published_warm_median_s=p['fit_gpu_warm_median_s'],estimated_published_eight_fit_s=p['fit_gpu_cold_s']+7*p['fit_gpu_warm_median_s'],published_search_legacy_operations=ops,estimated_search_seconds_from_local_dev_min=ops/hi if ops else None,estimated_search_seconds_from_local_dev_max=ops/lo if ops else None))
selected_rows=sum(j['k']*100 for j in config['jobs']);archive_bytes=selected_rows*81*2
# Max-size search state, 4 chains and 10k puzzles, using source tensor dtypes.
# Per-chain: tgt int64, fr bool; six int32 cell arrays x/freelist/fmask/y/ycert/flag;
# G(81,9)float32 and four scalars nfree/P/Q/found.
state_per_chain_bytes=81*(8+1+6*4+9*4)+4*4
state_per_puzzle_bytes=8+1+8+1+4+1+8
checkpoint_bytes=40000*state_per_chain_bytes+10000*state_per_puzzle_bytes+8192
permanent_model_bytes=102*(263169*4+16384)
# JSON estimate scales the observed full-test output by number of records, with 2x headroom.
json_template=ROOT/'upstream/experiments/runs/CRAM-S/full/test/cram0-hb-fit-s0.json'
json_bytes=2*102*json_template.stat().st_size*10000/412786
reserve=archive_bytes+permanent_model_bytes+json_bytes+6*checkpoint_bytes
usage=shutil.disk_usage(ROOT)
first=[r for r in estimates if r['k']==1000]
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU-only audit; no fit/search dispatched, no source/config mutation, no test rerun',
'preflight':{'jobs':len(verified),'unique_datasets':len(unique_data),'all_arrays_shapes_hashes_manifests_and_group_layouts_valid':True,'all_frozen_source_files_valid':len(config['source_sha256']),'test_dataset_sha256':test_pin['sha256'],'reference_seed':42,'config_sha256':hashes.sha(C.CONFIG),'data_and_recipe_rows':verified,'seconds':time.perf_counter()-start,'bytes_rehashed':hashes.bytes_read,'validated_cached_hashes':hashes.hits},
'estimate_method':{'fit':'Published H200 cold + seven times published warm median; an estimate because individual repeats are absent. Excludes file reads, serialization, process startup, and CUDA context. Not a local fit measurement.','search':'Published legacy search operation counts divided by measured proxy operations/second of prior local 10k development searches. Workload projection only; model connectivity, long-tail behavior, recreated data, and compilation can change elapsed time. No inference was run for these estimates.','published_source':str(pub_path.relative_to(ROOT)),'published_source_sha256':hashes.sha(pub_path),'local_calibrations':calibrations,'missing_published_search_rows':[r['id'] for r in estimates if r['published_search_legacy_operations'] is None]},
'first_k1000_estimate':{'fit_eight_runs_h200_seconds_min':min(r['estimated_published_eight_fit_s'] for r in first),'fit_eight_runs_h200_seconds_max':max(r['estimated_published_eight_fit_s'] for r in first),'search_seconds_workload_projection_min':min(r['estimated_search_seconds_from_local_dev_min'] for r in first),'search_seconds_workload_projection_max':max(r['estimated_search_seconds_from_local_dev_max'] for r in first)},
'all_published_fit_work_seconds_estimate':sum(r['estimated_published_eight_fit_s'] for r in estimates),
'known_search_work_gpu_seconds_projection_range':[sum(r['estimated_search_seconds_from_local_dev_min'] or 0 for r in estimates),sum(r['estimated_search_seconds_from_local_dev_max'] or 0 for r in estimates)],
'per_job_estimates':estimates,
'disk':{'free_bytes':usage.free,'total_bytes':usage.total,'selected_training_rows_all_jobs':selected_rows,'uncompressed_training_archive_payload_bytes':archive_bytes,'model_payload_plus_16k_overhead_bytes':permanent_model_bytes,'test_json_estimate_with_2x_headroom_bytes':json_bytes,'max_search_checkpoint_payload_bytes_per_lane':checkpoint_bytes,'output_reserve_bytes':reserve,'reserve_description':'Uncompressed train payload + models + observed JSON size scaled with 2x headroom + two max-size checkpoint copies for each of 3 lanes. Compression/header/runtime variations are not an exact storage prediction.','fits_in_current_free_space':reserve<usage.free},
'queue_optimization':{'worker_sha256':hashes.sha(ROOT/'scripts/data_efficiency_queue.py'),'validation':'artifacts/data_efficiency/queue-wait-validation.json','change':'Wait for child exit with 15-second progress timeout, replacing unconditional 15-second sleeps; failure handling and dispatch order unchanged.','maximum_avoided_sleep_seconds_all_204_phases':204*15,'uniform_exit_phase_expected_avoided_sleep_seconds_all_204_phases':204*7.5,'not_elapsed_speedup_measurement':'The upper bound/expectation refer only to avoided queue sleep, not inference speedup.'},
'gpu_lane_coordination':'All additional lanes remain controlled by root; no MPS adoption or new GPU jobs. Figure4 MPS gain does not establish a data-efficiency benefit.'}
(ROOT/'artifacts/data_efficiency/runtime-preflight-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_job_estimates','estimate_method','preflight')},indent=2))
print(json.dumps({k:v for k,v in result['preflight'].items() if k!='data_and_recipe_rows'},indent=2))
