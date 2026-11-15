"""H4al: unchanged degree curvature on two synthetic graph families."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import signal
import subprocess
import time
from weak_weight_study import ROOT,HERE,RIPPER,sha,save
import heterogeneous_noise_study as inherited
from regularity_intensity_study import corrupt

OUT=ROOT/'artifacts/research/20261114/foureyes/varying_degree'
PUBLIC=ROOT/'results/research/foureyes/varying_degree/controls'
PROTOCOL=HERE/'varying_degree.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
DESIGN=ROOT/'artifacts/research/20261114/foureyes/varying_degree_design'
ARMS=['adamw','canonical','restricted','regularity']
CONDITIONS=['cycle81','cliques9']
GROUPS=64
ROWS=2400


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def bind():
    inherited.OUT=OUT;inherited.PUBLIC=PUBLIC;inherited.PROTOCOL=PROTOCOL
    inherited.CONDITIONS=CONDITIONS
    return inherited


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate_path=HERE/'varying-degree-preflight.json';gate=json.loads(gate_path.read_text());assert gate['status']=='passed'
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest
    parent=json.loads((HERE/'fresh_curvature.json').read_text());lp=json.loads((HERE/'degree_curvature.json').read_text())
    p=dict(id='H4al-two-synthetic-degrees',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=[[g] for g in range(64)],group_count=64,rows_per_group=2400,conditions=CONDITIONS,arms=ARMS,
        graph_seed_base=202612100000,coloring_seed_base=202612110000,heatmap_seed_base=202612120000,noise_rng_base=202612130000,augmentation_rng_base=202612140000,
        seed_rule='Index=2*bank+family_index; separate graph/coloring/hotmap/noise streams add index. BP augmentation adds bank and is shared across family conditions.',
        scope='Two explicitly synthetic families on81nodes: one Hamiltonian81cycle and nine disjoint9cliques, randomly cell-permuted independently in each bank.64banks/family; not arbitrary degree2/8 graphs, no official source or puzzle inputs.',
        generator='Cycle: uniform initial digit0..8 and independent uniform nonzero mod9 increments on80path edges; reject closing equality, retain first2400accepted. Each proper labeled9coloring has equal probability. Fixed512candidate batches, cap10000; shortage=incomplete with no replacement rescue. Cliques: independent uniform9digit permutations for every row/block. All81cellsfree. Retain any duplicate boards; no outcome filtering.',
        observation_scope='The9clique family has exactly9instances ofeachdigit globally; cycle family neednot. Different source distributions are explicit. Graph orders sealed before coloring/noise generation; all code and seeds frozen before any actual graph/source draws.',
        noise='Perbank independently seeded18hotcells withweight32, others1; exactly9freecells selected by unchanged exponentialrace and nonzero offsets. Allcellsfree; same corruptedobservations for allestimators. Graph family data distributions differ, not a paired family comparison.',
        primary='Separately for each family: at least61/64 certified integral exact selected graphs.',
        degree_endpoint='Separately for each family: true degree selected in at least61/64 resolved banks. This weaker diagnostic follows from exactgraph success; not independent corroboration.',
        decision=lp['decision'],degrees=lp['degrees'],candidate_degrees=lp['candidate_degrees'],selector=lp['selector'],prior='Common degree and bounded grid0..40 on81nodes. No true degree2/8/20, true edge count, graph family/order or geometry enters inference. Tie rule chooses lowest degree.',objective=lp['objective'],proof=lp['proof'],solver=lp['solver'],
        true_degrees_for_evaluation_only={'cycle81':2,'cliques9':8},
        controls='BPAdamW32 first; canonicalPL256; rawrestrictedcountgap; rawregularity; centeredregularity. Fixeddegree20 cell diagnostic, not a target. Identical frozen recipes; no matchedcompute/optimum claim. Truth/order never passed to estimators.',
        regularity_rule=parent['regularity_rule'],adamw=parent['adamw'],canonical=parent['canonical'],inherited_rules=parent['inherited_rules'],gate=parent['gate'],
        budget=dict(preparation_wall_seconds=90,gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,selection_wall_seconds=30,audit_wall_seconds=240,total_gpu_minutes_upper_bound=15),
        lp_budget=dict(workers=8,new_LPs=5120,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0),
        source_sha256=gate['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in [gate_path,HERE/'fresh_curvature.json',HERE/'degree_curvature.json',ROOT/'results/research/foureyes/canonical_confirm/models.npz']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    from varying_degree_sampling import graph_order,sample
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True);start=time.monotonic()
    orders=np.stack([graph_order(p['graph_seed_base']+i) for i in range(128)])
    np.savez_compressed(OUT/'graph-orders.npz',orders=orders)
    save(OUT/'graph-order-seal.json',dict(protocol_sha256=sha(PROTOCOL),orders_sha256=sha(OUT/'graph-orders.npz'),sealed_before_colorings_and_noise=True,utc=datetime.now(timezone.utc).isoformat()))
    entries=[];jobs=[];heatmaps=[]
    for group in range(64):
        for fi,condition in enumerate(CONDITIONS):
            index=2*group+fi;clean,receipt=sample(orders[index],condition,p['coloring_seed_base']+index)
            hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+index).choice(81,18,replace=False));heatmaps.append(hot.tolist());rng=np.random.default_rng(p['noise_rng_base']+index)
            raw=dict(solutions=clean,free=np.ones((2400,81),bool),uniforms=1-rng.random((2400,81)),offsets=rng.integers(1,9,(2400,81),dtype=np.int64))
            noisy,cells=corrupt(np,clean,raw['free'],raw['uniforms'],raw['offsets'],hot,'hot18_weight32')
            job_id=f'g{group:02d}-{condition}';path=OUT/'inputs'/f'{job_id}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**raw,noisy=noisy,cells=cells)
            entries.append(dict(job_id=job_id,index=index,group=group,condition=condition,banks=[group],hot_cells=hot.tolist(),sampler_receipt=receipt,path=str(path.relative_to(ROOT)),sha256=sha(path)))
            jobs.append(dict(job_id=job_id,bank=group,condition=condition,status='pending'))
            assert time.monotonic()-start<p['budget']['preparation_wall_seconds']
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),hot_cells_by_input=heatmaps,entries=entries,graph_order_seal_sha256=sha(OUT/'graph-order-seal.json')))
    save(OUT/'queue.json',jobs);save(OUT/'preparation.json',dict(wall_seconds=time.monotonic()-start,entries=128,synthetic_observations=307200,new_official_source_rows=0,new_solver_inputs=0));print(json.dumps(dict(prepared=128,wall_seconds=time.monotonic()-start)))


def select_graphs():
    import numpy as np
    from regularity_gap import select
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic()
    source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['input_manifest_sha256']==sha(OUT/'inputs.json')
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==128 and all(j['status']=='complete' for j in queue)
    destination=OUT/'regularity';destination.mkdir(exist_ok=False);rows=[];supports=[]
    for e in manifest['entries']:
        d=OUT/'fits'/e['job_id'];receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['input_sha256']==e['sha256']
        assert sha(d/'models.npz')==receipt['output_sha256']['models.npz']
        with np.load(d/'models.npz',allow_pickle=False) as z:counts=z['counts'].copy()
        support,record=select(counts)
        record.update(job_id=e['job_id'],group=e['group'],protocol_sha256=sha(PROTOCOL),models_sha256=sha(d/'models.npz'))
        path=destination/(e['job_id']+'.json');save(path,record)
        supports.append(support);rows.append(dict(group=e['group'],job_id=e['job_id'],record_sha256=sha(path)))
        assert time.monotonic()-start<p['budget']['selection_wall_seconds']
    np.savez_compressed(destination/'supports.npz',supports=np.stack(supports),groups=np.repeat(np.arange(64),2),conditions=np.tile(CONDITIONS,64))
    save(destination/'receipt.json',dict(status='complete',protocol_sha256=sha(PROTOCOL),rows=rows,supports_sha256=sha(destination/'supports.npz'),wall_seconds=time.monotonic()-start,selector_sha256=sha(HERE/'regularity_gap.py')))
    print(json.dumps(dict(status='complete',groups=64,entries=128,selection_wall_seconds=time.monotonic()-start)))


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic();cpu_start=inherited.cpu_seconds()
    workers=[];logs=[];samples=[];status='failed';failure=None
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            assert time.monotonic()-start<p['budget'][phase+'_phase_wall_seconds'],'phase_wall_budget'
            assert all(w.poll() in [None,0] for w in workers),'worker_failure'
            q=json.loads((OUT/'queue.json').read_text())
            samples.append(dict(elapsed=time.monotonic()-start,complete=sum(x['status']=='complete' for x in q),running=sum(x['status']=='running' for x in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(.5)
        assert all(w.returncode==0 for w in workers);status='complete'
    except BaseException as error:failure=repr(error);raise
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for f in logs:f.close()
        result=dict(status=status,failure=failure,wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=inherited.cpu_seconds()-cpu_start,samples=samples,returncodes=[w.returncode for w in workers])
        save(OUT/f'{phase}-launch.json',result);print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','gate','fit','select','launch-gate','launch-fit']);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.action in ['gate','fit']:getattr(bind(),args.action)(args.gpu)
    elif args.action.startswith('launch-'):launch(args.action.split('-')[1])
    elif args.action=='select':select_graphs()
    else:globals()[args.action]()
