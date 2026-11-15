"""H1cg fresh heterogeneous acquisition; both methods receive one sealed query record."""
from datetime import datetime,timezone
from fractions import Fraction
import argparse,json,os,shutil
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from run import ROOT,HERE,sha,save
from partial_degree_study import npz
from balanced_mask_math import line_complements
from higher_noise_study import selected_answers
from estimated_channel_study import fit_models,numeric_state
from pair_likelihood_math import UncertifiedCost
from cell_channel_math import score as supplied_score,independent_score as supplied_reference,validate_channels
from estimated_channel_math import score as estimated_score,independent_score as estimated_reference,estimate,independent_estimate
OUT=ROOT/'artifacts/research/20261114/ripper/cell_channel'
REGISTRY=OUT.parent/'cell_channel_registration'
PUBLIC=ROOT/'results/research/ripper/cell_channel'
PROTOCOL=HERE/'protocol-cell-channel.json'
PARENT=ROOT/'results/research/ripper/heterogeneous_channel'
ARMS=['supplied_per_cell','estimated_scalar']
RATES=((3,80),(3,40),(3,20))


def read(path):return json.loads(path.read_text())


def verify(raw=False,public=False):
    p=read(PROTOCOL)
    for rel,h in (p['source_sha256']|p['input_sha256']).items():
        path=ROOT/rel
        if public and rel.startswith(str(REGISTRY.relative_to(ROOT))+'/'):
            assert path.name in ['excluded_ids.npz','ledger.json'];path=PUBLIC/path.name
        assert sha(path)==h,rel
    if raw:
        for rel,h in p['raw_input_sha256'].items():assert sha(ROOT/rel)==h,rel
    return p


def make_masks(seed):
    rng=np.random.default_rng(seed);cells=rng.permutation(81);lines=rng.permutation(90)
    mask=np.zeros((90,81),bool);mask[:,cells]=line_complements()[lines];return mask


def make_cell_channels(seed):
    order=np.random.default_rng(seed).permutation(81).astype(np.int64);classes=np.empty(81,np.int8)
    classes[order]=np.repeat(np.arange(3,dtype=np.int8),27);return dict(classes=classes,permutation=order)


def freeze():
    from cell_channel_data import channel_event
    assert not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists() and not PUBLIC.exists()
    gatepath=HERE/'cell-channel-preflight.json';seedpath=HERE/'cell-channel-seed-audit.json';proposal=HERE/'cell-channel-proposal.json'
    gate=read(gatepath);seeds=read(seedpath);design=read(proposal)
    assert gate['status']=='passed' and not gate['cohort_draws_or_estimates_read'] and gate['proposal_sha256']==sha(proposal)
    assert seeds['status']=='passed' and gate['seed_audit_sha256']==sha(seedpath)
    mathpath=HERE/'cell-channel-math-check.json';mathgate=read(mathpath)
    assert gate['mixed_math_gate_sha256']==sha(mathpath) and mathgate['status']=='passed' and mathgate['proposal_sha256']==sha(proposal)
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in seeds['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==h,rel
    parent=read(HERE/'protocol-heterogeneous-channel.json');prior=npz(PARENT/'excluded_ids.npz');selected=npz(PARENT/'selected_sources.npz')['all_source_row']
    excluded=np.union1d(prior['excluded'],selected);ripper=np.union1d(prior['ripper_prior'],selected);old=read(PARENT/'source-audit.json')
    assert len(excluded)==old['updated_combined_exclusion_union']==2437929 and len(ripper)==old['updated_ripper_ledger_union']==2201504 and np.isin(ripper,excluded).all()
    REGISTRY.mkdir();np.savez_compressed(REGISTRY/'excluded_ids.npz',excluded=excluded,ripper_prior=ripper)
    save(REGISTRY/'ledger.json',dict(status='sealed_before_new_draws',scope='Declared combinedRipper/H4ak exclusions throughH1cf; not globalrepository untouchedness.',excluded_unique=len(excluded),ripper_prior_unique=len(ripper),eligible_rows=3831994-len(excluded),
        parent_protocol_sha256=sha(HERE/'protocol-heterogeneous-channel.json'),parent_source_audit_sha256=sha(PARENT/'source-audit.json'),parent_selected_ids_sha256=sha(PARENT/'selected_sources.npz'),parent_exclusions_sha256=sha(PARENT/'excluded_ids.npz'),excluded_file_sha256=sha(REGISTRY/'excluded_ids.npz')))
    inputs=[gatepath,seedpath,proposal,HERE/'protocol-heterogeneous-channel.json',HERE/'protocol-channel-misspecification.json',HERE/'protocol-estimated-channel.json',
        HERE/'cell-channel-math-check.json',HERE/'estimated-channel-math-check.json',REGISTRY/'excluded_ids.npz',REGISTRY/'ledger.json']
    inputs += [PARENT/name for name in ['excluded_ids.npz','selected_sources.npz','source-audit.json']]
    p=dict(design,status='frozen_before_fresh_draws_masks_noise_estimates_scores_and_fits',frozen_utc=datetime.now(timezone.utc).isoformat(),
        sampling=design['source'],source=parent['source'],source_csv_sha256=parent['source_csv_sha256'],source_seed_base=design['seeds']['source'],balanced_seed_base=design['seeds']['balanced_mask'],
        cell_channel_seed_base=design['seeds']['cell_channel_assignment'],noise_seed_base=design['seeds']['first_noise'],repeat1_seed_base=design['seeds']['repeat1'],repeat2_seed_base=design['seeds']['repeat2'],
        acquisition_channels=[channel_event(rate) for rate in RATES],solver=dict(parent['solver'],random_seed_base=design['seeds']['solver_base']),source_sha256=gate['source_sha256'],
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs},raw_input_sha256={parent['source']:parent['source_csv_sha256']})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),excluded_ids=len(excluded),eligible_rows=3831994-len(excluded))),flush=True)


def select():
    p=verify();assert not OUT.exists();excluded=npz(REGISTRY/'excluded_ids.npz')['excluded']
    eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True);assert len(eligible)==1394065
    draws=np.stack([np.random.default_rng(p['source_seed_base']+seed).choice(eligible,90,replace=False) for seed in range(72)])
    OUT.mkdir();np.savez_compressed(OUT/'selected_sources.npz',source_rows=draws,all_source_row=np.unique(draws))
    save(OUT/'identity-seal.json',dict(status='sealed_before_selected_labels',protocol_sha256=sha(PROTOCOL),selected_sources_sha256=sha(OUT/'selected_sources.npz'),new_unique_source_ids=len(np.unique(draws)),crossbank_repeated_instances=draws.size-len(np.unique(draws))))


def predraw_all():
    p=verify();assert (OUT/'identity-seal.json').exists() and not (OUT/'input_manifest.json').exists()
    for seed in range(72):
        folder=OUT/f's{seed}';folder.mkdir();np.savez_compressed(folder/'masks.npz',balanced=make_masks(p['balanced_seed_base']+seed));np.savez_compressed(folder/'cell-channels.npz',**make_cell_channels(p['cell_channel_seed_base']+seed))
        save(folder/'predraw-receipt.json',dict(status='sealed_before_selected_labels',seed=seed,protocol_sha256=sha(PROTOCOL),identity_seal_sha256=sha(OUT/'identity-seal.json'),masks_sha256=sha(folder/'masks.npz'),cell_channels_sha256=sha(folder/'cell-channels.npz')))


def materialize():
    p=verify();predraws={}
    for seed in range(72):
        folder=OUT/f's{seed}';receipt=read(folder/'predraw-receipt.json')
        assert receipt['seed']==seed and receipt['protocol_sha256']==sha(PROTOCOL) and receipt['masks_sha256']==sha(folder/'masks.npz') and receipt['cell_channels_sha256']==sha(folder/'cell-channels.npz')
        predraws[str(seed)]=sha(folder/'predraw-receipt.json')
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];draws=npz(OUT/'selected_sources.npz')['source_rows'];wanted={};solutions=np.empty((72,90,81),np.int8)
    for seed in range(72):
        for offset,row in enumerate(draws[seed]):wanted.setdefault(int(row),[]).append((seed,offset))
    remaining=set(wanted)
    for index,board in selected_answers(source,wanted):
        assert board.shape==(81,)
        for seed,offset in wanted[index]:solutions[seed,offset]=board
        remaining.remove(index)
        if not remaining:break
    assert not remaining;entries=[]
    for seed in range(72):
        path=OUT/f'training-s{seed}.npz';np.savez_compressed(path,solutions=solutions[seed],source_row=draws[seed])
        entries.append(dict(seed=seed,role='training',completion_path=str(path.relative_to(ROOT)),completion_sha256=sha(path)))
    manifest=dict(protocol_sha256=sha(PROTOCOL),all72_full_masks_sealed_before_materialization=True,predraw_receipts_sha256=predraws,identity_seal_sha256=sha(OUT/'identity-seal.json'),selected_sources_sha256=sha(OUT/'selected_sources.npz'),inputs=entries)
    save(OUT/'input_manifest.json',manifest);save(OUT/'all_source_manifest.json',dict(manifest,purpose='Future exclusions must include all72fresh banks.'))


def reconstruct(mask,clean,noise,classes):
    assert mask.shape==clean.shape==(90,81) and mask.dtype==bool and clean.dtype==np.int8 and ((clean>=0)&(clean<=8)).all()
    validate_channels(classes,81);assert np.array_equal(np.bincount(classes,minlength=3),[27,27,27])
    threshold=np.array([float(Fraction(*RATES[int(c)])) for c in classes]);records=dict(given=mask)
    for name in ['first','repeat1','repeat2']:
        u,o=[noise[name+'_'+suffix] for suffix in ['iid_uniforms','iid_offsets']]
        assert u.shape==o.shape==clean.shape and u.dtype==np.float64 and o.dtype==np.int8
        assert ((u>=0)&(u<1)).all() and ((o>=1)&(o<=8)).all()
        labels=np.where(u<threshold,(clean+o)%9,clean).astype(np.int8);observed=np.full((90,81),-1,np.int8);observed[mask]=labels[mask];records[name]=observed
    return records


def arm_values(records):return np.stack([records[name] for name in ['first','repeat1','repeat2']]),records['given']


def prepare():
    from cell_channel_data import audit_queries
    p=verify(True);source_audit=read(OUT/'source-audit.json');assert source_audit['status']=='validated' and source_audit['protocol_sha256']==sha(PROTOCOL) and source_audit['input_manifest_sha256']==sha(OUT/'input_manifest.json')
    assert not (OUT/'query-record-seal.json').exists();manifest=read(OUT/'input_manifest.json');receipts={};audits={}
    for seed in range(72):
        folder=OUT/f's{seed}';mask=npz(folder/'masks.npz')['balanced'];classes=npz(folder/'cell-channels.npz')['classes'];entry=manifest['inputs'][seed];assert sha(ROOT/entry['completion_path'])==entry['completion_sha256']
        clean=npz(ROOT/entry['completion_path'])['solutions'];noise={}
        for name,key in [('first','noise_seed_base'),('repeat1','repeat1_seed_base'),('repeat2','repeat2_seed_base')]:
            rng=np.random.default_rng(p[key]+seed);noise[name+'_iid_uniforms']=rng.random((90,81));noise[name+'_iid_offsets']=rng.integers(1,9,(90,81),dtype=np.int8)
        np.savez_compressed(folder/'potential_noise.npz',**noise);np.savez_compressed(folder/'query-records.npz',**reconstruct(mask,clean,noise,classes))
        save(folder/'query-record-audit.json',audit_queries(seed,p));receipts[str(seed)]=sha(folder/'query-records.npz');audits[str(seed)]=sha(folder/'query-record-audit.json')
    save(OUT/'query-record-seal.json',dict(status='sealed',protocol_sha256=sha(PROTOCOL),all72_query_records_validated_before_scoring=True,record_sha256=receipts,audit_sha256=audits,source_audit_sha256=sha(OUT/'source-audit.json'),physical_noisy_queries=1399680))


def method_costs(values,given,arm,channel=None):
    assert arm in ARMS
    if arm=='estimated_scalar':
        assert channel is None
        estimator=estimate(values,given);assert estimator==independent_estimate(values,given) and estimator['total_comparisons']==19440
        if not estimator['fit_allowed']:return dict(estimator_unexecuted=np.array(True)),estimator,None,None
        channel=(estimator['rate_numerator'],estimator['rate_denominator']);producer,reference=estimated_score,estimated_reference
    else:
        validate_channels(channel,given.shape[1]);estimator=None;producer,reference=supplied_score,supplied_reference
    arrays=producer(values,given,channel);assert (arrays['joint_count']==71).all()
    try:ref=dict(certified=True,output=reference(values,given,channel),error=None)
    except UncertifiedCost as error:ref=dict(certified=False,output=None,error=str(error))
    if ref['certified']:
        expected,joint,_=ref['output'];assert np.array_equal(joint,arrays['joint_count']) and np.array_equal(expected[arrays['cost_certified']],arrays['cost'][arrays['cost_certified']])
    return arrays,estimator,numeric_state(arrays,ref),ref


def bank(seed,p):
    directory=OUT/f's{seed}';seal=read(OUT/'query-record-seal.json')
    assert seal['protocol_sha256']==sha(PROTOCOL) and seal['all72_query_records_validated_before_scoring'] and seal['record_sha256'][str(seed)]==sha(directory/'query-records.npz')
    values,given=arm_values(npz(directory/'query-records.npz'));classes=npz(directory/'cell-channels.npz')['classes'];references={};completed=0
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;channel=classes if arm=='supplied_per_cell' else None
        arrays,estimator,numeric,reference=method_costs(values,given,arm,channel);references[arm]=reference
        allowed=estimator is None or estimator['fit_allowed']
        if allowed:arrays,records=fit_models(arrays,numeric,index,p,directory,arm)
        else:records=[]
        np.savez_compressed(directory/f'{arm}-models.npz',**arrays)
        save(directory/f'{arm}-receipt.json',dict(index=index,seed=seed,arm=arm,protocol_sha256=sha(PROTOCOL),estimator=estimator,numeric=numeric,records=records,
            models_sha256=sha(directory/f'{arm}-models.npz'),query_records_sha256=sha(directory/'query-records.npz'),query_seal_sha256=sha(OUT/'query-record-seal.json'),
            predraw_receipt_sha256=sha(directory/'predraw-receipt.json'),unexecuted_grid_reason='inadmissible_channel_estimate' if not allowed else (None if numeric['fit_allowed'] else 'uncertified_producer_or_independent_log_cost_interval')))
        completed+=len(records);save(directory/'progress.json',dict(completed_LPs=completed,pid=os.getpid()))
    from audit_cell_channel import audit_bank
    save(directory/'audit.json',audit_bank(seed,False,p,references));print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def worker(worker_id):
    assert 0<=worker_id<8;p=verify()
    for seed in range(worker_id,72,8):bank(seed,p)


def summary():
    from audit_cell_channel import aggregate,accounting_total
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True);rows=[];files={};audits=[]
    for seed in range(72):
        folder=OUT/f's{seed}';audit=read(folder/'audit.json');assert audit['status']=='validated' and audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_cell_channel.py')
        for name,h in audit['artifact_sha256'].items():assert sha(folder/name)==h
        rows.extend(audit['rows']);audits.append(audit);dest=PUBLIC/f's{seed}';dest.mkdir()
        for name in ['audit.json']+list(audit['artifact_sha256']):shutil.copyfile(folder/name,dest/name);files[f's{seed}/{name}']=sha(dest/name)
    for name in ['selected_sources.npz','identity-seal.json','input_manifest.json','all_source_manifest.json','source-audit.json','query-record-seal.json']:
        shutil.copyfile(OUT/name,PUBLIC/name);files[name]=sha(PUBLIC/name)
    for name in ['excluded_ids.npz','ledger.json']:shutil.copyfile(REGISTRY/name,PUBLIC/name);files[name]=sha(PUBLIC/name)
    identity=read(OUT/'source-audit.json');result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],rows=rows,**aggregate(rows),source_audit=identity,
        query_accounting=accounting_total(audits,identity),attempted_LPs=sum(r['attempted_LPs'] for r in rows),new_certified_LPs=sum(r['certified_grid_LPs'] for r in rows),mutations_rejected=sum(r['mutations_rejected'] for r in rows),analytic_zero_proofs=sum(r['analytic_zero_proof'] for r in rows),artifact_sha256=files,all144_arms_retained=True)
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result);print(json.dumps(dict(endpoints=result['endpoints'],grouped=result['grouped'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','select','predraw-all','materialize','source-audit','prepare','worker','summary']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.action=='predraw-all':predraw_all()
    elif args.action=='source-audit':
        from cell_channel_data import sources
        sources()
    elif args.action=='worker':worker(args.seed)
    else:globals()[args.action]()
