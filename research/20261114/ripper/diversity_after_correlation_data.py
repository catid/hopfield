"""H1ci independent source, mask, threshold and acquired-query validation."""
import csv,json,hashlib
from fractions import Fraction
import numpy as np
from diversity_after_correlation_study import ROOT,HERE,OUT,PUBLIC,REGISTRY,PROTOCOL,PARENT,RATE,RHO,ARMS,sha,save,npz,verify,read
from audit_balanced_mask_math import reference_lines


def identity_audit(public=False):
    p=verify(public=public);base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2450858 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2214433 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-correlated-repeat.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1381136 and draws.shape==(72,270) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all72_full_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==72
    assert [e['seed'] for e in manifest['inputs']]==list(range(72))
    for seed in range(72):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,270,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==270
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2214433+len(unique),updated_combined_exclusion_union=2450858+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=1574640),draws,manifest


def validate_boards(boards):
    assert boards.shape==(270,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
    grids=boards.reshape(-1,9,9);boxes=grids.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    for digit in range(9):assert ((grids==digit).sum(1)==1).all() and ((grids==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];validate_boards(boards)
        for index,board in zip(draws[seed],boards,strict=True):
            index=int(index)
            if index in wanted:assert np.array_equal(wanted[index],board)
            else:wanted[index]=board
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for index,fields in enumerate(reader):
            if index not in remaining:continue
            board=np.array([int(c)-1 for c in fields[column]],np.int8);assert np.array_equal(board,wanted[index]);remaining.remove(index)
            if not remaining:break
    assert not remaining
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),identity_seal_sha256=sha(OUT/'identity-seal.json'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all72_full_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def reference_masks(seed):
    rng=np.random.default_rng(seed);lines=reference_lines();mask=np.ones((90,81),bool)
    permutation=list(map(int,rng.permutation(81)));order=list(map(int,rng.permutation(90)))
    for row,line in enumerate(order):
        for cell in lines[line]:mask[row,permutation[cell]]=False
    return mask


def literal_account(mask):
    assert mask.shape==(270,81)
    repeated={(row,int(cell),query) for row in range(90) for cell in np.flatnonzero(mask[row]) for query in range(3)}
    diverse={(row,int(cell),0) for row in range(270) for cell in np.flatnonzero(mask[row])}
    assert len(repeated)==len(diverse)==19440 and len(repeated&diverse)==6480 and len(repeated|diverse)==32400
    return dict(full_physical_noisy_union=len(repeated|diverse),supplied_correlated90_deployment=len(repeated),supplied_single270_deployment=len(diverse),shared_between_methods=len(repeated&diverse))


def channel_event(rate):
    threshold=float(Fraction(*rate));ratio=Fraction(threshold);scaled=ratio*2**53
    count=(scaled.numerator+scaled.denominator-1)//scaled.denominator
    return dict(nominal=list(rate),threshold_hex=threshold.hex(),binary64_ratio=[ratio.numerator,ratio.denominator],event_probability=[count,2**53])


def reference_corrupt(clean,uniforms,offsets,rate):
    assert clean.shape==uniforms.shape==offsets.shape==(90,81)
    assert clean.dtype==offsets.dtype==np.int8 and uniforms.dtype==np.float64
    assert ((clean>=0)&(clean<=8)).all() and ((uniforms>=0)&(uniforms<1)).all() and ((offsets>=1)&(offsets<=8)).all()
    result=np.empty_like(clean);threshold=float(Fraction(*rate))
    for row in range(90):
        for cell in range(81):
            result[row,cell]=(int(clean[row,cell])+int(offsets[row,cell]))%9 if float(uniforms[row,cell])<threshold else int(clean[row,cell])
    return result


def audit_queries(seed,p):
    from diversity_after_correlation_study import reconstruct,arm_values
    folder=OUT/f's{seed}';entry=read(OUT/'input_manifest.json')['inputs'][seed]
    assert sha(ROOT/entry['completion_path'])==entry['completion_sha256']
    clean=npz(ROOT/entry['completion_path'])['solutions'];mask=npz(folder/'masks.npz')['balanced'];noise=npz(folder/'potential_noise.npz');records=npz(folder/'query-records.npz')
    expected_masks=np.concatenate([reference_masks(base+seed) for base in p['balanced_seed_bases']])
    assert np.array_equal(mask,expected_masks) and len(noise)==11
    assert set(records)=={'given','first','repeat1','repeat2','diversity1','diversity2'} and np.array_equal(records['given'],mask)
    rate=RATE;diagnostics={};potentials={}
    branch=noise['shared_uniforms'];expected_branch=np.random.default_rng(p['shared_branch_seed_base']+seed).random((90,81))
    assert branch.dtype==np.float64 and branch.shape==(90,81) and np.array_equal(branch,expected_branch)
    shared=branch<float(Fraction(*RHO))
    groups=[('first','noise_seed_base',0),('repeat1','repeat1_seed_base',0),('repeat2','repeat2_seed_base',0),('diversity1','diversity1_seed_base',1),('diversity2','diversity2_seed_base',2)]
    for name,key,block in groups:
        rng=np.random.default_rng(p[key]+seed);u=noise[name+'_iid_uniforms'];o=noise[name+'_iid_offsets']
        assert np.array_equal(rng.random((90,81)),u) and np.array_equal(rng.integers(1,9,(90,81),dtype=np.int8),o)
        block_clean=clean[90*block:90*(block+1)];given=mask[90*block:90*(block+1)]
        potentials[name]=reference_corrupt(block_clean,u,o,rate);labels=np.empty((90,81),np.int8)
        for row in range(90):
            for cell in range(81):labels[row,cell]=potentials['first'][row,cell] if name in ['repeat1','repeat2'] and bool(shared[row,cell]) else potentials[name][row,cell]
        observed=records[name]
        assert observed.dtype==np.int8 and observed.shape==(90,81) and (observed[~given]==-1).all()
        assert np.array_equal(observed[given],labels[given])
        bad=labels!=block_clean;diagnostics[name]=dict(queried_cells=int(given.sum()),queried_errors=int(bad[given].sum()),full_cell_errors=int(bad.sum()))
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=clean).all()
    altered=reconstruct(mask,changed,noise)
    for arm in ARMS:
        before,given=arm_values(records,arm);after,given2=arm_values(altered,arm)
        assert np.array_equal(given,given2) and (before[:,given]!=after[:,given]).all()
    repeated,rg=arm_values(records,ARMS[0]);diverse,dg=arm_values(records,ARMS[1]);assert np.array_equal(repeated[0],diverse[0,:90]) and np.array_equal(rg,dg[:90])
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
        query_records_sha256=sha(folder/'query-records.npz'),completion_sha256=entry['completion_sha256'],masks_sha256=sha(folder/'masks.npz'),noise_sha256=sha(folder/'potential_noise.npz'),
        fresh_raw_correspondence_verified=True,new_queries=32400,scorer_receives_clean_values=False,
        complete_label_substitution=True,literal_shared_first90_response_and_mask=True,channel=channel_event(rate),shared_branch_event=channel_event(RHO),realized_noise=diagnostics)

