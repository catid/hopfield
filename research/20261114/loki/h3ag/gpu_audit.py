"""Dimension-15 adaptation of the independently checked H3ad saved-state audit."""
import hashlib
import numpy as np
from common import ROOT,SRC,OUT,read,sha
def audit(bank,kernel,seed,cap,m,data,keys,capacity):
    folder=OUT/f'b{bank}/search/{kernel}/s{seed}';r=read(folder/'result.json');assert r['status']=='complete'and r['cap']==cap and r['records_sha256']==sha(folder/'records.npz')
    assert r['model_sha256']==sha(OUT/f'b{bank}/model.npz')and r['source_sha256']==sha(SRC/'driver.py')and r['backend_sha256']==sha(SRC.parent/f'h3aa/{kernel}_backend.py')and r['calibration_selection_sha256']==sha(OUT/f'b{bank}/calibration-selection.json')and r['protocol_sha256']==sha(SRC/'protocol.json')
    assert kernel==('arbitrary'if(r['gpu']+seed)%2==0 else'permutation')
    path=OUT/f'b{bank}/initial/s{seed}.npz';initial=np.unpackbits(np.load(path)['boards'],axis=-1)[...,:225].astype(bool);assert r['initial_file_sha256']==sha(path)and r['initial_unpacked_bytes_sha256']==hashlib.sha256(initial.tobytes()).hexdigest()
    z=dict(np.load(folder/'records.npz'));boards=np.unpackbits(z['boards'],axis=-1)[...,:225].astype(bool);inputs=data['test_inputs'];assert np.array_equal(inputs,z['inputs'])and(boards>=inputs[:,None,:]).all()and(boards.sum(-1)==15).all()
    grid=boards.reshape(-1,15,15);perm=(grid.sum(1)==1).all(1)&(grid.sum(2)==1).all(1)
    if kernel=='permutation':assert perm.all()
    valid=np.array([[row.tobytes()in keys and row[given].all()for row in samples]for given,samples in zip(inputs,boards)]);assert np.array_equal(valid,z['valid']);stopped=z['stopped'];primary=(stopped&valid).any(1);assert np.array_equal(primary,z['certified_success'])and not(primary&(capacity==0)).any()
    original=z['energy'].argmin(1);selected=np.where(stopped.any(1),np.where(stopped,z['energy'],np.inf).argmin(1),original);ids=np.arange(len(inputs));assert np.array_equal(original,z['original_index'])and np.array_equal(selected,z['certificate_index'])and np.array_equal(valid[ids,original],z['n20'])and np.array_equal(valid[ids,selected],z['n20_certificate'])
    denom=np.minimum(data['test_solution_counts'],20);found=np.array([min(len({row.tobytes()for row in samples[ok]}),d)for samples,ok,d in zip(boards,valid,denom)]);assert np.array_equal(found,z['found'])and np.array_equal(denom,z['denom'])
    assert((z['moves']>=0)&(z['moves']<=cap)).all()and(z['moves'][~stopped]==cap).all()and((z['accepted']>=0)&(z['accepted']<=z['moves'])).all()
    assert r['proposal_moves']==int(z['moves'].sum())and r['accepted_swaps']==int(z['accepted'].sum())
    flat=boards.reshape(-1,225);J=m['J'].astype(np.float64);F=m['F'].astype(np.float64);energies=[];conflicts=[]
    for lo in range(0,len(flat),4096):
        x=flat[lo:lo+4096].astype(np.float64);energies.append((.5*x*(x@J)).sum(1));conflicts.append(((x*(x@F)).sum(1)/2).astype(np.int64))
    energy=np.concatenate(energies).reshape(stopped.shape);count=np.concatenate(conflicts).reshape(stopped.shape);assert np.array_equal(count,z['conflicts'])and(count[stopped]==0).all()and(energy[stopped]==0).all()and(count[~stopped]>0).all()
    wrong=int((stopped&~valid).sum());assert wrong==r['wrong_stops'];error=np.abs(energy-z['energy']);exact_original=energy.argmin(1);exact_selected=np.where(stopped.any(1),np.where(stopped,energy,np.inf).argmin(1),exact_original)
    assert r['certified_success_pct']==100*float(primary.mean())and r['original_accuracy_pct']==100*float(z['n20'].mean())and r['certificate_accuracy_pct']==100*float(z['n20_certificate'].mean())and r['coverage_pct']==100*float(found.sum()/denom.sum())
    row=dict(bank=bank,kernel=kernel,seed=seed,gpu=r['gpu'],cap=cap,unique_inputs=4096,chains=20,certified_valid_inputs=int(primary.sum()),original_correct=int(z['n20'].sum()),certificate_first_correct=int(z['n20_certificate'].sum()),coverage_numerator=int(found.sum()),coverage_denominator=int(denom.sum()),wrong_stops=wrong,capped_chains=int((~stopped).sum()),proposal_moves=int(z['moves'].sum()),accepted_swaps=int(z['accepted'].sum()),worker_solve_seconds=r['worker_solve_seconds'],warmup_seconds=r['warmup_seconds'],initial_file_sha256=r['initial_file_sha256'],initial_unpacked_bytes_sha256=r['initial_unpacked_bytes_sha256'],records_sha256=r['records_sha256'])
    diagnostic=dict(bank=bank,kernel=kernel,seed=seed,all_q15_and_givens_preserved=True,returned_permutation_states=int(perm.sum()),all_integer_conflict_counts_exact=True,wrong_stops=wrong,running_energy_absolute_error_max=float(error.max()),numerical_gate_passed=bool(error.max()<=.1),
        exact_energy_original_accuracy_pct=100*float(valid[ids,exact_original].mean()),exact_energy_certificate_accuracy_pct=100*float(valid[ids,exact_selected].mean()),exact_energy_certificate_count_change=int(valid[ids,exact_selected].sum())-int(z['n20_certificate'].sum()),exact_energy_original_count_change=int(valid[ids,exact_original].sum())-int(z['n20'].sum()),failed_certified_despite_capacity=int(((capacity>0)&~primary).sum()),fallback_success_without_capacity=int(((capacity==0)&z['n20_certificate']).sum()))
    return z,row,diagnostic

