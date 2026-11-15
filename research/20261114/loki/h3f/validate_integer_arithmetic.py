"""Reproduce the bounded integer-arithmetic check underlying the conditional proof."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts/research/20261114/loki/h3f'
PUBLIC=ROOT/'results/research/loki/h3f'


def main():
    path=OUT/'model.pt';model=torch.load(path,weights_only=False);J=model['S'][:,:,1,1].numpy()
    assert set(np.unique(J))=={-1.,0.,11.} and model['delta']==11.
    rng=np.random.default_rng(38112);x=rng.integers(0,2,144).astype(np.float32)
    fields=x@J;energy=np.float32(.5*x@fields)
    for step in range(20000):
        cell=int(rng.integers(144));old=x[cell];new=np.float32(1-old);local=fields[cell]
        energy=np.float32(np.float32(energy+new*local)-old*local)
        fields=np.float32(np.float32(fields+new*J[cell])-old*J[cell]);x[cell]=new
        if step%100==0:
            direct=x.astype(np.int64)@J.astype(np.int64)
            assert np.array_equal(fields,direct)
            assert energy==int(x.astype(np.int64)@direct)//2
    threshold=np.float32(float(model['delta'])-.001);costs=np.arange(-2000,2000)
    assert 10<threshold<11 and np.array_equal(costs.astype(np.float32)>=threshold,costs>=11)
    report=dict(status='passed',random_single_cell_flips=20000,checks_every=100,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        threshold_float32=float(threshold),integer_pass_condition='flip cost >=11',unit_only_in_temperature=True,
        conservative_bounds=dict(field_abs=11*143,energy_abs=11*(144*143//2),energy_update_intermediate_abs=11*(144*143//2)+11*143),
        float32_exact_integer_limit=2**24,scope='Synthetic state arithmetic check plus analytical bounds; no task accuracy or convergence measurement.')
    for folder in [OUT,PUBLIC]:(folder/'integer-arithmetic-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
