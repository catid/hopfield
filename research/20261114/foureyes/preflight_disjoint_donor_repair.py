"""Synthetic compact-dtype compatibility gates; no study outcomes are opened."""
from pathlib import Path
import time
import numpy as np
import disjoint_donor_repair as S
from disjoint_donor_math import counts,independent_counts,replay_noise as original_replay
from disjoint_donor_repair_math import replay_noise,validate_labels


def rejected(fn,*args):
    try:fn(*args)
    except AssertionError:return
    raise AssertionError('Invalid synthetic case accepted')


def main():
    gate=S.HERE/'disjoint-donor-repair-preflight.json'
    assert not gate.exists() and not S.REPAIR_PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    start=time.monotonic();rng=np.random.default_rng(982094);clean=rng.integers(0,9,(2400,81),dtype=np.int64)
    hs,ns=982095,982096;g=np.random.default_rng(ns);u=1-g.random(clean.shape);offset=g.integers(1,9,clean.shape,dtype=np.int64)
    weights=np.ones(81);weights[np.random.default_rng(hs).choice(81,18,replace=False)]=32
    cells=np.argsort(-np.log(u)/weights,axis=1,kind='stable')[:,:9]
    expected=clean.copy();np.put_along_axis(expected,cells,(np.take_along_axis(clean,cells,1)+np.take_along_axis(offset,cells,1))%9,1)
    base=dict(solutions=clean,free=np.ones(clean.shape,bool),uniforms=u,offsets=offset,cells=cells,noisy=expected)
    assert np.array_equal(original_replay(base,hs,ns),expected)
    accepted=[];badcases=0
    for dtype in [np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64]:
        raw=dict(base,solutions=clean.astype(dtype),noisy=expected.astype(dtype))
        actual=replay_noise(raw,hs,ns);assert actual.dtype==dtype and np.array_equal(actual,expected)
        validate_labels(actual[600:1200],(600,81))
        assert np.array_equal(counts(actual[600:1200]),independent_counts(expected[600:1200]))
        accepted.append(str(np.dtype(dtype)))
        bad=dict(raw,noisy=raw['noisy'].astype(np.uint16 if dtype!=np.uint16 else np.int16))
        rejected(replay_noise,bad,hs,ns);badcases+=1
    for dtype in [bool,np.float32,np.float64,object]:
        rejected(validate_labels,clean.astype(dtype),(2400,81));badcases+=1
    for value in [-1,9,255]:
        altered=clean.copy();altered[0,0]=value;rejected(validate_labels,altered,(2400,81));badcases+=1
    rejected(validate_labels,clean,(600,81));badcases+=1
    for name,dtype in [('free',np.uint8),('uniforms',np.float32),('offsets',np.int32),('cells',np.int32)]:
        rejected(replay_noise,dict(base,**{name:base[name].astype(dtype)}),hs,ns);badcases+=1
    for name in ['solutions','noisy']:
        altered=base[name].copy();altered[0,0]=(altered[0,0]+1)%9
        rejected(replay_noise,dict(base,**{name:altered}),hs,ns);badcases+=1
    # Freeze source bindings only. The original phase checker opens receipts/hashes,
    # never the stored graph arrays; classifications are deferred until repair freeze.
    original=S.read(S.PROTOCOL);S.verify(original);S.check_seal(S.ORIGINAL)
    sources=original['source_sha256'].copy()
    for name in ['disjoint_donor_repair.py','disjoint_donor_repair_math.py','audit_disjoint_donor_repair.py','preflight_disjoint_donor_repair.py']:
        path=S.HERE/name;sources[str(path.relative_to(S.ROOT))]=S.sha(path)
    receipt=dict(status='passed',study_graph_outcomes_read=False,accepted_integer_formats=accepted,rejected_cases=badcases,
                 compact_dtype_preserved=True,independent_producer_noise_parity=True,donor_counts_parity=True,original_int64_behavior_preserved=True,
                 original_sealed_outputs_hash_checked=True,original_sources_sha256_checked=True,
                 proposal_sha256=S.sha(S.HERE/'disjoint-donor-repair-proposal.json'),source_sha256=sources,wall_seconds=time.monotonic()-start)
    S.save(gate,receipt);print('REPAIR_PREFLIGHT_PASSED',accepted,badcases,flush=True)


if __name__=='__main__':main()
