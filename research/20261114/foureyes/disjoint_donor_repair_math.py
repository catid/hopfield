"""H4az audit-only repair: preserve compact integer labels during exact replay."""
import hashlib
import numpy as np


def counts(rows):
    assert rows.ndim==2 and np.issubdtype(rows.dtype,np.integer) and ((rows>=0)&(rows<=8)).all()
    i,j=np.triu_indices(rows.shape[1],1)
    return (rows[:,i]==rows[:,j]).sum(0,dtype=np.int64)


def independent_counts(rows):
    return np.array([np.count_nonzero(rows[:,i]==rows[:,j]) for i in range(rows.shape[1]) for j in range(i+1,rows.shape[1])],np.int64)


def noise_stream(hot_seed,noise_seed,rows=2400,n=81,k=9):
    hot=sorted(map(int,np.random.default_rng(hot_seed).choice(n,18,replace=False)))
    rng=np.random.default_rng(noise_seed);uniform=1-rng.random((rows,n));offsets=rng.integers(1,9,(rows,n),dtype=np.int64)
    weights=np.array([32 if i in hot else 1 for i in range(n)])
    priority=-np.log(uniform)/weights
    cells=np.array([[i for _,i in sorted((float(v),i) for i,v in enumerate(row))[:k]] for row in priority],np.int64)
    return uniform,offsets,cells


def validate_labels(rows,shape):
    assert isinstance(rows,np.ndarray) and rows.shape==shape
    assert rows.dtype.kind in 'iu' and ((rows>=0)&(rows<=8)).all()


def replay_noise(raw,hot_seed,noise_seed):
    clean=raw['solutions'];validate_labels(clean,(2400,81))
    validate_labels(raw['noisy'],clean.shape)
    assert raw['free'].dtype==bool and raw['free'].shape==clean.shape and raw['free'].all()
    uniform,offsets,cells=noise_stream(hot_seed,noise_seed)
    for key,expected in [('uniforms',uniform),('offsets',offsets),('cells',cells)]:
        assert raw[key].dtype==expected.dtype and np.array_equal(raw[key],expected),key
    noisy=clean.copy()
    for row in range(2400):
        for cell in cells[row]:noisy[row,cell]=(int(clean[row,cell])+int(offsets[row,cell]))%9
    assert raw['noisy'].dtype==noisy.dtype and np.array_equal(raw['noisy'],noisy)
    assert ((noisy!=clean).sum(1)==9).all()
    return noisy


def row_digests(rows):
    return [hashlib.sha256(np.asarray(row,dtype='<i8').tobytes()).hexdigest() for row in rows]


def duplicate_counts(target,donor):
    a,b=row_digests(target),row_digests(donor);known=set(a)
    return dict(target_duplicate_instances=len(a)-len(set(a)),donor_duplicate_instances=len(b)-len(set(b)),
        donor_rows_matching_any_target=sum(h in known for h in b),shared_unique_values=len(set(a)&set(b)))
