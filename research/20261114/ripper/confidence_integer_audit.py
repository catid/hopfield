"""Independent scaled-integer fields, energy, focus and certificate checks."""
import numpy as np


def check_state(search,weight16):
    assert weight16.shape==(3240,) and set(np.unique(weight16))<={0,1,16}
    pair=np.triu_indices(81,1);matrix=np.zeros((81,81),np.int64)
    matrix[pair]=weight16;matrix[(pair[1],pair[0])]=weight16
    y=search.y.cpu().numpy().astype(np.int64);fr=search.fr.cpu().numpy();n=len(y)
    if n==0:return 0
    assert np.all((0<=y)&(y<9))
    fields=np.stack([(y==digit).astype(np.int64)@matrix.T for digit in range(9)],axis=2)
    energy=((y[:,pair[0]]==y[:,pair[1]])*weight16.astype(np.int64)).sum(1)
    assert np.array_equal(search.G.cpu().numpy()*16,fields)
    assert np.array_equal(search.P.cpu().numpy()*16,energy)
    own=np.take_along_axis(fields,y[:,:,None],axis=2).squeeze(2)
    assert search.theta==.45 and search.ebar==search.tau==0 and search.tol==.001
    assert np.array_equal(search.flag.cpu().numpy(),((own>=8)&fr).astype(np.int32))
    clues=search.x.cpu().numpy()-1;assert np.array_equal(y[~fr],clues[~fr])
    found=search.found.cpu().numpy()>=0;cert=search.ycert.cpu().numpy()[found]
    cert_energy=((cert[:,pair[0]]==cert[:,pair[1]])*weight16.astype(np.int64)).sum(1)
    assert not cert_energy.any()
    cert_free=fr[found];cert_clues=clues[found]
    assert np.array_equal(cert[~cert_free],cert_clues[~cert_free])
    return n
