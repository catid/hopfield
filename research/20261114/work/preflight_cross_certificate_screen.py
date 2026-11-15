"""Synthetic fixed-threshold, finite-value and shape gates, no source labels."""
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import numpy as np
from cross_certificate_screen_policy import screen


def main():
    margins=[-math.inf,math.nan,math.inf,-1e-4,np.nextafter(-1e-6,-math.inf),-1e-6,np.nextafter(-1e-6,math.inf),0.,1.]
    cases=[(f,b,l) for f in itertools.product([False,True],repeat=4) for b in margins for l in margins]
    flags=np.array([r[0] for r in cases],bool);bp=np.array([r[1] for r in cases]);lp=np.array([r[2] for r in cases]);got=screen(*(flags[:,i] for i in range(4)),bp,lp)
    expected=[]
    for f,b,l in cases:
        own=all(f) and math.isfinite(b) and b>=-1e-6;cross=f[0] and f[3] and math.isfinite(l) and l>=-1e-6;expected.append((own,cross,own and cross))
    assert all(np.array_equal(x,np.asarray(expected)[:,i]) for i,x in enumerate(got))
    assert list(inspect.signature(screen).parameters)==['hidden_good','finite','stopped','clamps_ok','bp_margin','lp_margin']
    try:screen(flags[:,0],flags[:1,1],flags[:,2],flags[:,3],bp,lp)
    except ValueError:pass
    else:raise AssertionError('Unexpected shape broadcasting')
    root=Path(__file__).resolve().parents[3];target=root/'artifacts/research/20261114/work/cross_certificate_screen';target.mkdir(exist_ok=True)
    r={'status':'passed','synthetic_cases':len(cases),'nonfinite_and_adjacent_ulp_boundaries':True,'label_free_signature':True,'shape_guard':True,
       'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['cross_certificate_screen_policy.py','preflight_cross_certificate_screen.py']},
       'scope':'Synthetic numerical acceptance only; no source graph labels or new source margins.'}
    (target/'synthetic-gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
