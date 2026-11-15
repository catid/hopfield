"""Hand-enumerated articulation motifs, fixed before H2ap trajectories."""
import numpy as np


def motifs():
    backbone=[(8,c) for c in range(9)]
    fork=[(7,4),(6,4),(5,4),(5,3),(5,2),(5,5),(5,6)]
    cycle=[(7,4),(6,4),(5,4),(4,4),(3,4),(3,5),(4,5)]
    two=[(7,2),(6,2),(7,6),(6,6)]
    specifications=[('fork',fork,[fork,fork[1:],fork[2:],fork[3:5],fork[5:]],[(8,4),(7,4),(6,4),(5,4),(5,4)]),('cycle',cycle,[cycle,cycle[1:],cycle[2:],cycle[3:],cycle[4:]],[(8,4),(7,4),(6,4),(5,4),(4,4)]),('multiple',two,[two[:2],two[2:]],[(8,2),(8,6)])]
    out=[]
    for name,branch,groups,cuts in specifications:
        t=np.zeros((9,9),np.uint8)
        for x in backbone+branch:t[x]=1
        t[8,0]=2;t[8,8]=3
        for rotation in range(4):
            rt=np.rot90(t,rotation).copy();expected=[]
            for cells,cut in zip(groups,cuts):
                m=np.zeros_like(t,bool)
                for x in cells:m[x]=True
                c=np.zeros_like(t,bool);c[cut]=True
                expected.append({'cells':list(map(int,np.flatnonzero(np.rot90(m,rotation)))),'articulations':[int(np.flatnonzero(np.rot90(c,rotation))[0])]})
            expected.sort(key=lambda r:r['cells']);out.append({'name':f'{name}-rotation{rotation}','tokens':rt,'board':rt>0,'expected_witnesses':expected,'expected_candidate_count':len(groups),'expected_duplicate_count':0})
    return out
