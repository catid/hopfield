"""H3y source-only helpers; no target labels or target geometry."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'research/20261114/loki/h3y'
OUT = ROOT/'artifacts/research/20261114/loki/h3y'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p): return json.loads(Path(p).read_text())
def line_enumeration(n,lines):
    used=[set()for _ in lines];placed=[];answers=[]
    def visit(r):
        if r==n:answers.append(placed.copy());return
        for c in range(n):
            values=[a*r+b*c for a,b in lines]
            if any(v in s for v,s in zip(values,used)):continue
            for v,s in zip(values,used):s.add(v)
            placed.append(c);visit(r+1);placed.pop()
            for v,s in zip(values,used):s.remove(v)
    visit(0);return np.array(answers,np.uint8).reshape(-1,n)
def bit_enumeration(n,lines,leapers=()):
    attacks={(r,c):[sum(1<<d for d in range(n)if any(a*(t-r)+b*(d-c)==0 for a,b in lines)or tuple(sorted((abs(t-r),abs(d-c))))in leapers)for t in range(n)]for r in range(n)for c in range(n)}
    answers=[];full=(1<<n)-1
    def visit(r,blocked,cols):
        if r==n:answers.append(cols);return
        choices=full&~blocked[r]
        while choices:
            bit=choices&-choices;choices^=bit;c=bit.bit_length()-1;next_blocked=blocked[:]
            for t in range(r+1,n):next_blocked[t]|=attacks[r,c][t]
            visit(r+1,next_blocked,cols+[c])
    visit(0,[0]*n,[]);return np.array(answers,np.uint8).reshape(-1,n)
def counts_and_lines(columns,n):
    counts={(r,c):0 for r in range(n)for c in range(1-n,n)if r>0 or c>0}
    for cols in columns.astype(int):
        for r,t in itertools.combinations(range(n),2):counts[t-r,int(cols[t]-cols[r])]+=1
    lines=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
        members=[v for(r,c),v in counts.items()if a*r+b*c==0]
        if members.count(0)>=3 and not any(members):lines.append((a,b))
    return counts,lines
def candidate_orbits(counts,lines):
    zeros={o for o,v in counts.items()if v==0 and not any(a*o[0]+b*o[1]==0 for a,b in lines)}
    generators=sorted({tuple(sorted(map(abs,o)))for o in zeros});orbits=[]
    for u,v in generators:
        orbit=sorted({(r,c)for r,c in [(u,v),(u,-v),(v,u),(v,-u)]})
        assert set(orbit)<=zeros;orbits.append(orbit)
    assert set().union(*(set(o)for o in orbits))==zeros
    return generators,orbits
def bits(a):return int.from_bytes(np.packbits(a,bitorder='little').tobytes(),'little')
