"""Independent brute cut-vertex definition for pendant-component candidates."""
import numpy as np


def geometry(tokens,board):
    h,w=board.shape;selected=set(map(int,np.flatnonzero(board)));endpoints=set(map(int,np.flatnonzero(tokens>=2)))
    def neighbors(u):
        r,c=divmod(u,w)
        return ([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else [])
    def parts(vertices):
        unseen=set(vertices);out=[]
        while unseen:
            first=min(unseen);unseen.remove(first);part={first};queue=[first]
            while queue:
                for v in neighbors(queue.pop()):
                    if v in unseen:unseen.remove(v);part.add(v);queue.append(v)
            out.append(part)
        return out
    components=parts(selected);component_of={u:c for c in components for u in c};cuts=[];raw=[];witness={};oversize=singles=protected=0
    for cut in sorted(selected):
        pieces=parts(component_of[cut]-{cut})
        if len(pieces)<2:continue
        cuts.append(cut)
        for part in pieces:
            cells=tuple(sorted(part))
            if endpoints.intersection(part):protected+=1;continue
            if len(part)==1:singles+=1;continue
            if len(part)>256:oversize+=1;continue
            raw.append(cells);witness.setdefault(cells,[]).append(cut)
    unique=sorted(witness);return unique,{'articulation_vertices':cuts,'raw_component_candidates':len(raw),'unique_component_candidates':len(unique),'duplicate_component_candidates':len(raw)-len(unique),'oversize_components':oversize,'singletons':singles,'endpoint_components_excluded':protected,'witnesses':[{'cells':list(c),'articulations':sorted(witness[c])} for c in unique]}
