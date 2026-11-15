"""Exact conditioned hierarchical prior; no floating-point sampling weights."""
import itertools
import math
import numpy as np


def prior_rows(cardinalities):
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];assert len(cardinalities)==len(choices)==1541
    return [dict(candidate=cid,orbits=[list(orbits[j]) for j in chosen],source_population=int(M),eligible=int(M)>=8,prior_integer_weight=1485//math.comb(55,len(chosen)) if int(M)>=8 else 0) for cid,(chosen,M) in enumerate(zip(choices,cardinalities))]


def truth_from_ticket(rows,ticket):
    cumulative=np.cumsum([r['prior_integer_weight'] for r in rows],dtype=np.int64);assert 0<=ticket<int(cumulative[-1]);index=int(np.searchsorted(cumulative,ticket,side='right'));assert rows[index]['eligible'];return rows[index]['candidate']


def draw_truth(rows,seed):
    mass=sum(r['prior_integer_weight'] for r in rows);ticket=int(np.random.default_rng(seed).integers(mass));return truth_from_ticket(rows,ticket),ticket
