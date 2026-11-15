"""Positive-count-specific likelihood and paired uniform source draws."""
import hashlib
import math
import numpy as np


def score(population,orbits,positives):return math.comb(55,orbits)*math.comb(population,positives) if population>=positives else None


def draw_pair(population,seed16,seed8):
    indices16=np.random.default_rng(seed16).choice(len(population),16,replace=False);subset8=np.random.default_rng(seed8).choice(16,8,replace=False);return indices16,subset8,population[indices16],population[indices16[subset8]]


def bank_hash(sample):return hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()
