"""Label-independent 72-cell masks from affine lines over GF(9)."""
import numpy as np


def add(x,y):return (x%3+y%3)%3+3*((x//3+y//3)%3)


def multiply(x,y):
    a,b=x%3,x//3;c,d=y%3,y//3
    return (a*c+2*b*d)%3+3*((a*d+b*c)%3)


def line_complements():
    masks=np.ones((90,81),bool)
    for slope in range(9):
        for intercept in range(9):
            for x in range(9):masks[9*slope+intercept,9*x+add(multiply(slope,x),intercept)]=False
    for x in range(9):masks[81+x,9*x:9*x+9]=False
    return masks


def make_balanced_masks(seed):
    rng=np.random.default_rng(seed);template=line_complements();parts=[]
    for cycle in range(2):
        cell_order=rng.permutation(81);line_order=rng.permutation(90);part=np.zeros((90,81),bool);part[:,cell_order]=template[line_order];parts.append(part)
    return np.concatenate(parts,axis=0)
