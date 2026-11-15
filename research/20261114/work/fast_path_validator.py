"""Compiled exact simple-path validator, independent of shortestness labels."""
import ctypes
import numpy as np
import study as S

LIBRARY=S.OUT/"selective_fallback/path_validator.so"


class Validator:
    def __init__(self,path=LIBRARY):
        self.lib=ctypes.CDLL(str(path));self.fun=self.lib.paths_valid
        pointer=ctypes.POINTER(ctypes.c_uint8)
        self.fun.argtypes=[pointer,pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int,pointer];self.fun.restype=None

    def __call__(self,tokens,predictions):
        t=np.ascontiguousarray(tokens,dtype=np.uint8);p=np.ascontiguousarray(predictions,dtype=np.uint8)
        assert t.shape==p.shape and t.ndim==3
        out=np.empty(len(t),np.uint8);pointer=ctypes.POINTER(ctypes.c_uint8)
        self.fun(t.ctypes.data_as(pointer),p.ctypes.data_as(pointer),*t.shape,out.ctypes.data_as(pointer))
        return out.astype(bool)
