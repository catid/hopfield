"""Uniform deflate-level1 NumPy archives; exact arrays, no pickle."""
from pathlib import Path
import zipfile
import numpy as np


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for key,value in arrays.items():
                with archive.open(key+'.npy','w',force_zip64=True) as target:np.lib.format.write_array(target,np.asanyarray(value),allow_pickle=False)
