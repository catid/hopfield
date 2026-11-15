"""Pre-freeze synthetic full-receipt JSON gate and exact scientific-code diff."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[4];S=R/'research/20261114/loki/h3aq';O=R/'artifacts/research/20261114/loki/h3aq'
def module(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def inner():
    import numpy as np
    oldpath=S.parent/'h3ap/construct.py';newpath=S/'construct.py';old=module(oldpath,'old_construct');new=module(newpath,'new_construct');oldtext=oldpath.read_text();newtext=newpath.read_text();a=next(n for n in ast.parse(oldtext).body if isinstance(n,ast.FunctionDef)and n.name=='construct');b=next(n for n in ast.parse(newtext).body if isinstance(n,ast.FunctionDef)and n.name=='construct');assert ast.dump(ast.Module(body=a.body[:-1],type_ignores=[]))==ast.dump(ast.Module(body=b.body[:-1],type_ignores=[]));assert ast.dump(a.body[-1].value.elts[0])==ast.dump(b.body[-1].value.elts[0]);fields={'observed_finite_offsets','retained_zero_offsets'}
    for node in(a,b):node.body[-1].value.elts[1].keywords=[k for k in node.body[-1].value.elts[1].keywords if k.arg not in fields]
    assert ast.dump(a)==ast.dump(b)
    for name in('source.py','study.py','reverse_literal.cpp'):assert(S/name).read_bytes()==(S.parent/'h3ap'/name).read_bytes()
    boards=np.zeros((16,144),bool)
    for i in range(16):boards[i,12*np.arange(12)+(np.arange(12)+i)%12]=True
    offsets=np.array([(r,c)for r in range(12)for c in range(-11,12)if r>0 or c>0],np.int64);lines=np.array([(0,1),(1,-1),(1,0),(1,1)],np.int64);cases=[]
    for augment in(False,True):
        F,receipt=new.construct(boards,offsets,lines,augment);oldF,oldreceipt=old.construct(boards,offsets,lines,augment);assert F.dtype==oldF.dtype and F.shape==oldF.shape and F.tobytes()==oldF.tobytes();full=dict(index=0,family=0,arm='D4_unfiltered'if augment else'raw_unfiltered',group=0,**receipt,model_sha256='synthetic');encoded=json.dumps(full,allow_nan=False);decoded=json.loads(encoded);assert decoded['raw_labels']==16 and decoded['attempted_views']==(128 if augment else 16)
        try:json.dumps(dict(index=0,family=0,arm=full['arm'],group=0,**oldreceipt,model_sha256='synthetic'));old_failed=False
        except TypeError:old_failed=True
        assert old_failed;cases.append(dict(augment=augment,full_receipt_JSON_roundtrip=True,predecessor_failure_reproduced=True,F_dtype_shape_and_bytes_identical=True))
    return dict(status='passed',synthetic_only_no_real_source_or_target_reads=True,cases=cases,AST_computation_and_nonoffset_metadata_identical=True,unchanged_files=['source.py','study.py','reverse_literal.cpp'],exact_constructor_text_diff=''.join(difflib.unified_diff(oldtext.splitlines(keepends=True),newtext.splitlines(keepends=True),fromfile='h3ap/construct.py',tofile='h3aq/construct.py')))
def main():
    O.mkdir(parents=True,exist_ok=True)
    if '--inner'in sys.argv:print(json.dumps(inner()));return
    assert not(S/'protocol.json').exists()and not(O/'preflight.json').exists();start=time.perf_counter();child=subprocess.run([sys.executable,str(Path(__file__)),'--inner'],capture_output=True,text=True,check=True);wall=time.perf_counter()-start;result=json.loads(child.stdout);a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);result.update(measured_CPU_seconds=a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime,measured_child_wall_seconds=wall,charged_wall_seconds=wall+.1,wall_charge_note='Whole child elapsed plus conservative0.1s launcher/startup allowance',preflight_before_successor_protocol=True);(O/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
