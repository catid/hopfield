"""Synthetic H4be driver completion, owned process termination and late failure."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import run_repeat_diversity as runner


class Completed:
    pid=-1
    def __init__(self,code):self.returncode=code
    def poll(self):return self.returncode


def gate():
    cases=[];original_popen=subprocess.Popen
    for mode in ["complete","worker_failure","owned_child_budget","late_export_failure","summary_write_failure"]:
        with tempfile.TemporaryDirectory(prefix="h4be-driver-gate-") as directory:
            root=Path(directory);out=root/"out";public=root/"public";protocol=root/"protocol.json";protocol.write_text('{}\n')
            children=[];seals=[]
            real_save,real_copyfile=runner.save,runner.shutil.copyfile
            def save(path,value):
                real_save(path,value)
                if mode=="summary_write_failure" and Path(path)==public/"summary.json":raise OSError("synthetic_summary_failure")
            def copyfile(source,target,**kwargs):
                if mode=="late_export_failure" and Path(source).name=="source-seal.json":raise OSError("synthetic_export_failure")
                return real_copyfile(source,target,**kwargs)
            def seal(name):
                assert name==["source","fit"][len(seals)]
                assert not list((out/"audits").glob("*.json"))
                seals.append(name);real_save(out/f"{name}-seal.json",dict(synthetic=True))
            def popen(args,**kwargs):
                assert kwargs["start_new_session"] is True
                script=Path(args[1]).name;worker=int(args[-1])
                phase="source" if script=="repeat_diversity_source.py" else "fit" if script=="repeat_diversity_fit.py" else "audit"
                if mode=="owned_child_budget":child=original_popen([str(runner.CPU_PYTHON),"-c","import time; time.sleep(60)"],**kwargs)
                else:child=Completed(int(mode=="worker_failure"))
                children.append(child)
                if mode=="owned_child_budget" and len(children)==8:p["budget"]["aggregate_CPU_seconds"]=0
                if mode in ["complete","late_export_failure","summary_write_failure"]:
                    assert len(seals)=={"source":0,"fit":1,"audit":2}[phase]
                    for i in range(worker,256,8):
                        target=out/("audits" if phase=="audit" else "source_audit" if phase=="source" else "fits")/f"b{i:03d}.json"
                        real_save(target,dict(index=i,synthetic=True))
                    real_save(out/"progress"/f"{phase}{worker}.json",dict(completed_banks=32))
                return child
            p=dict(budget=dict(complete_wall_seconds=120,aggregate_CPU_seconds=600),accounting={})
            with contextlib.ExitStack() as stack:
                for key,value in dict(OUT=out,PUBLIC=public,PROTOCOL=protocol,START=runner.time.monotonic()).items():stack.enter_context(patch.object(runner,key,value))
                stack.enter_context(patch.object(runner,"protocol",return_value=p))
                stack.enter_context(patch.object(runner,"seal",side_effect=seal))
                stack.enter_context(patch.object(runner,"aggregate",side_effect=lambda rows:dict(synthetic_banks=len(rows))))
                stack.enter_context(patch.object(runner.subprocess,"Popen",side_effect=popen))
                stack.enter_context(patch.object(runner.shutil,"copyfile",side_effect=copyfile))
                stack.enter_context(patch.object(runner,"save",side_effect=save))
                with contextlib.redirect_stdout(io.StringIO()):
                    try:runner.launch()
                    except AssertionError as error:
                        assert mode in ["worker_failure","owned_child_budget"]
                        assert str(error)==("CPU_budget" if mode=="owned_child_budget" else "owned_worker_failure")
                    except OSError as error:
                        assert mode in ["late_export_failure","summary_write_failure"]
                        assert str(error)==("synthetic_export_failure" if mode=="late_export_failure" else "synthetic_summary_failure")
                    else:assert mode=="complete"
            runtime=json.loads((out/"runtime.json").read_text())
            assert runtime["status"]==("complete" if mode=="complete" else "incomplete")
            assert all(c.poll() is not None for c in children)
            if mode=="owned_child_budget":assert len(children)==8 and all(c.returncode!=0 for c in children)
            if mode=="complete":assert len(children)==24 and runtime["returncodes"]==[0]*24 and len(seals)==2
            else:
                assert not (out/"summary.json").exists() and not (public/"summary.json").exists()
                if mode in ["late_export_failure","summary_write_failure"]:assert len(list((out/"audits").glob("*.json")))==256
            cases.append(dict(mode=mode,status="passed",owned_children=len(children),phases_sealed=seals))
    return cases


if __name__=="__main__":print(json.dumps(gate(),indent=2))
