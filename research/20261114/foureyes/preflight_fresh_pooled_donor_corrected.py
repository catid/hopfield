"""Synthetic lifecycle gates for the correction-only audit driver; no cohort reads."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import patch
import audit_fresh_pooled_donor_corrected as auditor
import run_fresh_pooled_donor_corrected as runner


class Completed:
    pid = -1

    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


def gate():
    started = time.monotonic()
    old = (runner.HERE/"audit_fresh_pooled_donor.py").read_text()
    needle = 'assert env["numpy"] == "2.3.4" and env["physical_GPU"]'
    assert old.count(needle) == 1
    assert old.replace(needle, needle.replace("2.3.4", "2.5.4")) == (runner.HERE/"audit_fresh_pooled_donor_corrected.py").read_text()
    cases = []
    original_popen = subprocess.Popen
    for mode in ["complete", "worker_failure", "owned_child_budget", "late_export_failure", "summary_write_failure"]:
        with tempfile.TemporaryDirectory(prefix="h4bd-synthetic-") as folder:
            root = Path(folder)
            parent, out, public = [root/name for name in ["parent", "out", "public"]]
            parent.mkdir()
            for name in ["fits", "observed", "source_audit", "raw"]:
                (parent/name).mkdir()
                (parent/name/"synthetic.json").write_text('{}\n')
            for name in ["orders.npz", "orders-seal.json", "source-seal.json", "grid-seal.json", "donor-seal.json", "final-seal.json"]:
                (parent/name).write_text('synthetic fixture\n')
            (parent/"runtime.json").write_text('{"status":"incomplete"}\n')
            protocol = root/"protocol.json"
            protocol.write_text('{}\n')
            original = {str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
            children = []
            checks = []
            real_copyfile, real_save = runner.shutil.copyfile, runner.save

            def copyfile(source, target, **kwargs):
                if mode == "late_export_failure" and Path(source).name == "orders-seal.json":
                    raise OSError("synthetic_export_failure")
                return real_copyfile(source, target, **kwargs)

            def save(path, value):
                real_save(path, value)
                if mode == "summary_write_failure" and Path(path) == public/"summary.json":
                    raise OSError("synthetic_summary_failure")

            def popen(args, **kwargs):
                assert kwargs["start_new_session"] is True
                if mode == "owned_child_budget":
                    child = original_popen([str(runner.CPU_PYTHON), "-c", "import time; time.sleep(60)"], **kwargs)
                else:
                    child = Completed(int(mode == "worker_failure"))
                children.append(child)
                if mode in ["complete", "late_export_failure", "summary_write_failure"]:
                    worker = int(args[-1])
                    for i in range(worker, 256, 8):
                        runner.save(out/"audits"/f"b{i:03d}.json", {"index":i,"synthetic":True})
                    runner.save(out/"progress"/f"worker{worker}.json", {"completed_banks":32})
                return child

            p = {"budget":{"wall_seconds":120,"aggregate_CPU_seconds":0 if mode == "owned_child_budget" else 600}}
            with contextlib.ExitStack() as stack:
                for name, value in dict(PARENT=parent,OUT=out,PUBLIC=public,CORRECTED=protocol,PROTOCOL=protocol).items():
                    stack.enter_context(patch.object(runner, name, value))
                stack.enter_context(patch.object(runner,"corrected_protocol",return_value=p))
                stack.enter_context(patch.object(runner,"validate_sources",side_effect=lambda:checks.append(True)))
                stack.enter_context(patch.object(runner.subprocess,"Popen",side_effect=popen))
                stack.enter_context(patch.object(runner.shutil,"copyfile",side_effect=copyfile))
                stack.enter_context(patch.object(runner,"save",side_effect=save))
                stack.enter_context(patch.object(auditor,"aggregate",side_effect=lambda rows:{"synthetic_banks":len(rows)}))
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        runner.launch()
                    except AssertionError as exc:
                        assert mode != "complete" and str(exc) == ("CPU_budget" if mode == "owned_child_budget" else "owned_audit_failure")
                    except OSError as exc:
                        assert mode in ["late_export_failure", "summary_write_failure"]
                        assert str(exc) == ("synthetic_export_failure" if mode == "late_export_failure" else "synthetic_summary_failure")
                    else:
                        assert mode == "complete"
            record = json.loads((out/"runtime.json").read_text())
            assert record["status"] == ("complete" if mode == "complete" else "incomplete")
            assert len(children) == 8 and all(c.poll() is not None for c in children)
            assert original == {str(p.relative_to(parent)):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
            if mode == "complete":
                assert record["returncodes"] == [0]*8 and len(checks)==2
                assert json.loads((public/"summary.json").read_text())["synthetic_banks"] == 256
                assert (public/"original-incomplete-runtime.json").read_bytes() == (parent/"runtime.json").read_bytes()
            else:
                assert not (out/"summary.json").exists() and not (public/"summary.json").exists()
                if mode in ["late_export_failure", "summary_write_failure"]:
                    assert len(list((out/"audits").glob("*.json"))) == 256 and public.exists()
                else:
                    assert not public.exists()
            cases.append({"mode":mode,"status":"passed","owned_children":len(children),"parent_unchanged":True})
    return dict(status="passed",cohort_reads=0,new_observations=0,new_fits=0,cases=cases,
                exact_single_string_correction=True,wall_seconds=time.monotonic()-started)


if __name__ == "__main__":
    print(json.dumps(gate(),indent=2))
