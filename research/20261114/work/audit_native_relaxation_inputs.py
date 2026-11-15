"""Validate frozen inputs on CPUs while the paired GPU jobs are running."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
import time
import audit as A
import audit_native_relaxation as V
import study as S


def main():
    dest=S.OUT/"native_relaxation";pp=S.HERE/"native_relaxation-protocol.json";manifest=json.loads((dest/"datasets.json").read_text());assert manifest["protocol_sha256"]==S.sha(pp)
    for ds in manifest["datasets"]:assert S.sha(S.ROOT/ds["path"])==ds["sha256"]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:distances=list(pool.map(V.check_dataset,[S.ROOT/ds["path"] for ds in manifest["datasets"]]))
    result={"status":"passed","utc":S.utc(),"protocol_sha256":S.sha(pp),"datasets":manifest["datasets"],"distances":distances,"inputs_checked":sum(map(len,distances)),"processes":8,"oracle_seconds":time.perf_counter()-started,"script_sha256":S.sha(__file__),"graph_checker_sha256":S.sha(A.__file__),"dataset_checker_source_sha256":hashlib.sha256(inspect.getsource(V.check_dataset).encode()).hexdigest(),"outcome_data_read":False}
    S.save(dest/"independent-input-audit.json",result);S.emit("native_relaxation_inputs_audited",inputs=result["inputs_checked"],seconds=result["oracle_seconds"])


if __name__=="__main__":main()
