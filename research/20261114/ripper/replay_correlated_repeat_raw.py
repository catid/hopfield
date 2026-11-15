"""Read-only postrun H1ch raw CSV/noise/numerical/proof replay; never fits."""
from concurrent.futures import ProcessPoolExecutor
import json,time
import correlated_repeat_data as D
from correlated_repeat_study import read,OUT,PROTOCOL,sha
from audit_correlated_repeat import audit_bank


def bank(seed):
    assert D.audit_queries(seed,read(PROTOCOL))==read(OUT/f's{seed}/query-record-audit.json')
    assert audit_bank(seed,False,read(PROTOCOL))==read(OUT/f's{seed}/audit.json')
    return seed


def main():
    start=time.perf_counter();original=D.save
    def compare(path,result):assert read(path)==result
    D.save=compare
    try:D.sources()
    finally:D.save=original
    with ProcessPoolExecutor(max_workers=8) as pool:assert list(pool.map(bank,range(72)))==list(range(72))
    print(json.dumps(dict(status='passed',read_only=True,protocol_sha256=sha(PROTOCOL),raw_csv_and_all72_noise_archives_replayed=True,all144_costs_grids_and_proofs_replayed=True,new_fits=0,wall_seconds=time.perf_counter()-start)))


if __name__=='__main__':main()
