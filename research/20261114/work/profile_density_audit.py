"""Bounded source profiling only; never reports unfinished science as verified."""
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import signal
import time
from audit_density_transfer import job_check
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'artifacts/research/20261114/work/density_transfer'


def main():
    def alarm(signum,frame):raise TimeoutError('Bounded60s single-job profiling budget')
    signal.signal(signal.SIGALRM,alarm);p=json.loads((D/'protocol.json').read_text());profile=cProfile.Profile();started=time.perf_counter();status='completed';signal.alarm(60)
    try:profile.enable();job_check((ROOT,D,p['jobs'][0],p))
    except TimeoutError:status='profile_budget_exhausted'
    finally:
        profile.disable();signal.alarm(0);elapsed=time.perf_counter()-started
    path=D/'audit-profile.pstats';profile.dump_stats(path);buf=io.StringIO();pstats.Stats(profile,stream=buf).sort_stats('cumulative').print_stats(35);(D/'audit-profile.txt').write_text(buf.getvalue());r={'status':status,'kind':'bounded_source_runtime_diagnostic_not_scientific_audit','job':p['jobs'][0],'budget_seconds':60,'seconds':elapsed,'profile_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'full_audit_completed':False};(D/'audit-profile.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));print(buf.getvalue())


if __name__=='__main__':main()
