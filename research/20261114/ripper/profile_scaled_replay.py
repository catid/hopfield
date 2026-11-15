"""Bounded200-row CPU profile; deliberately no full-audit claim or writes."""
import builtins
import cProfile
import json
import pstats
import sys
import time
import audit_scaled_noise_acquisition as audit
from run import ROOT,HERE,sha,save,now

class ProfileComplete(Exception):pass


def bounded_range(*args):
    if args!=(4440,):return builtins.range(*args)
    def rows():
        yield from builtins.range(200)
        raise ProfileComplete()
    return rows()


def main():
    sys.argv=[__file__,'--pilot','--seed','0'];audit.range=bounded_range
    audit.save=lambda *args,**kwargs:None
    profiler=cProfile.Profile();start=time.monotonic();profiler.enable()
    try:audit.main()
    except ProfileComplete:pass
    else:raise AssertionError('Expected bounded profile stop')
    profiler.disable();stats=pstats.Stats(profiler);rows=[]
    for (file,line,function),(primitive,total,self_s,cumulative_s,callers) in stats.stats.items():
        rows.append({'file':file,'line':line,'function':function,'calls':total,'self_s':self_s,'cumulative_s':cumulative_s})
    rows.sort(key=lambda r:r['self_s'],reverse=True)
    r={'status':'bounded_profile_only','utc':now(),'wrapper_sha256':sha(__file__),'reference_sha256':sha(HERE/'audit_scaled_noise_acquisition.py'),
       'replayed_rows':200,'full_audit_claim':False,'existing_audits_modified':False,'elapsed_s':time.monotonic()-start,'top_self_time':rows[:30]}
    save(ROOT/'artifacts/research/20261114/ripper/scaled_replay_profile.json',r);print(json.dumps(r))


if __name__=='__main__':main()
