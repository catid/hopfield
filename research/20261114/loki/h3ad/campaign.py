"""Refill Loki only after the preceding H3ac study releases both GPUs."""
import argparse
import os
import resource
import subprocess
import sys
import time
from common import SRC,OUT,OLD,read,save,now,frozen
PYTHON=sys.executable

def call(name,*args):subprocess.run([PYTHON,str(SRC/f'{name}.py'),*map(str,args)],check=True)
def pair(name,*args):
    tasks=[]
    for gpu in[0,1]:
        stream=(OUT/(name+'-'+'-'.join(map(str,args))+f'-g{gpu}.log')).open('w');tasks.append((subprocess.Popen([PYTHON,str(SRC/f'{name}.py'),*map(str,args),'--gpu',str(gpu)],stdout=stream,stderr=subprocess.STDOUT),stream))
    codes=[]
    for process,stream in tasks:codes.append(process.wait());stream.close()
    assert codes==[0,0],(name,args,codes)

def main(mode):
    frozen();OUT.mkdir(parents=True,exist_ok=True)
    if mode=='prepare':
        assert not(OUT/'cpu-preparation.json').exists();start=time.process_time();wall=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_CHILDREN);before=usage.ru_utime+usage.ru_stime;call('generate');call('source','prepare')
        for bank in range(2):call('calibration','--bank',bank,'prepare')
        usage=resource.getrusage(resource.RUSAGE_CHILDREN);save(OUT/'cpu-preparation.json',dict(status='complete',utc=now(),controller_cpu_seconds=time.process_time()-start,aggregate_child_cpu_seconds=usage.ru_utime+usage.ru_stime-before,wall_seconds=time.perf_counter()-wall,no_gpu_workers_started=True));return
    assert read(OUT/'cpu-preparation.json')['status']=='complete';assert read(OUT/'enumeration-gate.json')['status']=='complete_verified';assert not(OUT/'campaign.json').exists();assert read(OLD/'campaign.json')['status']=='complete'
    save(OUT/'campaign.json',dict(status='running',started_utc=now(),stage='fit',pid=os.getpid()));pair('source','worker')
    for bank in range(2):
        if not read(OUT/f'b{bank}/model.json')['source_eligible']:continue
        save(OUT/'campaign.json',dict(status='running',utc=now(),stage='calibration',bank=bank,pid=os.getpid()));pair('calibration','--bank',bank,'worker','--phase','grid');call('calibration','--bank',bank,'select')
        if(OUT/f'b{bank}/refinement-request.json').exists():pair('calibration','--bank',bank,'worker','--phase','refinement');call('calibration','--bank',bank,'select')
    call('prepare')
    for dataset in read(OUT/'datasets.json')['datasets']:
        bank=dataset['bank'];save(OUT/'campaign.json',dict(status='running',utc=now(),stage='target',bank=bank,pid=os.getpid()));pair('driver','--bank',bank)
    save(OUT/'campaign.json',dict(status='complete',completed_utc=now(),pid=os.getpid()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','gpu']);args=a.parse_args();main(args.mode)
