"""Two physical GPU lanes with automatic refill; no tuning after targets."""
import os
import subprocess
import sys
from common import SRC,OUT,read,save,now,frozen
PYTHON=sys.executable

def call(name,*args):
    subprocess.run([PYTHON,str(SRC/f'{name}.py'),*map(str,args)],check=True)

def pair(name,*args):
    processes=[]
    for gpu in[0,1]:
        log=OUT/(name+'-'+'-'.join(map(str,args))+f'-g{gpu}.log')
        stream=log.open('w');processes.append((subprocess.Popen([PYTHON,str(SRC/f'{name}.py'),*map(str,args),'--gpu',str(gpu)],stdout=stream,stderr=subprocess.STDOUT),stream))
    codes=[]
    for process,stream in processes:codes.append(process.wait());stream.close()
    assert codes==[0,0],(name,args,codes)

def main():
    frozen();assert not(OUT/'campaign.json').exists();save(OUT/'campaign.json',dict(status='preparing',started_utc=now(),pid=os.getpid()))
    call('source','prepare')
    for bank in range(4):call('calibration','--bank',bank,'prepare')
    pair('source','worker')
    for bank in range(4):
        if not read(OUT/f'b{bank}/model.json')['source_eligible']:continue
        save(OUT/'campaign.json',dict(status='running',utc=now(),stage='calibration',bank=bank,pid=os.getpid()));pair('calibration','--bank',bank,'worker','--phase','grid');call('calibration','--bank',bank,'select')
        if(OUT/f'b{bank}/refinement-request.json').exists():pair('calibration','--bank',bank,'worker','--phase','refinement');call('calibration','--bank',bank,'select')
    call('prepare')
    for bank in read(OUT/'dataset.json')['selected_caps']:
        save(OUT/'campaign.json',dict(status='running',utc=now(),stage='target',bank=bank,pid=os.getpid()));pair('driver','--bank',bank)
    save(OUT/'campaign.json',dict(status='complete',completed_utc=now(),pid=os.getpid()))
if __name__=='__main__':main()
