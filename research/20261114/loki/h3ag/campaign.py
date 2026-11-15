"""Single bounded GPU comparison after model-only CPU compilation."""
import os
import subprocess
import sys
from common import SRC,OUT,OLD,read,save,now,frozen
PYTHON=sys.executable

def call(name,*args):subprocess.run([PYTHON,str(SRC/f'{name}.py'),*map(str,args)],check=True)
def pair(bank):
    tasks=[]
    for gpu in[0,1]:
        stream=(OUT/f'target-b{bank}-g{gpu}.log').open('w');tasks.append((subprocess.Popen([PYTHON,str(SRC/'driver.py'),'--bank',str(bank),'--gpu',str(gpu)],stdout=stream,stderr=subprocess.STDOUT),stream))
    for process,stream in tasks:assert process.wait()==0;stream.close()

def main():
    frozen();assert read(OLD/'campaign.json')['status']=='complete';assert not(OUT/'campaign.json').exists();save(OUT/'campaign.json',dict(status='running',stage='compilation',utc=now(),pid=os.getpid()));call('compile_gate');call('graph_gate');call('prepare')
    # Label-free CPU querying overlaps the GPU comparisons; no GPU used by query.py.
    stream=(OUT/'query.log').open('w');query=subprocess.Popen([PYTHON,str(SRC/'query.py')],stdout=stream,stderr=subprocess.STDOUT)
    for bank in range(2):save(OUT/'campaign.json',dict(status='running',stage='GPU_controls',bank=bank,utc=now(),pid=os.getpid()));pair(bank)
    assert query.wait()==0;stream.close();save(OUT/'campaign.json',dict(status='complete',completed_utc=now(),pid=os.getpid()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'campaign.json',dict(status='stopped_with_failure',completed_utc=now(),error=repr(e),pid=os.getpid()));raise
