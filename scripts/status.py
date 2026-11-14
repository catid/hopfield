"""Read-only inventory and progress snapshots for the authorized reproduction hosts."""
import argparse,json,subprocess,datetime
from pathlib import Path
HOSTS=('foureyes','ripper','work','loki')
CMD="nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv,noheader; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader"
def query(host):
 args=['bash','-c',CMD] if host=='foureyes' else ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',host,CMD]
 p=subprocess.run(args,capture_output=True,text=True,timeout=20)
 return {'host':host,'exit_code':p.returncode,'gpu_and_processes':p.stdout.strip(),'error':p.stderr.strip()}
if __name__=='__main__':
 from concurrent.futures import ThreadPoolExecutor
 a=argparse.ArgumentParser();a.add_argument('--save',action='store_true');ns=a.parse_args()
 with ThreadPoolExecutor(max_workers=4) as ex: hosts=list(ex.map(query,HOSTS))
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hosts':hosts}
 print(json.dumps(out,indent=2))
 if ns.save:
  path=Path(__file__).resolve().parents[1]/'artifacts/utilization.jsonl'
  with path.open('a') as f:f.write(json.dumps(out)+'\n')
