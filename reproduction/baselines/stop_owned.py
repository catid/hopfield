"""Stop only verified task baseline controllers and their captured descendants."""
from pathlib import Path
import os,json,signal,time,socket
ROOT=Path(__file__).resolve().parent
procs={}
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  args=d.joinpath('cmdline').read_bytes().split(b'\0');args=[x.decode(errors='replace') for x in args if x]
  stat=d.joinpath('stat').read_text().split(') ',1)[1].split()
  procs[int(d.name)]=dict(pid=int(d.name),ppid=int(stat[1]),start=stat[19],argv=args)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
roots=[]
for pid,r in procs.items():
 a=r['argv'];base=Path(a[0]).name if a else ''
 if base.startswith('python') and any(Path(x).name in ['lattice_queue.py','gram_queue.py','install_native_adam.py'] for x in a[1:3]):roots.append(pid)
 if base=='uv' and a[1:3]==['pip','install'] and any('flash-attn==2.8.3'==x or 'adam-atan2' in x for x in a[3:]) and any('reproduction/baselines' in x for x in a):roots.append(pid)
owned=set(roots)
while True:
 more={pid for pid,r in procs.items() if r['ppid'] in owned}-owned
 if not more:break
 owned.update(more)
# Any surviving direct fit/check is included only with the task-owned code path.
for pid,r in procs.items():
 a=r['argv']
 if a and Path(a[0]).name.startswith('python') and any('/hopfield/reproduction/baselines/code/' in x or '/hopfield/reproduction/baselines/check_lattice_gpu.py'==x for x in a[1:3]):owned.add(pid)
def same(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]==procs[pid]['start']
 except FileNotFoundError:return False
stopped=[]
for pid in roots+sorted(owned-set(roots)):
 if same(pid):
  try:os.kill(pid,signal.SIGTERM);stopped.append(pid)
  except ProcessLookupError:pass
time.sleep(3)
forced=[]
for pid in owned:
 if same(pid):
  try:os.kill(pid,signal.SIGKILL);forced.append(pid)
  except ProcessLookupError:pass
record=dict(host=socket.gethostname(),reason='User narrowed scope to CRAM main results controls and ablations; baseline reproductions canceled',
 controllers=roots,terminated=stopped,killed_after_grace=forced,
 processes={str(pid):dict(ppid=procs[pid]['ppid'],argv=procs[pid]['argv'][:3]) for pid in sorted(owned)},
 timestamp=time.time())
(ROOT/'artifacts'/f'cancellation-{socket.gethostname()}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
