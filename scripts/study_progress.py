"""Read-only live GPU/process/counter snapshots across the authorized hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("foureyes", "ripper", "work", "loki")
PROBE = r'''
import datetime as dt,json,os,socket,subprocess,time
from pathlib import Path
root=Path('/home/catid/hopfield'); now=time.time()
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
def tail(path):
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-16384));lines=f.read().decode(errors='replace').splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except ValueError:pass
 return None
gpus=command(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'])
apps=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
processes=[]
for line in apps['stdout'].splitlines():
 try:
  pid=int(line.split(',')[0]);p=Path('/proc')/str(pid)
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  processes.append({'pid':pid,'gpu_uuid':line.split(',')[1].strip(),
   'argv':(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))[:-1],
   'cpu_ticks':int(stat[11])+int(stat[12]),'start_ticks':int(stat[19]),
   'ticks_per_second':os.sysconf('SC_CLK_TCK')})
 except (ValueError,OSError,IndexError):pass
patterns=['artifacts/objectives/*-state.json','artifacts/figure4/*-state.json',
 'artifacts/d2/*-state.json','artifacts/main_followups/seed*.json',
 'artifacts/data_efficiency/gpu*-worker.json','artifacts/long_bptt/seed*.json',
 'artifacts/maze_hard_ablations/queue-*.json']
states=[]
for pattern in patterns:
 for p in sorted(root.glob(pattern)):
  try:
   d=json.loads(p.read_text())
   owner=d.get('hostname',d.get('host',d.get('lane','').split(':')[0]))
   if owner and owner.split('.')[0]!=socket.gethostname().split('.')[0]:continue
   v={k:d[k] for k in ['status','job','current_job','stage','worker_pid','pid','child_pid','progress','latest_training_counter','elapsed_s','utc','updated_utc'] if k in d}
   v.update(path=str(p.relative_to(root)),age_seconds=max(0,now-p.stat().st_mtime));states.append(v)
  except (OSError,ValueError) as e:states.append({'path':str(p.relative_to(root)),'read_error':str(e)})
long_fits=[]
for p in sorted(root.glob('upstream/experiments/runs/ALT/longD3/*/log.jsonl')):
 try:long_fits.append({'path':str(p.relative_to(root)),'age_seconds':max(0,now-p.stat().st_mtime),'last_counter':tail(p)})
 except OSError:pass
search_checkpoints=[]
for p in sorted(root.glob('artifacts/difficulty/full/test/*-state/ckpt.pt')):
 st=p.stat();search_checkpoints.append({'path':str(p.relative_to(root)),'bytes':st.st_size,'age_seconds':now-st.st_mtime,'mtime_ns':st.st_mtime_ns})
print(json.dumps({'host':socket.gethostname(),'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
 'gpus':gpus,'gpu_apps':apps,'processes':processes,'states':states,
 'long_fit_counters':long_fits,'difficulty_checkpoints':search_checkpoints}))
'''


def query(host):
    local = socket.gethostname().split(".")[0] == host
    argv = ["python3", "-c", PROBE] if local else [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10", host, "python3 -c " + shlex.quote(PROBE)]
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=45)
        if p.returncode:
            return {"host": host, "error": p.stderr.strip(), "exit_code": p.returncode}
        return json.loads(p.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        return {"host": host, "error": repr(error)}


def snapshot(save=False):
    out = ROOT / "artifacts/study_progress"
    previous = {}
    if (out / "latest.json").exists():
        previous = json.loads((out / "latest.json").read_text())
    previous_hosts = {h["host"]: h for h in previous.get("hosts", [])}
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        hosts = list(pool.map(query, HOSTS))
    for host in hosts:
        old = previous_hosts.get(host["host"], {})
        processes = {(p["pid"], p["start_ticks"]): p for p in old.get("processes", [])}
        for p in host.get("processes", []):
            before = processes.get((p["pid"], p["start_ticks"]))
            if before:
                p["cpu_seconds_since_previous_snapshot"] = (p["cpu_ticks"] - before["cpu_ticks"]) / p["ticks_per_second"]
        counters = {p["path"]: p for p in old.get("long_fit_counters", [])}
        for p in host.get("long_fit_counters", []):
            before = counters.get(p["path"])
            if before:
                p["counter_changed_since_previous_snapshot"] = p["last_counter"] != before["last_counter"]
    result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "hosts": hosts,
              "interpretation": "GPU utilization, live process CPU counters, explicit training updates and search checkpoint modification times are retained separately. An unchanged sparse counter alone does not establish a stall."}
    if save:
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "latest.tmp.json"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(out / "latest.json")
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    return result


def brief(result):
    values = []
    for h in result["hosts"]:
        if "error" in h:
            values.append({"host": h["host"], "error": h["error"]})
            continue
        values.append({"host": h["host"], "gpus": h["gpus"]["stdout"],
                       "live_gpu_pids": [p["pid"] for p in h["processes"]],
                       "last_long_fit_updates": [p["last_counter"].get("update") for p in h["long_fit_counters"] if isinstance(p["last_counter"], dict)],
                       "queue_statuses": [{k: s[k] for k in ("path", "status", "job", "stage") if k in s} for s in h["states"]]})
    return {"utc": result["utc"], "hosts": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--watch", action="store_true", help="Save a snapshot every minute until interrupted")
    p.add_argument("--full", action="store_true")
    a = p.parse_args()
    if a.watch:
        (ROOT / "artifacts/study_progress").mkdir(parents=True, exist_ok=True)
        lock = (ROOT / "artifacts/study_progress/watch.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        result = snapshot(save=a.save or a.watch)
        print(json.dumps(result if a.full else brief(result), indent=None if a.watch else 2), flush=True)
        if not a.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
