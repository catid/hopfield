"""Run frozen hand-set Sudoku controls, development gate then full test once."""
from pathlib import Path
import argparse,subprocess,os,json,hashlib,datetime,socket
p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);a=p.parse_args()
root=Path('/home/catid/hopfield');src=root/'upstream';out=root/'artifacts/hand_sudoku';out.mkdir(exist_ok=True)
for seed in a.seeds:
 record=out/f'seed{seed}.json'
 meta={'seed':seed,'host':socket.gethostname(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_commit':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),'script_sha256':hashlib.sha256((src/'experiments/hand_set.py').read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':[]}
 for phase in ['dev','test']:
  argv=[str(src/'.venv/bin/python'),'-u','hand_set.py','sudoku','--data',phase,'--seed',str(seed)]
  print(json.dumps({'seed':seed,'phase':phase,'argv':argv}),flush=True)
  with (out/f'seed{seed}-{phase}.log').open('a') as f: res=subprocess.run(argv,cwd=src/'experiments',stdout=f,stderr=subprocess.STDOUT)
  meta['phases'].append({'phase':phase,'argv':argv,'exit_code':res.returncode})
  record.write_text(json.dumps(meta,indent=2)+'\n')
  res.check_returncode()
 meta['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();record.write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({'seed':seed,'status':'complete'}),flush=True)
