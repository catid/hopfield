"""CPU-only baseline source import, frozen configuration, and dataset checks."""
from pathlib import Path
import argparse,subprocess,sys,os,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('task');a=p.parse_args()
ROOT=Path(__file__).resolve().parent;UP=ROOT.parents[1]/'upstream';CODE=ROOT/'code'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WANDB_MODE']='disabled';os.environ['WANDB_DISABLED']='true'
name=a.method;task=a.task
source=UP/'third_party'/({'hrm':'hrm/HRM'}.get(name,name))
sys.path[:0]=[str(source),str(CODE/'trm/trm_shim')]

def recipe(sh,var):
 cmd=f'source {ROOT}/env.sh; source {CODE}/{sh}; printf "%s\\0" "${{{var}[@]}}"'
 return subprocess.check_output(['bash','-c',cmd]).decode().strip('\0').split('\0')

def inspect_data(path):
 d=Path(path);rows={}
 for split in ['train','test','dev']:
  if not (d/split/'dataset.json').exists():continue
  m=json.loads((d/split/'dataset.json').read_text());files={}
  for n in ['inputs','labels','group_indices','puzzle_indices','puzzle_identifiers']:
   f=d/split/f'all__{n}.npy';x=np.load(f,mmap_mode='r')
   files[n]=dict(shape=list(x.shape),dtype=str(x.dtype))
  assert files['inputs']['shape']==files['labels']['shape']
  rows[split]=dict(metadata=m,arrays=files)
 assert 'train' in rows
 return rows

if name in ['trm','gram','hrm']:
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf
 if name=='hrm':
  data=os.environ['TRM_DATA']+'/sudoku-extreme-1k-aug-1000' if task=='sudoku' else str(UP/'data/eqr/maze-30x30-unique-1k')
  epochs=40000 if task=='sudoku' else 20000
  ov=[f'data_path={data}',f'epochs={epochs}',f'eval_interval={epochs//10}','lr=1e-4','puzzle_emb_lr=1e-4','weight_decay=1.0','puzzle_emb_weight_decay=1.0','+seed=0']
 else:ov=recipe(f'{name}/{name}_configs.sh' if name=='trm' else 'gram/configs.sh',f'{name.upper()}_{task.upper()}')+['seed=0']
 with initialize_config_dir(config_dir=str(source/'config'),version_base=None):cfg=compose(config_name='cfg_pretrain',overrides=ov)
 config=OmegaConf.to_container(cfg,resolve=True)
 # HRM pretrain import additionally requires its native dependencies.
 imported=False;error=None
 try:import pretrain;imported=True
 except Exception as e:error=f'{type(e).__name__}: {e}'
 paths=[config['data_path']] if name=='hrm' else config['data_paths']
elif name=='ebr':
 import pretrain
 from utils.config import compose_config
 vals=recipe('ebr/configs.sh',f'EBR_{task.upper()}');cfg,config=compose_config('train/'+vals[0],vals[1:]+['seed=0','eval_interval_steps=null','wandb_mode=disabled'])
 ov=vals;paths=[cfg.dataset.data_path];imported=True;error=None
elif name=='eqr':
 sys.path.insert(0,str(CODE/'eqr'));import eqr_common as C
 data=os.environ['EQR_DATA']+('/sudoku-s0' if task=='sudoku' else '/maze-unique')
 cfg=C.load_config(task,data,0);C.install_stand_ins()
 import pretrain
 config=cfg.model_dump();ov=[task];paths=[data];imported=True;error=None
else:raise ValueError(name)
report=dict(method=name,task=task,source=str(source),commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
 pretrain_import_ok=imported,import_error=error,overrides=ov,config=config,datasets={d:inspect_data(d) for d in paths},
 cpu_only=True,readiness='source/config/data checked; GPU fit/score not yet validated')
( ROOT/'artifacts'/f'preflight-{name}-{task}.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(name,task,'import',imported,'datasets',list(report['datasets']))
if not imported:raise SystemExit(1)
