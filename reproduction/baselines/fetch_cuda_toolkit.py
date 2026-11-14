from pathlib import Path
import json,urllib.request,hashlib,tarfile,shutil
ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'artifacts/cuda-12.8.1-redistrib.json').read_text())
dest=ROOT/'tools/cuda-12.8.1';cache=ROOT/'tools/downloads';cache.mkdir(parents=True,exist_ok=True)
for name in ['cuda_nvcc','cuda_cudart','cuda_cccl']:
 r=j[name]['linux-x86_64'];p=cache/Path(r['relative_path']).name
 if not p.exists():urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/'+r['relative_path'],p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 staging=cache/name;staging.mkdir(exist_ok=True)
 with tarfile.open(p) as t:t.extractall(staging,filter='data')
 src=next(staging.iterdir());shutil.copytree(src,dest,dirs_exist_ok=True,symlinks=True)
 print(name,'verified and staged',flush=True)
