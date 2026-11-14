from pathlib import Path
import json,urllib.request,hashlib,tarfile,subprocess,os
ROOT=Path(__file__).resolve().parent
meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/adam-atan2/0.0.3/json'))
r=next(v for v in meta['urls'] if v['packagetype']=='sdist')
cache=ROOT/'tools/downloads';cache.mkdir(parents=True,exist_ok=True);p=cache/r['filename']
if not p.exists():urllib.request.urlretrieve(r['url'],p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['digests']['sha256']
area=ROOT/'tools/adam-atan2-source';area.mkdir(exist_ok=True)
with tarfile.open(p) as t:t.extractall(area,filter='data')
src=next(area.iterdir())
base=ROOT.parents[1]/'upstream/.venv/lib/python3.13/site-packages/nvidia'
env=dict(os.environ,CUDA_HOME=str(ROOT/'tools/cuda-12.8.1'),CUDA_VISIBLE_DEVICES='',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='12.0',CPATH=':'.join(str(p) for p in base.glob('*/include')))
record=dict(pypi_version='0.0.3',source_url=r['url'],source_sha256=r['digests']['sha256'],metadata_note='Released sdist produces package version0.0.0; installing verified local sdist avoids uv index-version mismatch; numerical source unchanged',cuda_arch_list='12.0')
(ROOT/'artifacts/adam-atan2-source.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(subprocess.run(['uv','pip','install','--python',str(ROOT/'.venv/bin/python'),'--no-deps','--no-build-isolation',str(src)],env=env).returncode)
