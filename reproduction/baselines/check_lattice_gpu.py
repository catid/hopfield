"""Frozen TRMP lattice loss/gradient parity across CPU, CUDA and compiled bf16.

Uses only eight training puzzles and a separate cache. No test inputs are opened.
"""
from pathlib import Path
import copy,json,os,sys,time,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/trmp'))
import lattice as L
import lattice_fit as F


def run(net,x,y,amp=False,compile=False):
    def forward(x,y):
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=amp):h=net(x)
        return L.lattice_loss(h,x,y,x.sum(-1)==1)[0]
    fn=torch.compile(forward) if compile else forward
    if x.is_cuda:torch.cuda.synchronize()
    begin=time.monotonic();loss=fn(x,y);loss.backward()
    if x.is_cuda:torch.cuda.synchronize()
    unused=[n for n,p in net.named_parameters() if p.grad is None]
    grad=torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1).float().cpu() for p in net.parameters()])
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    return dict(loss=float(loss.detach()),seconds=time.monotonic()-begin,unused_parameters=unused),grad


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-20))


if __name__=='__main__':
    out=ROOT/'smoke/lattice-parity';out.mkdir(parents=True,exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'.inductor')
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw=F.read_raw(F.Config(seed=0));x,digits=F.lattice_of(raw['solutions'][:8],raw['clue_mask'][:8])
    y=torch.nn.functional.one_hot(digits,9).float()
    model=L.LatticeNet()
    cpu,cg=run(copy.deepcopy(model),x,y)
    gpu,gg=run(copy.deepcopy(model).cuda(),x.cuda(),y.cuda())
    bf,bg=run(copy.deepcopy(model).cuda(),x.cuda(),y.cuda(),amp=True)
    compiled,fg=run(copy.deepcopy(model).cuda(),x.cuda(),y.cuda(),amp=True,compile=True)
    report=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in model.parameters()),
      cpu=cpu,cuda_fp32=gpu,cuda_bf16=bf,cuda_compiled_bf16=compiled,
      fp32_loss_relative=abs(gpu['loss']/cpu['loss']-1),fp32_gradient_relative=relative(gg,cg),
      bf16_loss_relative=abs(bf['loss']/gpu['loss']-1),bf16_gradient_relative=relative(bg,gg),
      compiled_loss_relative_vs_bf16=abs(compiled['loss']/bf['loss']-1),compiled_gradient_relative_vs_bf16=relative(fg,bg),
      peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/trmp/lattice.py',ROOT/'code/trmp/lattice_fit.py']})
    thresholds=dict(fp32_loss_relative=5e-4,fp32_gradient_relative=5e-4,bf16_loss_relative=.02,bf16_gradient_relative=.08,compiled_loss_relative_vs_bf16=.002,compiled_gradient_relative_vs_bf16=.02)
    report['declared_tolerances']=thresholds
    report['passed']=all(report[k]<v for k,v in thresholds.items()) and cpu['unused_parameters']==gpu['unused_parameters']==bf['unused_parameters']==compiled['unused_parameters']
    (out/'check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit('Lattice numeric parity exceeded declared tolerance')
