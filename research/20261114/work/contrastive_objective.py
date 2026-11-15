"""Fixed-readout answer-only PL and endpoint-negative contrastive objectives."""
import study as S
from functools import lru_cache

WEIGHT=.1
UPDATES=400
LR=.01
BATCH=64
PARAMETERS=('T','b','kappa')

@lru_cache(maxsize=1)
def torch_module():
    return S.imports()[0]


def make_trainable(model):
    for name,p in model.named_parameters():p.requires_grad_(name in PARAMETERS)
    return [getattr(model,name) for name in PARAMETERS]


def losses(model,t,y,a,weight):
    torch=torch_module()
    pl=model.pl_loss(t,y,a)
    label=model.energy(t,y,a);endpoint=model.energy(t,torch.zeros_like(y),a)
    selected_free=(y*model.free(t)).flatten(1).sum(1).clamp_min(1)
    normalized=(label-endpoint)/selected_free
    contrast=torch.nn.functional.softplus(1+normalized).mean()
    # Both arms compute the same diagnostic/loss graph. Weight0 is exactlyPL.
    return pl+weight*contrast,pl,contrast,normalized


def optimizer(torch,model):
    return torch.optim.AdamW(make_trainable(model),lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
