"""A compensated diagonal term changes only the continuous extension."""


def regularizer(model,tok,p,strength):
    return strength*(model.free(tok).to(p.dtype)*p*(1-p)).flatten(1).sum(1)


def energy(model,tok,p,a,strength):
    return model.energy(tok,p,a)+regularizer(model,tok,p,strength)


def field(model,tok,p,a,strength):
    base=model.field(tok,p,a)
    if strength==0:return base
    return base+strength*model.free(tok).to(p.dtype)*(2*p-1)


def flip_delta(model,tok,binary,a):
    # R is exactly zero at both ends of every allowed binary flip.
    return model.flip_delta(tok,binary,a)
