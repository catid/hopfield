"""Linear binary-energy/flip features for exactly23answer parameters."""
from fractions import Fraction as Q
import numpy as np


def slices(shape,dr,dc):
    h,w=shape
    return (slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc))), (slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))


def energy_features(tok,board,attention,muoffs,aoffs):
    p=np.where(tok==1,board,tok>=2).astype(np.int8);x=np.zeros(23,np.float64)
    for k,(dr,dc) in enumerate(muoffs):
        l,r=slices(tok.shape,int(dr),int(dc));left=p[l];right=p[r]
        for u in range(2):
            for v in range(2):x[4*k+2*u+v]=np.count_nonzero((left==u)&(right==v))
    x[16]=-np.count_nonzero((tok==1)&(p==0));x[17]=-np.count_nonzero((tok==1)&(p==1))
    for k,(dr,dc) in enumerate(aoffs):
        l,r=slices(tok.shape,int(dr),int(dc));x[18+k]=-np.sum(attention[k][l].astype(np.float64)*p[l]*p[r],dtype=np.float64)
    return x


def flip_features(tok,board,attention,muoffs,aoffs):
    p=np.where(tok==1,board,tok>=2).astype(np.int8);rr,cc=np.where(tok==1);n=len(rr);x=np.zeros((n,23),np.float64);yi=p[rr,cc];new=1-yi;dl=1-2*yi;h,w=tok.shape;ids=np.arange(n)
    for k,(dr,dc) in enumerate(muoffs):
        dr,dc=int(dr),int(dc)
        r,c=rr+dr,cc+dc;ok=(r>=0)&(r<h)&(c>=0)&(c<w);ii=ids[ok];nb=p[r[ok],c[ok]];x[ii,4*k+2*new[ok]+nb]+=1;x[ii,4*k+2*yi[ok]+nb]-=1
        r,c=rr-dr,cc-dc;ok=(r>=0)&(r<h)&(c>=0)&(c<w);ii=ids[ok];nb=p[r[ok],c[ok]];x[ii,4*k+2*nb+new[ok]]+=1;x[ii,4*k+2*nb+yi[ok]]-=1
    x[:,16]=dl;x[:,17]=-dl
    for k,(dr,dc) in enumerate(aoffs):
        dr,dc=int(dr),int(dc)
        if dr==dc==0:x[:,18+k]=-dl*attention[k,rr,cc].astype(np.float64);continue
        r,c=rr+dr,cc+dc;ok=(r>=0)&(r<h)&(c>=0)&(c<w);x[ids[ok],18+k]-=dl[ok]*attention[k,rr[ok],cc[ok]].astype(np.float64)*p[r[ok],c[ok]]
        r,c=rr-dr,cc-dc;ok=(r>=0)&(r<h)&(c>=0)&(c<w);x[ids[ok],18+k]-=dl[ok]*attention[k,r[ok],c[ok]].astype(np.float64)*p[r[ok],c[ok]]
    return x,np.ravel_multi_index((rr,cc),tok.shape)


def constraints(tok,label,attention,muoffs,aoffs):
    flip,flat=flip_features(tok,label,attention,muoffs,aoffs);e=energy_features(tok,label,attention,muoffs,aoffs);on=np.count_nonzero((tok==1)&label);off=np.count_nonzero((tok==1)&~label);assert on>0 and off>0
    endpoint=(energy_features(tok,np.zeros_like(label),attention,muoffs,aoffs)-e)/on;dense=(energy_features(tok,np.ones_like(label),attention,muoffs,aoffs)-e)/off
    return np.concatenate([flip,endpoint[None],dense[None]]),np.concatenate([np.zeros(len(flip),np.int8),[1,2]]),np.concatenate([flat,[-1,-1]])


def exact_row(tok,label,attention,muoffs,aoffs,kind,flat):
    """Independent rational construction for a candidate dual's support rows."""
    p=np.where(tok==1,label,tok>=2).astype(np.int8);h,w=tok.shape;zero=Q(0)
    if kind==0:
        r,c=divmod(int(flat),w);old=int(p[r,c]);new=1-old;dl=new-old;x=[zero]*23
        for k,(dr,dc) in enumerate(muoffs):
            dr,dc=int(dr),int(dc)
            if 0<=r+dr<h and 0<=c+dc<w:
                v=int(p[r+dr,c+dc]);x[4*k+2*new+v]+=1;x[4*k+2*old+v]-=1
            if 0<=r-dr<h and 0<=c-dc<w:
                u=int(p[r-dr,c-dc]);x[4*k+2*u+new]+=1;x[4*k+2*u+old]-=1
        x[16]=Q(dl);x[17]=Q(-dl)
        for k,(dr,dc) in enumerate(aoffs):
            dr,dc=int(dr),int(dc)
            if dr==dc==0:x[18+k]=-dl*Q.from_float(float(attention[k,r,c]));continue
            if 0<=r+dr<h and 0<=c+dc<w:x[18+k]-=dl*int(p[r+dr,c+dc])*Q.from_float(float(attention[k,r,c]))
            if 0<=r-dr<h and 0<=c-dc<w:x[18+k]-=dl*int(p[r-dr,c-dc])*Q.from_float(float(attention[k,r-dr,c-dc]))
        return list(map(Q,x))
    def feature(board):
        x=[zero]*23;z=np.where(tok==1,board,tok>=2).astype(np.int8)
        for k,(dr,dc) in enumerate(muoffs):
            l,r=slices(tok.shape,int(dr),int(dc));left=z[l];right=z[r]
            for u in range(2):
                for v in range(2):x[4*k+2*u+v]=Q(int(np.count_nonzero((left==u)&(right==v))))
        x[16]=Q(-int(np.count_nonzero((tok==1)&(z==0))));x[17]=Q(-int(np.count_nonzero((tok==1)&(z==1))))
        for k,(dr,dc) in enumerate(aoffs):
            l,r=slices(tok.shape,int(dr),int(dc));mask=(z[l]*z[r]).astype(bool);x[18+k]=-sum((Q.from_float(float(v)) for v in attention[k][l][mask]),zero)
        return x
    canonical=feature(p);negative=feature(np.zeros_like(label) if kind==1 else np.ones_like(label));n=int(np.count_nonzero((tok==1)&(label if kind==1 else ~label)));return [(a-b)/n for a,b in zip(negative,canonical)]
