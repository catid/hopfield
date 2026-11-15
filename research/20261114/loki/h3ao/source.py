"""Nested source-only count controls and eight primary N32 BP fits."""
import argparse
import math
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load

def count_model(d):
    keep=(d['counts']==0)&(d['exposure']>=math.log(264/.01));keep|=(d['offsets']@d['lines'].T==0).any(1);lookup={tuple(o):bool(value)for o,value in zip(d['offsets'],keep)};F=np.zeros((169,169),np.int32)
    for a in range(169):
        for b in range(a+1,169):
            dr,dc=b//13-a//13,b%13-a%13;F[a,b]=F[b,a]=any(r*dr+c*dc==0 for r,c in d['lines'])or lookup.get((dr,dc),False)
    return dict(F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),unit=np.array(1.),q=np.array(13))
def prepare():
    p=frozen();helper=load(SRC.parent/'h3ad/source.py','h3ao_matched_design');rows=[]
    for family,cfg in enumerate(p['families']):
        columns=np.load(ROOT/cfg['source_path']);population=np.zeros((len(columns),144),bool);population[np.arange(len(columns))[:,None],12*np.arange(12)[None,:]+columns]=True
        for bank in range(4):
            index=4*family+bank;seed=p['sampling']['stream_seeds'][index];indices=np.random.default_rng(seed).choice(len(population),64,replace=False);folder=OUT/f'b{index}';folder.mkdir(parents=True,exist_ok=True);np.save(folder/'ordered64-indices.npy',indices);doses=[]
            for dose in(16,32,64):
                dest=folder/f'd{dose}';dest.mkdir(exist_ok=True);d=helper.design(population[indices[:dose]]);d['population_indices']=indices[:dose];verification=helper.independent_design(d);np.savez_compressed(dest/'source-design.npz',**d);control=count_model(d);np.savez_compressed(dest/'count-model.npz',**control);doses.append(dict(dose=dose,design_sha256=sha(dest/'source-design.npz'),count_model_sha256=sha(dest/'count-model.npz'),**verification))
                if dose==32:(folder/'source-design.npz').write_bytes((dest/'source-design.npz').read_bytes())
            rows.append(dict(index=index,family=family,bank=bank,seed=seed,ordered64_indices=indices.tolist(),doses=doses,source_design_sha256=sha(folder/'source-design.npz')))
    save(OUT/'source-banks.json',dict(status='all_nested_source_banks_complete',utc=now(),banks=rows,within_family_N64_overlaps={str(f):[[len(set(rows[4*f+a]['ordered64_indices'])&set(rows[4*f+b]['ordered64_indices']))for b in range(4)]for a in range(4)]for f in range(2)},protocol_sha256=sha(SRC/'protocol.json')))
def worker(gpu):
    # This unchanged frozen worker reads the current common.SRC/OUT interface.
    load(SRC.parent/'h3an/source.py','h3ao_eight_primary_fit_worker').worker(gpu)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args();prepare()if args.mode=='prepare'else worker(args.gpu)
