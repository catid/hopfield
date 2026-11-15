"""Complete ordinary-queen universe filtering and independent literal-F parity."""
import itertools
import subprocess
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
def graph(n,leaper=None):
    F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr,dc=abs(a//n-b//n),abs(a%n-b%n);F[a,b]=F[b,a]=not dr or not dc or dr==dc or(leaper is not None and(dr,dc)in(leaper,leaper[::-1]))
    return F
def hist(columns):
    h=np.zeros(170**3,np.int64);cells=13*np.arange(13)[None,:]+columns.astype(np.int64)
    for k in(1,2,3):
        ix=np.array(list(itertools.combinations(range(13),k)));z=((cells[:,ix]+1)*np.array([170**(2-j)for j in range(k)])).sum(-1);h+=np.bincount(z.ravel(),minlength=170**3)
    assert h.sum()==len(columns)*377;return h
def main():
    p=frozen(True);source=read(OUT/'source-outcomes.json');helper=load(SRC.parent/'h3ad/generate.py','h3ao_ordinary_queen_reference');ordinary=helper.queens(13);assert len(ordinary)==73712;queenF=graph(13);binary=OUT/'reverse_literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True)
    def compiled(F,name):
        inp=OUT/f'{name}-F.bin';out=OUT/f'{name}-columns.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert int(result.stdout)==len(columns);return columns
    assert np.array_equal(ordinary,compiled(queenF,'ordinary'))
    references={}
    for family,cfg in enumerate(p['families']):
        truth=np.load(ROOT/cfg['target_path']);assert np.array_equal(truth,helper.filter_leaper(ordinary,tuple(cfg['leaper'])));Ftrue=graph(13,tuple(cfg['leaper']));assert np.array_equal(truth,compiled(Ftrue,f'family{family}'));h=hist(truth);keys=np.flatnonzero(h);folder=OUT/f'family{family}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'input-universe.npz',keys=keys,counts=h[keys]);np.save(folder/'true-columns.npy',truth);np.save(folder/'true-F.npy',Ftrue);references[family]=(truth,Ftrue,keys,{row.tobytes()for row in truth})
    full={r['group']:np.load(OUT/f"support{r['group']}/F.npz")['F']for r in source['supports']};assert all((F>=queenF).all()for F in full.values()),'Complete ordinary-queen filtering gate failed'
    save(OUT/'ordinary-filter-seal.json',dict(utc=now(),all_supports_contain_ordinary_queen_F=True,supports=len(full),complete_ordinary_states=len(ordinary),independent_ordinary_and_literalF_enumeration_equal=True))
    cells=13*np.arange(13)[None,:]+ordinary.astype(np.int64);ii,jj=np.triu_indices(13,1);ai,aj=np.triu_indices(169,1);infos=[]
    for record in source['supports']:
        group=record['group'];family=record['family'];F=full[group];truth,Ftrue,keys,truekeys=references[family];columns=ordinary[F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,compiled(F,f'support{group}'));valid=np.array([row.tobytes()in truekeys for row in columns]);h=hist(columns[valid]);flags=h[keys]>0;folder=OUT/f'support{group}';np.savez_compressed(folder/'capacity.npz',columns=columns,valid_state_mask=valid,key_capacity_bits=np.packbits(flags,bitorder='little'));info=dict(group=group,family=family,complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(flags.sum()),capacity_pct=100*float(flags.mean()),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),exact_graph=bool(np.array_equal(F,Ftrue)),capacity_sha256=sha(folder/'capacity.npz'));infos.append(info);print('SUPPORT',group,info,flush=True)
    rows=[]
    for source_row in source['rows']:
        info=infos[source_row['group']];primary=source_row['role']=='primary';eligible=source_row.get('source_eligible',True);rows.append(dict(source_row,**{key:value for key,value in info.items()if key not in('group','family')},family=source_row['index']//4,bank=source_row['index']%4,primary_exact_success=primary and eligible and info['exact_graph'],decision=('successful_fit'if eligible and info['exact_graph']else'failed_fit')if primary else'descriptive'))
    # BP first within each family/dose/bank; count identity is not an independent fit.
    rows.sort(key=lambda r:(r['family'],r['dose'],r['bank'],0 if r['method'].startswith('BP')else 1))
    family_results=[]
    for family in(0,1):
        primary=[r for r in rows if r['family']==family and r['role']=='primary'];success=sum(r['primary_exact_success']for r in primary);assert len(primary)==4;family_results.append(dict(family=family,primary_dose=32,eligible_exact_BP_fits=success,total_BP_fits=4,required=3,decision='supported'if success>=3 else'falsified'))
    save(OUT/'evaluation.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in family_results)else'falsified',family_results=family_results,rows=rows,support_results=infos,source_outcomes_sha256=sha(OUT/'source-outcomes.json'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
