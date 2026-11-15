"""Public nested-draw/design replay, learned native gate and complete-state audit."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

def graph(n,leaper=None):
    F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr,dc=abs(a//n-b//n),abs(a%n-b%n);F[a,b]=F[b,a]=not dr or not dc or dr==dc or(leaper is not None and(dr,dc)in(leaper,leaper[::-1]))
    return F
def hist(columns):
    h=np.zeros(170**3,np.int64);cells=13*np.arange(13)[None,:]+columns.astype(np.int64)
    for k in(1,2,3):
        ix=np.array(list(itertools.combinations(range(13),k)));keys=((cells[:,ix]+1)*np.array([170**(2-j)for j in range(k)])).sum(-1);h+=np.bincount(keys.ravel(),minlength=170**3)
    assert h.sum()==377*len(columns);return h
def reconstruct_design(d):
    canonical=[tuple(o)for o in d['offsets']];counts={o:0 for o in canonical};y=d['labels'];n=len(y)
    for board in y:
        for a,b in itertools.combinations(np.flatnonzero(board),2):counts[int(b)//12-int(a)//12,int(b)%12-int(a)%12]+=1
    marginal=y.sum(0).astype(np.int64);exposure=[sum(int(marginal[12*r+c])*int(marginal[12*(r+dr)+c+dc])for r in range(12)for c in range(12)if r+dr<12 and 0<=c+dc<12)/n for dr,dc in canonical];lines=[]
    for a in range(-2,3):
        for b in range(-2,3):
            if(a,b)==(0,0)or math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            on=[o for o in canonical if a*o[0]+b*o[1]==0]
            if len(on)>=3 and not any(counts[o]for o in on):lines.append((a,b))
    groups=[];nxt=len(lines)
    for dr,dc in canonical:
        match=[i for i,(a,b)in enumerate(lines)if a*dr+b*dc==0];assert len(match)<=1
        if match:groups.append(match[0])
        else:groups.append(nxt);nxt+=1
    assert np.array_equal([counts[o]for o in canonical],d['counts'])and np.array_equal(exposure,d['exposure'])and np.array_equal(lines,d['lines'])and np.array_equal(groups,d['groups'])
    retained=np.array([counts[o]==0 and exposure[i]>=math.log(264/.01)or any(a*o[0]+b*o[1]==0 for a,b in lines)for i,o in enumerate(canonical)]);lookup={o:i for i,o in enumerate(canonical)};F=np.zeros((169,169),np.int32)
    for a,b in itertools.combinations(range(169),2):
        o=(b//13-a//13,b%13-a%13);F[a,b]=F[b,a]=any(r*o[0]+c*o[1]==0 for r,c in lines)or(o in lookup and retained[lookup[o]])
    return F,canonical,lines,np.array(groups),retained

def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3ao';P=R/'results/research/loki/h3ao';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');banks=read(P/'source-banks.json');source=read(P/'source-outcomes.json');fits=read(P/'source-fits-seal.json');evaluation=read(P/'evaluation.json');filterseal=read(P/'ordinary-filter-seal.json')
    check('complete within all budgets',g['status']==raw['status']=='complete_verified'and g['aggregate_cpu_seconds']<=600 and g['aggregate_GPU_worker_seconds']<=120 and g['wall_seconds']<=900 and not g['monitor_expired'])
    check('prospective source then target order',p['frozen_utc']<banks['utc']<fits['utc']<source['utc']<filterseal['utc']<evaluation['utc'])
    check('protocol and receipt identities',sha(S/'protocol.json')==g['protocol_sha256']==source['protocol_sha256']and raw['source_fits_seal_sha256']==sha(P/'source-fits-seal.json')and evaluation['source_outcomes_sha256']==sha(P/'source-outcomes.json')and all(g[k]==raw[k]for k in raw if k!='utc'))
    for category in('source_sha256','source_data_sha256','target_data_sha256'):
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    populations={}
    for family,cfg in enumerate(p['families']):
        columns=np.load(R/cfg['source_path']);population=np.zeros((len(columns),144),bool);population[np.arange(len(columns))[:,None],12*np.arange(12)[None,:]+columns]=True;populations[family]=population
    expected_models={};eligibilities={}
    for index in range(8):
        family=index//4;folder=P/f'b{index}';indices=np.random.default_rng(p['sampling']['stream_seeds'][index]).choice(len(populations[family]),64,replace=False)
        check(f'stream{index}ordered draw',np.array_equal(indices,np.load(folder/'ordered64-indices.npy'))and indices.tolist()==banks['banks'][index]['ordered64_indices'])
        for dose in(16,32,64):
            d=np.load(folder/f'd{dose}/source-design.npz');model=np.load(folder/f'd{dose}/count-model.npz');check(f'stream{index}dose{dose}exact nested labels',np.array_equal(d['population_indices'],indices[:dose])and np.array_equal(d['labels'],populations[family][indices[:dose]]));F,canonical,lines,groups,retained=reconstruct_design(d);check(f'stream{index}dose{dose}literal design and uniform count control',np.array_equal(model['F'],F)and np.array_equal(model['J'],F.astype(np.float32))and not model['h'].any()and model['q']==13 and model['unit']==1);expected_models[index,dose]=F
            if dose!=32:continue
            learned=np.load(folder/'model.npz');params=np.load(folder/'parameters.npz');fit=read(folder/'model.json');gate=read(folder/'loss-validation.json');A=params['A'];b=params['b'];coefficients=A[groups];lookup={o:i for i,o in enumerate(canonical)};native=np.zeros((144,144),np.float64);J=np.zeros((169,169),np.float32)
            for a,c in itertools.combinations(range(144),2):native[a,c]=native[c,a]=float(coefficients[lookup[c//12-a//12,c%12-a%12]])
            margin=float(((1-2*d['labels'].astype(np.float64))*(d['labels'].astype(np.float64)@native+float(b[0]))).min());unit=float(coefficients[retained].mean())
            for a,c in itertools.combinations(range(169),2):
                if not F[a,c]:continue
                o=(c//13-a//13,c%13-a%13);line=next((j for j,(rr,cc)in enumerate(lines)if rr*o[0]+cc*o[1]==0),None);J[a,c]=J[c,a]=A[line]if line is not None else coefficients[lookup[o]]
            eligible=bool(np.isfinite(A).all()and np.isfinite(b).all()and np.isfinite(margin)and margin>0 and np.isfinite(unit)and unit>0 and np.isfinite(coefficients[retained]).all()and(coefficients[retained]>0).all()and np.isfinite(J).all()and(J[F.astype(bool)]>0).all());eligibilities[index]=eligible
            check(f'primaryfit{index}learned native gate and full transfer',fit['source_eligible']==eligible and fit['native_source_delta']==margin and fit['unit']==unit==learned['unit']and np.array_equal(F,learned['F'])and np.array_equal(J,learned['J'])and not learned['h'].any()and learned['q']==13)
            check(f'primaryfit{index}numerical training evidence',fit['updates']==4000 and fit['fit_seconds']<=30 and [r['update']for r in fit['curve']]==list(range(250,4001,250))and gate['status']=='passed'and gate['cpu_literal_loss_error']<1e-9 and gate['cpu_gpu_gradient_error']<1e-5 and gate['cpu_gpu_loss_error']<1e-5 and fit['parameters_sha256']==sha(folder/'parameters.npz')and fit['model_sha256']==fits['models'][str(index)]==sha(folder/'model.npz')and fit['source_design_sha256']==sha(folder/'source-design.npz')==sha(folder/'d32/source-design.npz'))
    check('no descriptive BP fit claims',source['new_BP_fits']==8 and source['unfitted_descriptive_BP_doses']==[16,64]and sum(r['role']=='primary'for r in source['rows'])==8 and len(source['rows'])==32)
    supports={r['group']:np.load(P/f"support{r['group']}/F.npz")['F']for r in source['supports']}
    for record in source['supports']:
        group=record['group'];check(f'support{group}hash',record['F_npz_sha256']==sha(P/f'support{group}/F.npz')and record['support_sha256']==hashlib.sha256(supports[group].tobytes()).hexdigest())
        for member in record['members']:check(f"support{group}member{member['index']}/{member['dose']}/{member['method']}",np.array_equal(supports[group],expected_models[member['index'],member['dose']]))
    check('ordinary-queen inclusion for every support',all((F>=graph(13)).all()for F in supports.values())and filterseal['all_supports_contain_ordinary_queen_F']and filterseal['supports']==len(supports))
    with tempfile.TemporaryDirectory(prefix='h3ao-public-')as tmp:
        tmp=Path(tmp);binary=tmp/'reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,n,name):
            inp=tmp/f'{name}.bin';out=tmp/f'{name}cols.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(result.stdout);return columns
        ordinary=enumerate_F(graph(13),13,'ordinary');check('complete ordinary queen count',len(ordinary)==filterseal['complete_ordinary_states']==73712);references={}
        for family,cfg in enumerate(p['families']):
            for n,rel in[(12,cfg['source_path']),(13,cfg['target_path'])]:
                truth=enumerate_F(graph(n,tuple(cfg['leaper'])),n,f'f{family}n{n}');check(f'family{family}N{n}complete population',np.array_equal(truth,np.load(R/rel)))
                if n==13:
                    h=hist(truth);keys=np.flatnonzero(h);u=np.load(P/f'family{family}/input-universe.npz');check(f'family{family}full key census',np.array_equal(keys,u['keys'])and np.array_equal(h[keys],u['counts']));references[family]=(truth,graph(13,tuple(cfg['leaper'])),keys,{row.tobytes()for row in truth})
        for record in source['supports']:
            group=record['group'];F=supports[group];truth,Ftrue,keys,truekeys=references[record['family']];columns=enumerate_F(F,13,f'support{group}');cap=np.load(P/f'support{group}/capacity.npz');valid=np.array([row.tobytes()in truekeys for row in columns]);h=hist(columns[valid]);flags=h[keys]>0;r=g['support_results'][group];ai,aj=np.triu_indices(169,1)
            check(f'support{group}complete states including invalid',np.array_equal(columns,cap['columns'])and np.array_equal(valid,cap['valid_state_mask'])and r['complete_zero_F_states']==len(columns)and r['valid_zero_F_states']==int(valid.sum())and r['invalid_zero_F_states']==int((~valid).sum()))
            check(f'support{group}integer capacity and exact graph',np.array_equal(flags,np.unpackbits(cap['key_capacity_bits'],bitorder='little')[:len(keys)])and r['covered_keys']==int(flags.sum())and r['capacity_pct']==100*float(flags.mean())and r['false_forbidden_edges']==int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum())and r['missed_true_edges']==int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum())and r['exact_graph']==bool(np.array_equal(F,Ftrue))and r['capacity_sha256']==sha(P/f'support{group}/capacity.npz'))
    for row in g['rows']:
        info=g['support_results'][row['group']];check(f"row{row['index']}/{row['dose']}/{row['method']}all capacity metrics",all(row[k]==v for k,v in info.items()if k not in('group','family')))
        if row['role']=='primary':check(f"primaryfit{row['index']}success flag",row['primary_exact_success']==(eligibilities[row['index']]and info['exact_graph']))
        else:check('descriptive row',row['decision']=='descriptive'and not row['primary_exact_success'])
    for family in(0,1):
        rows=[r for r in g['rows']if r['family']==family and r['role']=='primary'];success=sum(r['primary_exact_success']for r in rows);check(f'family{family}primary fixed denominator',len(rows)==4 and g['family_results'][family]==dict(family=family,primary_dose=32,eligible_exact_BP_fits=success,total_BP_fits=4,required=3,decision='supported'if success>=3 else'falsified'))
    check('overall frozen conjunction',g['decision']==('supported'if all(r['decision']=='supported'for r in g['family_results'])else'falsified'))
    if require_summary:
        s=read(P/'summary.json');check('summary identities',s['status']=='validated'and s['rows']==g['rows']and s['family_results']==g['family_results']and s['support_results']==g['support_results']and s['protocol_sha256']==sha(S/'protocol.json')and s['audit_source_sha256']==sha(S/'analyze.py')and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        check('figure identities',all(sha(R/rel)==digest for rel,digest in read(P/'plot-provenance.json')['input_sha256'].items()))
    return dict(status='passed',checks=len(checks),all8_ordered_source_streams_and24_nested_designs_replayed=True,all8_native_BP_gates_and_full_transfers_rebuilt=True,all_complete_populations_and5_supports_reenumerated=True,all_invalid_states_and_full_integer_key_censuses_verified=True,all_primary_and_descriptive_decisions_verified=True,elapsed_seconds=time.perf_counter()-start)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
