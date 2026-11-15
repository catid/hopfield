"""Replay new bank draws, fitted support construction and complete public censuses."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np

def graph(n,leaper):
    F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr,dc=abs(a//n-b//n),abs(a%n-b%n);F[a,b]=F[b,a]=not dr or not dc or dr==dc or(dr,dc)in(leaper,leaper[::-1])
    return F
def views(board):
    cells=np.flatnonzero(board);r,c=cells//12,cells%12;maps=[(r,c),(c,11-r),(11-r,11-c),(11-c,r),(r,11-c),(11-c,11-r),(11-r,c),(c,r)];values=[]
    for rr,cc in maps:v=np.zeros(144,bool);v[12*rr+cc]=True;values.append(v)
    return list({v.tobytes():v for v in values}.values())
def offsets(boards):return{(int(b)//12-int(a)//12,int(b)%12-int(a)%12)for board in boards for a,b in itertools.combinations(np.flatnonzero(board),2)}
def hist(columns):
    h=np.zeros(170**3,np.int64);cells=13*np.arange(13)[None,:]+columns.astype(np.int64)
    for k in(1,2,3):
        indices=np.array(list(itertools.combinations(range(13),k)));keys=((cells[:,indices]+1)*np.array([170**(2-j)for j in range(k)])).sum(-1);h+=np.bincount(keys.ravel(),minlength=170**3)
    assert h.sum()==377*len(columns);return h
def poset_mass(frequency,dose):
    # Different exact algorithm: closure, contained-union binomials and poset inversion.
    closure={0}
    for mask in frequency:closure|={old|mask for old in list(closure)}
    mass={}
    for mask in sorted(closure,key=lambda m:(m.bit_count(),m)):
        available=sum(count for typ,count in frequency.items()if typ|mask==mask);contained=math.comb(available,dose);exact=contained-sum(weight for smaller,weight in mass.items()if smaller|mask==mask);assert exact>=0;mass[mask]=exact
    assert sum(mass.values())==math.comb(sum(frequency.values()),dose)
    return{k:v for k,v in mass.items()if v}

def verify(root,require_summary=True):
    started=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3an';P=R/'results/research/loki/h3an';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,ok):assert ok,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');source=read(P/'source-outcomes.json');banks=read(P/'source-banks.json');seal=read(P/'source-fits-seal.json');validation=read(P/'source-validation.json');evaluation=read(P/'evaluation.json')
    check('complete budget',g['status']==raw['status']=='complete_verified'and g['aggregate_cpu_seconds']<=600 and g['aggregate_GPU_worker_seconds']<=120 and g['wall_seconds']<=900 and not g['monitor_expired'])
    check('protocol and sealed order',p['frozen_utc']<banks['utc']<seal['utc']<source['utc']<evaluation['utc']and sha(S/'protocol.json')==g['protocol_sha256']==source['protocol_sha256'])
    check('stage receipts',all(g[k]==raw[k]for k in raw if k!='utc')and raw['source_fits_seal_sha256']==sha(P/'source-fits-seal.json')and source['source_validation_sha256']==sha(P/'source-validation.json')and evaluation['source_outcomes_sha256']==sha(P/'source-outcomes.json'))
    for category in('source_sha256','source_data_sha256','target_data_sha256'):
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    populations={};models={};allmasses={};supports={r['group']:np.load(P/f"support{r['group']}/F.npz")['F']for r in source['supports']};members={group:[]for group in supports}
    for family,cfg in enumerate(p['families']):
        columns=np.load(R/cfg['source_path']);pop=np.zeros((len(columns),144),bool);pop[np.arange(len(pop))[:,None],12*np.arange(12)[None,:]+columns]=True;populations[family]=pop
    for index in range(8):
        family=index//4;folder=P/f'b{index}';d=np.load(folder/'source-design.npz');model=np.load(folder/'model.npz');fit=read(folder/'model.json');params=np.load(folder/'parameters.npz');indices=np.sort(np.random.default_rng(p['training']['bank_seeds'][index]).choice(len(populations[family]),128,replace=False));complement=np.setdiff1d(np.arange(len(populations[family])),indices);population=np.load(folder/'acquisition-population.npy');b=source['banks'][index]
        check(f'fit{index}training and complement draw replay',np.array_equal(indices,d['population_indices'])and indices.tolist()==banks['banks'][index]['training_indices']and np.array_equal(d['labels'],populations[family][indices])and np.array_equal(complement,np.load(folder/'acquisition-population-indices.npy'))and np.array_equal(population,populations[family][complement]))
        check(f'fit{index}fit eligibility and immutable outputs',fit['status']=='complete'and fit['source_eligible']and fit['updates']==4000 and fit['fit_seconds']<=30 and fit['parameters_sha256']==sha(folder/'parameters.npz')and fit['model_sha256']==seal['models'][str(index)]==sha(folder/'model.npz')and fit['source_design_sha256']==sha(folder/'source-design.npz')and [r['update']for r in fit['curve']]==list(range(250,4001,250)))
        loss=read(folder/'loss-validation.json');check(f'fit{index}numeric training gates',loss['status']=='passed'and loss['cpu_literal_loss_error']<1e-9 and loss['cpu_gpu_loss_error']<1e-5 and loss['cpu_gpu_gradient_error']<1e-5 and loss['original_loss_function_sha256']==sha(R/'research/20261114/loki/h3u/driver.py'))
        canonical=[tuple(o)for o in d['offsets']];counts={o:0 for o in canonical}
        for board in d['labels']:
            for a,c in itertools.combinations(np.flatnonzero(board),2):counts[int(c)//12-int(a)//12,int(c)%12-int(a)%12]+=1
        marginal=d['labels'].sum(0).astype(np.int64);exposure=[sum(int(marginal[12*r+c])*int(marginal[12*(r+dr)+c+dc])for r in range(12)for c in range(12)if r+dr<12 and 0<=c+dc<12)/128 for dr,dc in canonical];lines=[]
        for a in range(-2,3):
            for c in range(-2,3):
                if(a,c)==(0,0)or math.gcd(abs(a),abs(c))!=1 or a<0 or(a==0 and c<0):continue
                on=[o for o in canonical if a*o[0]+c*o[1]==0]
                if len(on)>=3 and not any(counts[o]for o in on):lines.append((a,c))
        groups=[];nxt=len(lines)
        for dr,dc in canonical:
            hits=[j for j,(a,c)in enumerate(lines)if a*dr+c*dc==0];assert len(hits)<=1
            if hits:groups.append(hits[0])
            else:groups.append(nxt);nxt+=1
        check(f'fit{index}literal source design',np.array_equal([counts[o]for o in canonical],d['counts'])and np.array_equal(exposure,d['exposure'])and np.array_equal(lines,d['lines'])and np.array_equal(groups,d['groups']))
        A=params['A'];unary=params['b'];retained=np.array([counts[o]==0 and exposure[j]>=math.log(264/.01)or any(a*o[0]+c*o[1]==0 for a,c in lines)for j,o in enumerate(canonical)]);coef=A[np.array(groups)];unit=float(coef[retained].mean());lookup={o:j for j,o in enumerate(canonical)};F=np.zeros((169,169),np.int32);J=np.zeros((169,169),np.float32);native=np.zeros((144,144),np.float64)
        for a,c in itertools.combinations(range(144),2):native[a,c]=native[c,a]=float(coef[lookup[c//12-a//12,c%12-a%12]])
        margin=float(((1-2*d['labels'].astype(np.float64))*(d['labels'].astype(np.float64)@native+float(unary[0]))).min())
        for a,c in itertools.combinations(range(169),2):
            o=(c//13-a//13,c%13-a%13);line=next((j for j,(rr,cc)in enumerate(lines)if rr*o[0]+cc*o[1]==0),None)
            if line is not None:F[a,c]=F[c,a]=1;J[a,c]=J[c,a]=A[line]
            elif o in lookup and retained[lookup[o]]:F[a,c]=F[c,a]=1;J[a,c]=J[c,a]=coef[lookup[o]]
        check(f'fit{index}native positivity and full transfer',margin>0 and margin==fit['native_source_delta']and np.array_equal(F,model['F'])and np.array_equal(J,model['J'])and not model['h'].any()and model['q']==13 and model['unit']==unit==fit['unit']and np.isfinite(A).all()and np.isfinite(unary).all()and(coef[retained]>0).all());models[index]=F
        rawseen=[offsets([board])for board in population];augseen=[offsets(views(board))for board in population];allseen=set().union(*augseen);classes=sorted((dr,dc)for dr,dc in allseen&set(canonical)if F[max(0,-dc),13*dr+max(0,dc)]);m=len(classes);check(f'fit{index}observable classes',classes==[tuple(o)for o in b['classes']]and m==b['class_count']<=16)
        rawm=np.array([sum(1<<j for j,o in enumerate(classes)if o in seen)for seen in rawseen],np.int64);augm=np.array([sum(1<<j for j,o in enumerate(classes)if o in seen)for seen in augseen],np.int64);pair=rawm|(augm<<m);types,n=np.unique(pair,return_counts=True);frequency={int(a):int(c)for a,c in zip(types,n)};patterns=np.load(folder/'source-patterns.npz');check(f'fit{index}paired source masks',np.array_equal(rawm,patterns['raw_masks'])and np.array_equal(augm,patterns['augmented_masks'])and np.array_equal(pair,patterns['paired_masks'])and frequency=={int(k):v for k,v in b['frequency'].items()});mass=poset_mass(frequency,32);check(f'fit{index}independent exact poset inversion',mass=={int(k):int(v)for k,v in b['draw_masses'].items()}and max(b['distribution_receipt']['peak_reachable_unions_by_dose'])<=512);allmasses[index]=mass
        for mask_text,group in b['mask_to_support'].items():
            mask=int(mask_text);repaired=F.copy();deleted={classes[j]for j in range(m)if mask&(1<<j)}
            for a,c in itertools.combinations(range(169),2):
                if(c//13-a//13,c%13-a%13)in deleted:repaired[a,c]=repaired[c,a]=0
            check(f'fit{index}mask{mask}full support',np.array_equal(repaired,supports[group]));members[group].append(dict(index=index,mask=mask))
        for fixture in[r for r in validation['all_concrete_learner_calls']if r['index']==index]:
            ix=np.random.default_rng(fixture['seed']).choice(len(population),32,replace=False);boards=[v for board in population[ix]for v in views(board)]if fixture['augment']else list(population[ix]);unique=list({v.tobytes():v for v in boards}.values());cost=fixture['cost'];check(f"fit{index}concrete arm{fixture['augment']}draw/viewcost",ix.tolist()==fixture['indices']and cost['raw_labels']==32 and cost['attempted_views']==(256 if fixture['augment']else 32)and cost['unique_views']==len(unique)and cost['witnessed_finite_offsets']==[list(o)for o in sorted(offsets(unique)&set(canonical))])
    check('all8fits and16learner calls',len(source['banks'])==8 and len(validation['all_concrete_learner_calls'])==16)
    for support in source['supports']:
        group=support['group'];check(f'support{group}identity and members',support['members']==members[group]and support['F_npz_sha256']==sha(P/f'support{group}/F.npz')and support['support_sha256']==hashlib.sha256(supports[group].tobytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='h3an-public-')as tmp:
        tmp=Path(tmp);binary=tmp/'reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True);references={}
        for family,cfg in enumerate(p['families']):
            for n,path in[(12,cfg['source_path']),(13,cfg['target_path'])]:
                F=graph(n,tuple(cfg['leaper']));inp=tmp/f'f{family}n{n}.bin';out=tmp/f'f{family}n{n}cols.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);expected=np.load(R/path);check(f'family{family}N{n}complete reenumeration',int(result.stdout)==len(expected)and np.array_equal(columns,expected))
                if n==13:
                    u=np.load(P/f'family{family}/input-universe.npz');h=hist(columns);keys=np.flatnonzero(h);check(f'family{family}complete key census',np.array_equal(keys,u['keys'])and np.array_equal(h[keys],u['counts']));references[family]=(columns,F,keys,{row.tobytes()for row in columns})
        for support in source['supports']:
            group=support['group'];truth,Ftrue,keys,truekeys=references[support['family']];F=supports[group];inp=tmp/f's{group}.bin';out=tmp/f's{group}cols.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);cap=np.load(P/f'support{group}/capacity.npz');valid=np.array([row.tobytes()in truekeys for row in columns]);h=hist(columns[valid]);flags=h[keys]>0;r=g['support_results'][group];ai,aj=np.triu_indices(169,1)
            check(f'support{group}complete states and validity',int(result.stdout)==len(columns)and np.array_equal(columns,cap['columns'])and np.array_equal(valid,cap['valid_state_mask']));check(f'support{group}full capacity census',np.array_equal(flags,np.unpackbits(cap['key_capacity_bits'],bitorder='little')[:len(keys)])and r['covered_keys']==int(flags.sum())and r['capacity_pct']==100*float(flags.mean())and r['invalid_zero_F_states']==int((~valid).sum())and r['meets_95_percent']==(20*int(flags.sum())>=19*len(keys))and r['false_forbidden_edges']==int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum())and r['missed_true_edges']==int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()))
    for row in g['rows']:
        index=row['index'];b=source['banks'][index];m=b['class_count'];mapping=b['mask_to_support'];den=math.comb(b['source_complement_size'],32);cats={key:0 for key in('neither','raw_only','augmented_only','both')};rawbad=augbad=0
        for pair,weight in allmasses[index].items():
            a=g['support_results'][mapping[str(pair&((1<<m)-1))]];c=g['support_results'][mapping[str(pair>>m)]];agood=a['meets_95_percent'];cgood=c['meets_95_percent'];cats[('both'if cgood else'raw_only')if agood else('augmented_only'if cgood else'neither')]+=weight;rawbad+=weight*(a['invalid_zero_F_states']>0);augbad+=weight*(c['invalid_zero_F_states']>0)
        rawprob=Fraction(cats['both']+cats['raw_only'],den);augprob=Fraction(cats['both']+cats['augmented_only'],den);gain=augprob-rawprob;check(f'fit{index}exact probabilities and paired counts',row['paired_categories']=={k:str(v)for k,v in cats.items()}and row['raw_probability']==dict(numerator=str(rawprob.numerator),denominator=str(rawprob.denominator))and row['augmented_probability']==dict(numerator=str(augprob.numerator),denominator=str(augprob.denominator))and row['raw_success_probability_pct']==100*float(rawprob)and row['augmented_success_probability_pct']==100*float(augprob)and row['gain_percentage_points']==100*float(gain)and int(row['raw_unsafe_draws'])==rawbad and int(row['augmented_unsafe_draws'])==augbad)
        safe=rawbad==augbad==0;reliable=augprob>=Fraction(9,10);improved=gain>=Fraction(1,10);check(f'fit{index}frozen decision',row['zero_invalid_states_for_every_positive_probability_outcome']==safe and row['reliability_threshold_met']==reliable and row['gain_threshold_met']==improved and row['decision']==('supported'if safe and reliable and improved else'falsified'))
    check('primary conjunction',g['decision']==('supported'if all(r['decision']=='supported'for r in g['rows'])else'falsified'))
    if require_summary:
        s=read(P/'summary.json');check('summary',s['status']=='validated'and s['rows']==g['rows']and s['support_results']==g['support_results']and s['protocol_sha256']==sha(S/'protocol.json')and s['audit_source_sha256']==sha(S/'analyze.py')and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        check('figure input identities',all(sha(R/rel)==digest for rel,digest in read(P/'plot-provenance.json')['input_sha256'].items()))
    return dict(status='passed',checks=len(checks),all8_new_bank_and_complement_draws_replayed=True,all8_source_designs_native_margins_and_full_fitted_transfers_rebuilt=True,all_source_and_target_populations_and_supports_reenumerated=True,all_exact_masses_checked_by_independent_poset_inversion=True,all_integer_capacity_censuses_and_decisions_verified=True,elapsed_seconds=time.perf_counter()-started)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
