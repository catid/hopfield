"""Complete N13 capacity, with independent zero-F enumeration if inclusion fails."""
import itertools
import math
from fractions import Fraction
import subprocess
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now

def histogram(columns):
    h=np.zeros(170**3,np.int64);cells=13*np.arange(13)[None,:]+columns.astype(np.int64)
    for k in(1,2,3):
        ix=np.array(list(itertools.combinations(range(13),k)));powers=np.array([170**(2-j)for j in range(k)],np.int64);keys=((cells[:,ix]+1)*powers).sum(-1);h+=np.bincount(keys.reshape(-1),minlength=170**3)
    assert int(h.sum())==len(columns)*377;return h

def forward(F):
    placed=[];answers=[]
    def visit(row):
        if row==13:answers.append([cell%13 for cell in placed]);return
        for col in range(13):
            cell=13*row+col
            if all(not F[cell,other]for other in placed):placed.append(cell);visit(row+1);placed.pop()
    visit(0);return np.array(answers,np.uint8).reshape(-1,13)

def true_graph(leaper):
    F=np.zeros((169,169),np.int32)
    for a,b in itertools.combinations(range(169),2):
        dr,dc=abs(a//13-b//13),abs(a%13-b%13);F[a,b]=F[b,a]=not dr or not dc or dr==dc or(dr,dc)in(leaper,leaper[::-1])
    return F

def main():
    p=frozen(True);source=read(OUT/'source-outcomes.json');assert source['status']=='complete_source_only';helper=load_generator();ordinary=helper.queens(13);assert len(ordinary)==73712;references={};build=OUT/'reverse_literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(build)],check=True)
    for family,cfg in enumerate(p['families']):
        columns=np.load(ROOT/cfg['target_path']);independent=helper.filter_leaper(ordinary,tuple(cfg['leaper']));assert np.array_equal(columns,independent);F=true_graph(tuple(cfg['leaper']));folder=OUT/f'family{family}';folder.mkdir(exist_ok=True);raw=folder/'true-F.bin';target=folder/'true-columns.bin';F.astype(np.uint8).tofile(raw);r=subprocess.run([str(build),'13',str(raw),str(target)],capture_output=True,text=True,check=True);assert int(r.stdout)==len(columns)and np.array_equal(columns,np.fromfile(target,np.uint8).reshape(-1,13));h=histogram(columns);keys=np.flatnonzero(h);np.savez_compressed(folder/'input-universe.npz',keys=keys,counts=h[keys],clues=(keys%170>0).astype(int)+(keys//170%170>0)+(keys//170**2>0));np.save(folder/'true-F.npy',F);np.save(folder/'true-columns.npy',columns);references[family]=(columns,F,keys,{row.tobytes()for row in columns});save(folder/'reference.json',dict(family=family,states=len(columns),keys=len(keys),complete_literal_and_ordinary_filter_agree=True,exclusions=0))
    infos=[]
    for support in source['supports']:
        group=support['group'];family=support['family'];truth,Ftrue,keys,truekeys=references[family];F=np.load(OUT/f'support{group}/F.npz')['F'];assert np.array_equal(F,F.T)and not np.diag(F).any()
        for a,b in itertools.combinations(range(169),2):
            if a//13==b//13 or a%13==b%13:assert F[a,b],'Row/column zero-state eligibility failed'
        inclusion=bool((F>=Ftrue).all());cells=13*np.arange(13)[None,:]+truth.astype(np.int64);ii,jj=np.triu_indices(13,1)
        if inclusion:columns=truth[F[cells[:,ii],cells[:,jj]].sum(1,dtype=np.int64)==0];valid=np.ones(len(columns),bool);proof='Ftrue_subset_F_complete_truth_filter'
        else:
            folder=OUT/f'support{group}';inp=folder/'F.bin';out=folder/'columns.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(build),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert int(result.stdout)==len(columns)and np.array_equal(columns,forward(F));valid=np.array([row.tobytes()in truekeys for row in columns]);proof='independent_forward_and_compiled_reverse_zero_F_enumeration'
        h=histogram(columns[valid]);covered=h[keys]>0;ai,aj=np.triu_indices(169,1);folder=OUT/f'support{group}';np.savez_compressed(folder/'capacity.npz',columns=columns,valid_state_mask=valid,key_capacity_bits=np.packbits(covered,bitorder='little'));info=dict(group=group,family=family,complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(covered.sum()),capacity_pct=100*float(covered.mean()),meets_95_percent=20*int(covered.sum())>=19*len(keys),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),F_contains_true_F=inclusion,completeness_proof=proof,capacity_sha256=sha(folder/'capacity.npz'));infos.append(info);print('CAPACITY',group,family,info['capacity_pct'],info['invalid_zero_F_states'],flush=True)
    rows=[]
    for bank in source['banks']:
        index=bank['index'];family=bank['family']
        if bank['status']!='complete_source_only':rows.append(dict(index=index,family=family,decision='falsified',reason='source_fit_ineligible'));continue
        m=bank['class_count'];mapping=bank['mask_to_support'];mass={int(k):int(v)for k,v in bank['draw_masses'].items()};den=math.comb(bank['source_complement_size'],32);assert sum(mass.values())==den;rawgood=auggood=rawbad=augbad=0;categories={key:0 for key in('neither','raw_only','augmented_only','both')}
        for pair,weight in mass.items():
            raw=infos[mapping[str(pair&((1<<m)-1))]];aug=infos[mapping[str(pair>>m)]];a=raw['meets_95_percent'];b=aug['meets_95_percent'];rawgood+=weight*a;auggood+=weight*b;rawbad+=weight*(raw['invalid_zero_F_states']>0);augbad+=weight*(aug['invalid_zero_F_states']>0);categories[('both'if b else'raw_only')if a else('augmented_only'if b else'neither')]+=weight
        rawprob=Fraction(rawgood,den);augprob=Fraction(auggood,den);gain=augprob-rawprob;safe=rawbad==augbad==0;reliable=augprob>=Fraction(9,10);improved=gain>=Fraction(1,10);baseline=infos[mapping['0']]
        rows.append(dict(index=index,family=family,bank=index%4,method='BP AdamW fixed new fit; paired raw versus D4 positive repair',decision='supported'if safe and reliable and improved else'falsified',baseline_capacity_pct=baseline['capacity_pct'],raw_success_probability_pct=100*float(rawprob),augmented_success_probability_pct=100*float(augprob),gain_percentage_points=100*float(gain),raw_favorable_draws=str(rawgood),augmented_favorable_draws=str(auggood),total_draws=str(den),raw_probability=dict(numerator=str(rawprob.numerator),denominator=str(rawprob.denominator)),augmented_probability=dict(numerator=str(augprob.numerator),denominator=str(augprob.denominator)),paired_categories={k:str(v)for k,v in categories.items()},raw_unsafe_draws=str(rawbad),augmented_unsafe_draws=str(augbad),zero_invalid_states_for_every_positive_probability_outcome=safe,reliability_threshold_met=reliable,gain_threshold_met=improved,source_training_labels=128,source_acquisition_labels=32,attempted_D4_views=256))
    save(OUT/'evaluation.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in rows)else'falsified',rows=rows,support_results=infos,source_outcomes_sha256=sha(OUT/'source-outcomes.json'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))

def load_generator():
    from common import load
    return load(SRC.parent/'h3ad/generate.py','h3an_independent_reference')
if __name__=='__main__':main()
