"""Independently audit every hypothetical acquisition union, MAP tie and target census."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
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


def graph(n, orbits):
    F = np.zeros((n*n, n*n), np.int32)
    orbit_set = {tuple(sorted(o)) for o in orbits}
    for u in range(n*n):
        for v in range(u+1, n*n):
            a, b = abs(u//n-v//n), abs(u%n-v%n)
            F[u, v] = F[v, u] = a == 0 or b == 0 or a == b or tuple(sorted((a, b))) in orbit_set
    return F


def witnesses(columns, orbits):
    lookup = {tuple(o): j for j, o in enumerate(orbits)}
    hits = np.zeros((len(columns), len(orbits)), bool)
    for k, board in enumerate(columns):
        for a, b in itertools.combinations(range(len(board)), 2):
            orbit = tuple(sorted((b-a, abs(int(board[b])-int(board[a])))))
            if orbit in lookup:
                hits[k, lookup[orbit]] = True
    return hits


def histogram(columns):
    cells = 13*np.arange(13)[None, :] + columns.astype(np.int64)
    counts = np.zeros(170**3, np.int64)
    for width in (1, 2, 3):
        subsets = np.array(list(itertools.combinations(range(13), width)))
        keys = ((cells[:, subsets]+1)*np.array([170**(2-j) for j in range(width)])).sum(-1)
        counts += np.bincount(keys.ravel(), minlength=len(counts))
    assert counts.sum() == 377*len(columns)
    return counts


def acquisition_mass(frequency,dose):
    below=[math.comb(sum(count for pattern,count in enumerate(frequency) if pattern&~mask==0),dose) for mask in range(len(frequency))];out=[]
    for mask in range(len(frequency)):
        sub=mask;value=0
        while True:
            value+=(-1)**((mask^sub).bit_count())*below[sub]
            if sub==0:break
            sub=(sub-1)&mask
        out.append(value)
    return out


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3av';P=R/'results/research/loki/h3av';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,condition):assert condition,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');source=read(P/'source-selection.json');old=read(R/p['normalizers']['selection_path']);packed=np.load(R/p['normalizers']['bits_path'])['bits']
    check('completed within unchanged whole-study caps',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and p['budget']['GPUs']==0)
    check('source pattern seal before evaluator and receipt identities',p['frozen_utc']<source['utc']<raw['utc'] and source['status']=='complete_source_only' and source['no_true_source_acquisition_population_or_target_truth_read'] and sha(S/'protocol.json')==source['protocol_sha256']==g['protocol_sha256'] and g['source_selection_sha256']==sha(P/'source-selection.json') and all(g[k]==raw[k] for k in raw if k!='utc') and source['source_sha256']==sha(S/'source.py') and g['source_sha256']==sha(S/'study.py'))
    for category in ('source_sha256','source_data_sha256','target_data_sha256'):
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    orbits=list(itertools.combinations(range(1,12),2));choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)]
    check('fixed1541prior and hypothetical label accounting',p['candidate_prior']['orbits']==[list(o) for o in orbits] and len(source['candidates'])==1541 and p['acquisition']['initial']==p['acquisition']['additional']==16 and p['acquisition']['observed_new_labels']==0 and p['primary']==dict(probability=[19,20],capacity=[19,20],invalid_zero_states=0,all_banks_and_all_maximizers=True))
    from fractions import Fraction
    check('unchanged exact normalized hierarchical prior',sum(Fraction(1,3*math.comb(55,len(chosen))) for chosen in choices)==1)
    check('charged exhaustive small population parity gate',g['validation']==dict(fixtures=5,dose_cases=45,all_literal_subsets_mobius_and_grouped_DP_equal=True))
    supports={r['group']:np.load(P/f"support{r['group']}/model.npz")['F'] for r in source['supports']};members={group:[] for group in supports};patterns_by_bank=[];populations={};infos=[];resultrows=[]
    check('all support IDs within cap',set(supports)==set(range(len(supports))) and len(supports)<=3084)
    with tempfile.TemporaryDirectory(prefix='h3av-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,n,label):
            inp=tmp/f'{label}.bin';out=tmp/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],check=True,capture_output=True,text=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        ordinary=enumerate_F(graph(12,[]),12,'ordinary');check('complete N12 ordinary universe',ordinary.shape==(14200,12) and np.array_equal(ordinary,np.load(R/p['normalizers']['ordinary_path'])));incidence=witnesses(ordinary,orbits)
        for cid,chosen in enumerate(choices):
            rules=[orbits[j] for j in chosen];F=graph(12,rules);columns=enumerate_F(F,12,f'c{cid}');mask=~incidence[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);M=len(columns)
            check(f'candidate{cid}complete literal and packed normalizer',np.array_equal(columns,ordinary[mask]) and np.array_equal(np.packbits(mask,bitorder='little'),packed[cid]))
            base=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(o) for o in rules],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest())
            check(f'candidate{cid}all inherited source metadata',base==old['candidates'][cid])
            if M>=32:check(f'candidate{cid}exact sequential posterior identity',math.comb(M,16)*math.comb(M-16,16)==math.comb(M,32)*math.comb(32,16))
            score=math.comb(55,len(chosen))*math.comb(M,32) if M>=32 else None
            check(f'candidate{cid}all pooled32 posterior fields',source['candidates'][cid]==dict(base,orbit_count=len(chosen),models_in_class=math.comb(55,len(chosen)),sample32_eligible=M>=32,MAP32_integer_score_without_common3=str(score) if score is not None else None))
            for family,cfg in enumerate(p['families']):
                if rules==[tuple(cfg['leaper'])]:populations[family]=columns
        inherited=read(R/'research/20261114/loki/h3ao/protocol.json')
        for index,cfg in enumerate(p['banks']):
            bank=source['banks'][index];population=populations[index//4];draw=np.random.default_rng(inherited['sampling']['stream_seeds'][index]).choice(len(population),64,replace=False)[:16];columns=population[draw];d=np.load(R/cfg['design_path']);labels=np.zeros((16,144),bool);labels[np.arange(16)[:,None],12*np.arange(12)[None,:]+columns]=True;evidence=np.load(P/f'b{index}/initial-evidence.npz');seen=witnesses(columns,orbits)
            check(f'bank{index}original uniform initialdraw and witnesses',np.array_equal(draw,d['population_indices']) and np.array_equal(labels,d['labels']) and np.array_equal(columns,evidence['columns']) and np.array_equal(seen,evidence['orbit_incidence']) and bank['initial_evidence_sha256']==sha(P/f'b{index}/initial-evidence.npz'))
            unwitnessed=np.flatnonzero(~seen.any(0)).tolist();initial=[cid for cid,chosen in enumerate(choices) if not seen[:,chosen].any()];check(f'bank{index}all initiallycompatible candidates and full witnessmask vocabulary',initial==bank['initial_consistent_candidates'] and unwitnessed==bank['unwitnessed_orbit_indices'] and all(source['candidates'][cid]['complete_source_cardinality']>=16 for cid in initial));check(f'bank{index}all abstractmasks retained',len(bank['patterns'])==1<<len(unwitnessed));patterns=[]
            for mask in range(1<<len(unwitnessed)):
                observed={unwitnessed[bit] for bit in range(len(unwitnessed)) if mask&(1<<bit)};consistent=[cid for cid in initial if observed.isdisjoint(choices[cid])];eligible=[cid for cid in consistent if source['candidates'][cid]['complete_source_cardinality']>=32];best=min(int(source['candidates'][cid]['MAP32_integer_score_without_common3']) for cid in eligible);winners=[cid for cid in eligible if int(source['candidates'][cid]['MAP32_integer_score_without_common3'])==best];actual=bank['patterns'][mask]
                expected=dict(mask=mask,newly_witnessed_orbit_indices=sorted(observed),consistent_candidates=consistent,eligible32_candidates=eligible,ineligible32_candidates=[cid for cid in consistent if cid not in eligible],best_MAP32_integer_score_without_common3=str(best),maximizers=actual['maximizers'],unique=len(winners)==1)
                check(f'bank{index}mask{mask}all posterior choices and ties',expected==actual and winners==[r['candidate'] for r in actual['maximizers']])
                for m in actual['maximizers']:
                    check(f'bank{index}mask{mask}candidate{m["candidate"]}fullgraph',np.array_equal(supports[m['group']],graph(13,[orbits[j] for j in choices[m['candidate']]])));members[m['group']].append(dict(index=index,mask=mask,candidate=m['candidate']))
                patterns.append(expected)
            patterns_by_bank.append(patterns)
        references={}
        for family,cfg in enumerate(p['families']):
            folder=P/f'family{family}';Ftrue=graph(13,[tuple(cfg['leaper'])]);truth=enumerate_F(Ftrue,13,f'f{family}truth');population=populations[family];check(f'family{family}fullsourceandtargettruth',np.array_equal(population,np.load(R/cfg['source_path'])) and np.array_equal(population,np.load(folder/'source-population.npy')) and np.array_equal(truth,np.load(R/cfg['target_path'])) and np.array_equal(truth,np.load(folder/'true-columns.npy')) and np.array_equal(Ftrue,np.load(folder/'true-F.npy')));h=histogram(truth);keys=np.flatnonzero(h);u=np.load(folder/'input-universe.npz');check(f'family{family}all integerclue census',np.array_equal(keys,u['keys']) and np.array_equal(h[keys],u['counts']));references[family]=(truth,Ftrue,keys,set(map(bytes,truth)))
        for record in source['supports']:
            group=record['group'];folder=P/f'support{group}';F=supports[group];model=np.load(folder/'model.npz');check(f'support{group}full uniformmodel and allmembers includingzeromass',record['members']==members[group] and record['model_sha256']==sha(folder/'model.npz') and np.array_equal(F.astype(np.float32),model['J']) and not model['h'].any() and model['q']==13 and model['unit']==1 and all(m['index']//4==record['family'] for m in members[group]));truth,Ftrue,keys,truekeys=references[record['family']];columns=enumerate_F(F,13,f'support{group}');valid=np.array([bytes(row) in truekeys for row in columns]);flags=histogram(columns[valid])[keys]>0;cap=np.load(folder/'capacity.npz');check(f'support{group}completevalidinvalidstatesandcapacitybits',np.array_equal(columns,cap['columns']) and np.array_equal(valid,cap['valid_state_mask']) and np.array_equal(np.packbits(flags,bitorder='little'),cap['key_capacity_bits']));ai,aj=np.triu_indices(169,1)
            info=dict(group=group,family=record['family'],complete_zero_F_states=len(columns),valid_zero_F_states=int(valid.sum()),invalid_zero_F_states=int((~valid).sum()),true_states=len(truth),input_keys=len(keys),covered_keys=int(flags.sum()),capacity_pct=100*float(flags.mean()),meets_95_percent=20*int(flags.sum())>=19*len(keys),exact_graph=bool(np.array_equal(F,Ftrue)),false_forbidden_edges=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum()),missed_true_edges=int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum()),capacity_sha256=sha(folder/'capacity.npz'));check(f'support{group}all capacitymetrics',info==g['support_results'][group]);infos.append(info)
        for bank in source['banks']:
            index=bank['index'];family=index//4;folder=P/f'b{index}';population=populations[family];initial=set(map(bytes,np.load(folder/'initial-evidence.npz')['columns']));complement=np.array([row for row in population if bytes(row) not in initial],np.uint8);orbitsub=[orbits[j] for j in bank['unwitnessed_orbit_indices']];hit=witnesses(complement,orbitsub);masks=(hit.astype(np.int64)*(1<<np.arange(len(orbitsub)))).sum(1);frequency=np.bincount(masks,minlength=1<<len(orbitsub));evidence=np.load(folder/'acquisition-evidence.npz');check(f'bank{index}entire evaluator complement andjointwitnesses',len(complement)==len(population)-16 and np.array_equal(complement,evidence['complement_columns']) and np.array_equal(masks,evidence['board_masks']) and np.array_equal(frequency,evidence['frequency']));mass=acquisition_mass(frequency.tolist(),16);total=math.comb(len(complement),16);check(f'bank{index}fullconditionalmass normalization',all(value>=0 for value in mass) and sum(mass)==total);d=read(folder/'exact-distribution.json');outcomes=[];success=exact=uniqueexact=0
            for pattern,weight in zip(patterns_by_bank[index],mass):
                selected=[infos[m['group']] for m in pattern['maximizers']];event=all(r['invalid_zero_F_states']==0 and r['meets_95_percent'] for r in selected);exactevent=all(r['exact_graph'] for r in selected)
                if weight:check(f'bank{index}mask{pattern["mask"]}positiveprobabilitycardinalityfeasible',not pattern['ineligible32_candidates'])
                success+=weight*event;exact+=weight*exactevent;uniqueexact+=weight*(exactevent and pattern['unique']);outcomes.append(dict(mask=pattern['mask'],subset_count=str(weight),probability_numerator=str(weight),probability_denominator=str(total),primary_success=event,all_maximizers_exact_graph=exactevent,unique=pattern['unique'],selected_candidates=[m['candidate'] for m in pattern['maximizers']],selected_groups=[m['group'] for m in pattern['maximizers']]))
            check(f'bank{index}everyexactmassandeventincludingzeros',outcomes==d['outcomes'] and d['total_subsets']==str(total) and d['complement_size']==len(complement) and d['initial_labels']==d['hypothetical_additional_labels']==16 and d['all_masks_retained_including_zero_mass'] and d['mobius_and_grouped_DP_all_masses_identical'] and d['acquisition_evidence_sha256']==sha(folder/'acquisition-evidence.npz'))
            gcd=math.gcd(success,total);row=dict(index=index,family=family,bank=index%4,complement_size=len(complement),initial_labels=16,hypothetical_additional_labels=16,new_observed_labels=0,total_subsets=str(total),successful_subsets=str(success),reduced_success_numerator=str(success//gcd),reduced_success_denominator=str(total//gcd),success_probability=success/total,success_pct=100*success/total,all_maximizers_exact_graph_subsets=str(exact),all_maximizers_exact_graph_probability=exact/total,unique_exact_graph_subsets=str(uniqueexact),unique_exact_graph_probability=uniqueexact/total,abstract_masks=len(mass),positive_mass_masks=sum(v>0 for v in mass),decision='supported' if 20*success>=19*total else 'falsified',distribution_sha256=sha(folder/'exact-distribution.json'));check(f'bank{index}allprimaryprobabilitymetrics',row==g['rows'][index]);resultrows.append(row)
        check('all8bankconjunction',len(resultrows)==len(source['banks'])==8 and g['decision']==('supported' if all(r['decision']=='supported' for r in resultrows) else 'falsified'))
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ('decision','rows','support_results')) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        f=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in f['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in f['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_sourcepopulations_reenumerated=True,all_sequential_posterior_identities_and_allties_verified=True,all104abstractmasks_and_exact_conditional_masses_verified=True,all14complete_supports_including_zeromass_invalidstates_reenumerated=True,all8exact_reliability_thresholds_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


