"""Bounded complete-source alias partition and next-size behavior identifiability."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import resource
import signal
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();stages={};family_results=[]
    def expired(*_):raise TimeoutError('H3az600CPU900wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3az_rule');ordinaryhelper=load(SRC.parent/'h3ad/generate.py','h3az_ordinary');old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);bits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)];incidence=rule.incidence(ordinary,orbits);candidates=[];classes=[];lookup={};modelclass={}
        for cid,chosen in enumerate(choices):
            mask=~incidence[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[mask];F=rule.rule_graph(12,[orbits[j] for j in chosen]);record=old['candidates'][cid];assert np.array_equal(np.packbits(mask,bitorder='little'),bits[cid]);assert record==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest());candidates.append(record);key=bits[cid].tobytes()
            if key not in lookup:lookup[key]=len(classes);classes.append(dict(class_id=len(classes),source_population_size=len(columns),eligible_M_at_least16=len(columns)>=16,members=[]))
            group=lookup[key];classes[group]['members'].append(cid);modelclass[cid]=group
        families=[]
        for cfg in p['families']:
            chosen=tuple(sorted(orbits.index(tuple(o)) for o in cfg['true_orbits']));cid=choices.index(chosen);group=modelclass[cid];families.append(dict(family=cfg['index'],true_orbits=cfg['true_orbits'],true_candidate=cid,N12_population=candidates[cid]['complete_source_cardinality'],eligible=candidates[cid]['complete_source_cardinality']>=16,N12_class=group,N12_alias_members=classes[group]['members']))
        relevant=sorted({r['N12_class'] for r in families if r['eligible'] and len(r['N12_alias_members'])>1});N13models=sorted({cid for group in relevant for cid in classes[group]['members']});eligibleclasses=[c for c in classes if c['eligible_M_at_least16']];partition=dict(status='complete_N12_partition',utc=now(),candidates=candidates,classes=classes,families=families,relevant_nontrivial_classes=relevant,N13_model_ids=N13models,descriptive_all1541=dict(total_models=1541,total_equivalence_classes=len(classes),eligible_models=sum(len(c['members']) for c in eligibleclasses),eligible_classes=len(eligibleclasses),eligible_aliased_models=sum(len(c['members']) for c in eligibleclasses if len(c['members'])>1),eligible_nontrivial_classes=sum(len(c['members'])>1 for c in eligibleclasses)),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'N12-partition.json',partition);print('PARTITION',partition['descriptive_all1541'],'relevant',len(relevant),'N13models',len(N13models),flush=True)
        assert len(N13models)<=p['budget']['N13_model_cap'],'N13 model cap reached; no class pruning';binary=OUT/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True);populations={}
        def literal(F,n,label):
            inp=OUT/f'{label}-F.bin';out=OUT/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        def enumerate_stage(n,models):
            if not models:return []
            boards=ordinaryhelper.queens(n);assert len(boards)=={13:73712,14:365596}[n];assert np.array_equal(boards,literal(rule.rule_graph(n,[]),n,f'N{n}ordinary'));cells=n*np.arange(n)[None,:]+boards.astype(np.int64);a,b=np.triu_indices(n,1);records=[]
            for cid in models:
                F=rule.rule_graph(n,[orbits[j] for j in choices[cid]]);columns=boards[F[cells[:,a],cells[:,b]].sum(1,dtype=np.int64)==0];assert np.array_equal(columns,literal(F,n,f'N{n}c{cid}'));folder=OUT/f'N{n}/candidate{cid}';folder.mkdir(parents=True,exist_ok=True);np.save(folder/'columns.npy',columns);np.save(folder/'F.npy',F);populations[cid,n]=columns;records.append(dict(candidate=cid,n=n,states=len(columns),columns_sha256=sha(folder/'columns.npy'),F_sha256=sha(folder/'F.npy')));print('ENUMERATED',n,cid,len(columns),flush=True)
            return records
        stages['N13']=enumerate_stage(13,N13models);refined=[];modelrefined={}
        for group in relevant:
            sub={}
            for cid in classes[group]['members']:sub.setdefault(populations[cid,13].tobytes(),[]).append(cid)
            for members in sub.values():
                record=dict(refined_class=len(refined),N12_class=group,N13_states=len(populations[members[0],13]),members=members);refined.append(record)
                for cid in members:modelrefined[cid]=record['refined_class']
        N14models=sorted({cid for r in refined if len(r['members'])>1 for cid in r['members']});save(OUT/'pooled-N12-N13-partition.json',dict(utc=now(),refined_classes=refined,N14_model_ids=N14models,N13_stage_records=stages['N13'],N12_partition_sha256=sha(OUT/'N12-partition.json')));print('POOLED_CLASSES',len(refined),'N14models',len(N14models),flush=True);assert len(N14models)<=p['budget']['N14_model_cap'],'N14 model cap reached; no class pruning';stages['N14']=enumerate_stage(14,N14models);behavior=[];witnesses=[]
        for record in refined:
            members=record['members']
            if len(members)==1:behavior.append(dict(refined_class=record['refined_class'],members=members,N14_measured=False,N14_behavior_identified=True,reason='Unique model after pooled complete evidence.'));continue
            groups={}
            for cid in members:groups.setdefault(populations[cid,14].tobytes(),[]).append(cid)
            for a,b in itertools.combinations(members,2):
                A=populations[a,14];B=populations[b,14];As=set(map(bytes,A));Bs=set(map(bytes,B))
                for left,right,pop,other in [(a,b,A,Bs),(b,a,B,As)]:
                    difference=[row for row in pop if bytes(row) not in other]
                    if difference:witnesses.append(dict(refined_class=record['refined_class'],valid_candidate=left,invalid_candidate=right,directed_difference_size=len(difference),lexfirst_columns_zero_based=difference[0].tolist()))
            behavior.append(dict(refined_class=record['refined_class'],members=members,N14_measured=True,N14_behavior_identified=len(groups)==1,N14_behavior_classes=list(groups.values()),N14_population_sizes={str(cid):len(populations[cid,14]) for cid in members},reason='All surviving model state populations compared exactly.'))
        for family in families:
            if not family['eligible']:result=dict(family,decision='ineligible_N12_population_fewer_than16',case='ineligible',pooled_alias_members=None,N14_measured=False)
            elif len(family['N12_alias_members'])==1:result=dict(family,decision='supported',case='N12_unique_model_vacuous',pooled_alias_members=family['N12_alias_members'],N14_measured=False)
            else:
                group=modelrefined[family['true_candidate']];record=behavior[group];members=record['members'];case='N13_unique_model_vacuous' if len(members)==1 else 'remaining_alias_N14_behavior_compared';result=dict(family,decision='supported' if record['N14_behavior_identified'] else 'falsified',case=case,pooled_alias_members=members,N14_measured=record['N14_measured'],refined_class=group)
            family_results.append(result)
        save(OUT/'N14-behavior-comparison.json',dict(behavior_classes=behavior,witnesses=witnesses,all_directed_pair_differences_retained=True));eligible=[r for r in family_results if r['eligible']];decision='inconclusive_no_eligible_truths' if not eligible else 'supported' if all(r['decision']=='supported' for r in eligible) else 'falsified';save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,family_results=family_results,descriptive_all1541=partition['descriptive_all1541'],relevant_nontrivial_N12_classes=len(relevant),N13_models=len(N13models),N14_models=len(N14models),pooled_refined_classes=len(refined),stage_records=stages,N12_partition_sha256=sha(OUT/'N12-partition.json'),pooled_partition_sha256=sha(OUT/'pooled-N12-N13-partition.json'),behavior_comparison_sha256=sha(OUT/'N14-behavior-comparison.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_stages=stages,completed_family_results=family_results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
