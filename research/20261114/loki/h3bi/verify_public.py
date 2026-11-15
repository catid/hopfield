"""Independent catalog, certificate and checkpoint replay; never imports compiler."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np


def hits(boards,orbits):
    out=np.zeros((len(boards),55),bool)
    for j,(a,b) in enumerate(orbits):
        for r in range(boards.shape[1]):
            for s,dc in [(r+a,b),(r+b,a)]:
                if s<boards.shape[1]:out[:,j]|=abs(boards[:,r].astype(np.int16)-boards[:,s].astype(np.int16))==dc
    return out


def parts(n,orbits):
    base=np.zeros((n*n,n*n),np.int32);extra=np.zeros((55,n*n,n*n),np.int32);lookup={o:j for j,o in enumerate(orbits)}
    for a,b in itertools.combinations(range(n*n),2):
        dr=abs(a//n-b//n);dc=abs(a%n-b%n)
        if dr==0 or dc==0 or dr==dc:base[a,b]=base[b,a]=1
        key=tuple(sorted((dr,dc)))
        if key in lookup:extra[lookup[key],a,b]=extra[lookup[key],b,a]=1
    return base,extra


def certificate(proof,catalog,check,prefix):
    """Check supplied proof locally; exhaustive alternatives certify each false node."""
    candidates=proof['candidates'];full=(1<<len(candidates))-1;depth=proof['depth_limit'];states=proof['states'];root=f'{full}:{depth}';visited=set()
    check(prefix+'root',proof['root_state']==root and candidates==sorted(set(candidates)))
    def visit(key):
        if key in visited:return states[key]['feasible']
        visited.add(key);node=states[key];mask,left=map(int,key.split(':'));size=mask.bit_count();attempts=node['attempts']
        check(prefix+key+'identity',node['mask']==mask and node['depth_remaining']==left and node['models']==size and left>=0 and mask>0 and mask&full==mask and isinstance(node['feasible'],bool))
        if size==1:check(prefix+key+'leaf',node['feasible'] and node['reason']=='singleton_leaf' and not attempts);return True
        if size>2**left:check(prefix+key+'information',not node['feasible'] and node['reason']=='binary_information_bound' and not attempts);return False
        # Enumerate all raw signatures independently, group unordered label splits,
        # then choose the earliest complete-universe board for each partition.
        grouped={}
        for entry in catalog:
            pos=mask&entry['signature'];neg=mask^pos
            if not pos or not neg:continue
            unordered=tuple(sorted((pos,neg)));old=grouped.get(unordered)
            if old is None or entry['lexfirst_board_index']<old[0]:grouped[unordered]=(entry['lexfirst_board_index'],pos,neg)
        options=[dict(canonical_side=min(pos,neg),positive_mask=pos,negative_mask=neg,board_index=index) for index,pos,neg in sorted(grouped.values())]
        check(prefix+key+'completepartitions',node['canonical_partition_count']==len(options) and len(attempts)<=len(options))
        won=False
        for j,attempt in enumerate(attempts):
            option=options[j];a=option['negative_mask'];b=option['positive_mask'];expected=dict(option)
            if max(a.bit_count(),b.bit_count())>2**(left-1):expected['result']='child_binary_information_bound'
            else:
                negative=f'{a}:{left-1}';expected['negative_state']=negative
                if not visit(negative):expected['result']='negative_child_infeasible'
                else:
                    positive=f'{b}:{left-1}';expected['positive_state']=positive
                    if not visit(positive):expected['result']='positive_child_infeasible'
                    else:expected['result']='both_children_feasible';won=True
            check(prefix+key+f'attempt{j}',attempt==expected and (not won or j==len(attempts)-1))
        if won:check(prefix+key+'firstfeasible',node['feasible'] and node['reason']=='feasible_split' and node['chosen_split']==options[len(attempts)-1] and node['negative_state']==attempts[-1]['negative_state'] and node['positive_state']==attempts[-1]['positive_state'])
        else:check(prefix+key+'exhaustion',not node['feasible'] and node['reason']=='all_nontrivial_partitions_exhausted' and len(attempts)==len(options))
        return won
    feasible=visit(root);check(prefix+'allproofstatesreachable',visited==set(states) and proof['feasible']==feasible);leaf_depths={};queries=set()
    if feasible:
        for local,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];split=node['chosen_split'];index=split['board_index'];check(prefix+f'candidate{cid}norepeats{len(used)}',index not in used);used.append(index);queries.add(index);key=node['positive_state'] if split['positive_mask']&(1<<local) else node['negative_state']
            check(prefix+f'candidate{cid}correctleaf',states[key]['mask']==1<<local and len(used)<=depth);leaf_depths[str(cid)]=len(used)
    check(prefix+'alldepths',proof['leaf_depths']==leaf_depths and proof['worst_leaf_depth']==(max(leaf_depths.values()) if feasible else None))
    return len(states),sum(len(node['attempts']) for node in states.values()),queries


def brute(mask,depth,signatures):
    if mask.bit_count()==1:return True
    if depth==0:return False
    return any((p:=mask&s) and (n:=mask^p) and brute(p,depth-1,signatures) and brute(n,depth-1,signatures) for s in signatures)


def verify(root,require_summary=True):
    started=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bi';P=R/'results/research/loki/h3bi';B=R/'results/research/loki/h3bc';F=R/'results/research/loki/h3bf';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');mapping=read(P/'checkpoint-groups.json');toy=read(P/'synthetic-gate.json');cases=read(F/'case-results.json');prior=read(F/'summary.json');census=read(B/'summary.json');prep=read(B/'preparation.json');old=read(R/p['normalizers']['selection_path'])
    check('wholebudget',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and g['global_memo_states']<=250000 and g['global_canonical_split_attempts']<=10000000 and p['budget']['GPUs']==0)
    check('stagehashes',all(g[k]==raw[k] for k in raw if k!='utc') and p['frozen_utc']<mapping['utc']<raw['utc'] and g['protocol_sha256']==mapping['protocol_sha256']==sha(S/'protocol.json') and g['source_sha256']==sha(S/'study.py') and g['checkpoint_groups_sha256']==sha(P/'checkpoint-groups.json') and g['synthetic_gate_sha256']==mapping['synthetic_gate_sha256']==sha(P/'synthetic-gate.json'))
    for group in ['source_sha256','source_data_sha256']:
        for rel,digest in p[group].items():check(group+' '+rel,sha(R/rel)==digest)
    check('inheritedliteralcensusandcheckpointaudit',census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and prior['audit']['all1408_source_seeds_and_unique_bank_counts_verified'] and census['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and prior['public_sha256']['case-results.json']==sha(F/'case-results.json'))
    code=ast.parse((S/'compiler.py').read_text());function=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='compile_tree');check('truthblindcompilerinterface',not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(code)) and [a.arg for a in function.args.args]==['candidates','catalog','depth','budget'])
    total_states=1;total_splits=0
    for row in toy['fixtures']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(row['signatures'])];proof=row['result'];check('toy '+row['name'],proof['depth_limit']==row['depth'] and proof['feasible']==row['expected']==bool(brute((1<<len(proof['candidates']))-1,row['depth'],row['signatures'])));ns,nt,_=certificate(proof,catalog,check,'toy '+row['name']);total_states+=ns;total_splits+=nt
    check('syntheticcapsandcounters',toy['resource_caps_raise_incomplete']==dict(state_cap=True,split_cap=True) and toy['all_independent_brute_force_results_identical'] and toy['global_states_after_gate']==total_states and toy['global_split_attempts_after_gate']==total_splits and len(toy['fixtures'])==8)
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];bits=np.load(B/'predictions-bits.npy');h12=hits(ordinary,orbits);h13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);sourcepred=[];sourcepop={}
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());check(f'candidate{cid}source',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid]);packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}target',np.array_equal(packed,bits[cid]) and record==prep['populations'][cid])
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(bits,axis=1,count=73712,bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};groups={};expected_map=[]
    for case in cases:
        truth=case['candidate'];sample=sourcepop[truth][np.random.default_rng(case['seed']).choice(len(sourcepop[truth]),16,replace=False)];candidates=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();check(f'case{case["case"]}initial',candidates==case['initial_candidates']);queried=[]
        for step in case['steps']:
            index=step['query_index'];label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];check(f'case{case["case"]}step{len(queried)}',step['label']==label and step['candidates_before']==candidates and step['candidates_after']==remaining and index not in queried);queried.append(index);candidates=remaining
        check(f'case{case["case"]}final',case['remaining_candidates']==candidates and case['exact_true_singleton']==(candidates==[truth]))
        if not case['exact_true_singleton']:
            check(f'case{case["case"]}scope',2<=len(candidates)<=16 and len(queried)==4 and all(np.all(predictions[candidates,index]==predictions[candidates[0],index]) for index in queried));groups.setdefault(tuple(candidates),[]).append(case['case'])
        expected_map.append(dict(case=case['case'],known_four_query_success=case['exact_true_singleton'],remaining_candidates=candidates,original_queried_board_indices=queried,tree_group=None))
    entries=[dict(group=i,candidates=list(key),case_ids=groups[key]) for i,key in enumerate(sorted(groups))]
    for entry in entries:
        for case in entry['case_ids']:expected_map[case]['tree_group']=entry['group']
    check('all1408mappings704unresolved',len(cases)==1408 and sum(c['exact_true_singleton'] for c in cases)==704 and mapping['groups']==entries and mapping['case_map']==expected_map and mapping['all1408_cases_retained'] and mapping['all704_unresolved_mapped'] and [r['case'] for r in expected_map if not r['known_four_query_success']]==p['primary_checkpoint_case_ids'])
    expected_rows=[];feasible_cases=0;leaves=0;real_states=0;real_splits=0
    for entry in entries:
        folder=P/f'group{entry["group"]}';row=read(folder/'result.json');stored=read(folder/'catalog.json');inp=read(folder/'compiler-input.json');proof=read(folder/'proof.json');candidates=entry['candidates'];codes=np.sum(predictions[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True)
        catalog=[dict(signature=int(s),lexfirst_board_index=int(np.flatnonzero(codes==s)[0]),board_count=int(count)) for s,count in zip(signatures,counts)];check(f'group{entry["group"]}completecatalog',stored==dict(candidates=candidates,universe_boards=73712,signatures=catalog) and sum(r['board_count'] for r in catalog)==73712);check(f'group{entry["group"]}blindinputs',inp==dict(candidates=candidates,catalog_sha256=sha(folder/'catalog.json'),depth=4,allowed_inputs_only=True) and proof['candidates']==candidates and proof['depth_limit']==4)
        ns,nt,queries=certificate(proof,catalog,check,f'group{entry["group"]}');total_states+=ns;total_splits+=nt;real_states+=ns;real_splits+=nt
        check(f'group{entry["group"]}hashesandcounters',all(row[k]==v for k,v in entry.items()) and row['feasible']==proof['feasible'] and row['worst_leaf_depth']==proof['worst_leaf_depth'] and row['signature_count']==len(catalog) and row['memo_states']==ns and row['canonical_split_attempts']==nt and row['catalog_sha256']==sha(folder/'catalog.json') and row['compiler_input_sha256']==sha(folder/'compiler-input.json') and row['proof_sha256']==sha(folder/'proof.json'))
        if proof['feasible']:
            feasible_cases+=len(entry['case_ids']);check(f'group{entry["group"]}noinheritedqueries',all(not queries.intersection(expected_map[c]['original_queried_board_indices']) for c in entry['case_ids']))
            for local,cid in enumerate(candidates):
                key=proof['root_state'];used=[]
                while proof['states'][key]['reason']!='singleton_leaf':
                    node=proof['states'][key];split=node['chosen_split'];index=split['board_index'];check(f'group{entry["group"]}actualprediction{cid}depth{len(used)}',bool(predictions[cid,index])==bool(split['positive_mask']&(1<<local)) and index not in used);used.append(index);key=node['positive_state'] if predictions[cid,index] else node['negative_state']
                check(f'group{entry["group"]}allcandidateleaf{cid}',proof['states'][key]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4);leaves+=1
        expected_rows.append(row)
    decision='supported' if feasible_cases==704 else 'falsified';check('wholeprimaryandcounters',g['groups']==expected_rows and g['decision']==decision and g['distinct_checkpoint_sets']==len(entries) and g['unresolved_cases']==704 and g['feasible_checkpoint_cases']==feasible_cases and g['infeasible_checkpoint_cases']==704-feasible_cases and g['known_success_cases']==704 and g['global_memo_states']==total_states and g['global_canonical_split_attempts']==total_splits and g['new_source_draws']==g['new_oracle_labels']==0)
    if require_summary:
        s=read(P/'summary.json');keys=['decision','distinct_checkpoint_sets','unresolved_cases','feasible_checkpoint_cases','infeasible_checkpoint_cases','known_success_cases','global_memo_states','global_canonical_split_attempts','new_source_draws','new_oracle_labels'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_and_original_policy_audit_inherited_by_hash=True,new_literal_enumerations=0,all1541_fresh_graph_filters_verified=True,all1408_source_checkpoints_reconstructed=True,all704_unresolved_mappings_and704_known_successes_verified=True,complete_catalogs_reconstructed=len(entries),every_visited_proof_state_and_exhaustion_verified=True,real_proof_states=real_states,real_split_attempts=real_splits,feasible_candidate_branches_verified=leaves,synthetic_infeasible_proofs_and_brute_force_verified=True,compiler_has_no_imports_or_truth_case_arguments=True,no_new_labels_or_draws=True,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
