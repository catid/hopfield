"""Independent coordinate augmentation, paired grouped-DP and integer census audit."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np


def coordinate_views(board):
    cells=np.flatnonzero(board);r,c=cells//13,cells%13
    maps=[(r,c),(c,12-r),(12-r,12-c),(12-c,r),(r,12-c),(12-c,12-r),(12-r,c),(c,r)]
    result=[]
    for rows,cols in maps:
        view=np.zeros(169,bool);view[13*rows+cols]=True;result.append(view)
    return list({view.tobytes():view for view in result}.values())


def literal_offsets(boards):
    answer=set()
    for board in boards:
        for a,b in itertools.combinations(np.flatnonzero(board),2):
            answer.add((int(b)//13-int(a)//13,int(b)%13-int(a)%13))
    return answer


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);P=R/'results/research/loki/h3am';S=R/'research/20261114/loki/h3am';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,condition):assert condition,name;checks.append(name)
    spec=importlib.util.spec_from_file_location('h3al_public_auditor',R/'research/20261114/loki/h3al/verify_public.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    inherited_audit=core.verify(R)
    check('inherited complete source and target evidence reverified',inherited_audit['status']=='passed')
    p=read(S/'protocol.json');raw=read(P/'study-result.json');gate=read(P/'capacity-gate.json');source=read(P/'source-outcomes.json');validation=read(P/'source-validation.json');seal=read(P/'filtering-completeness-seal.json')
    check('complete within budget',raw['status']==gate['status']=='complete_verified'and gate['aggregate_cpu_seconds']<=120 and gate['wall_seconds']<=180 and not gate['monitor_expired'])
    check('protocol identities',sha(S/'protocol.json')==raw['protocol_sha256']==source['protocol_sha256'])
    check('source and stage identities',raw['source_sha256']==sha(S/'study.py')and source['source_sha256']==sha(S/'source_outcomes.py')and validation['learner_sha256']==sha(S/'learner.py'))
    check('paired source sealed before evaluation',source['utc']<seal['utc']<raw['utc']and p['frozen_utc']<source['utc'])
    check('receipts match',raw['source_outcomes_sha256']==sha(P/'source-outcomes.json')and source['source_validation_sha256']==sha(P/'source-validation.json')and all(raw[k]==gate[k]for k in raw if k!='utc'))
    for rel,digest in p['source_sha256'].items():check('source '+rel,sha(R/rel)==digest)
    for rel,digest in p['target_data_sha256'].items():check('target '+rel,sha(R/rel)==digest)
    learner_tree=ast.parse((S/'learner.py').read_text());imports=[node for node in ast.walk(learner_tree)if isinstance(node,(ast.Import,ast.ImportFrom))]
    check('learner imports only numpy and itertools',all(isinstance(node,ast.Import)and all(alias.name in ('numpy','itertools')for alias in node.names)for node in imports))
    calls=[node.func for node in ast.walk(learner_tree)if isinstance(node,ast.Call)]
    check('no learner I/O or dynamic execution',not any((isinstance(f,ast.Name)and f.id in ('open','eval','exec','__import__','globals','locals'))or(isinstance(f,ast.Attribute)and f.attr in ('load','save','savez','savez_compressed','read','write','read_bytes','write_bytes','tofile','fromfile'))for f in calls))
    population=np.load(R/'results/research/loki/h3al/source-population-n13.npy');population_keys={board.tobytes()for board in population}
    check('same frozen population',sha(R/'results/research/loki/h3al/source-population-n13.npy')==p['source_population']['sha256'])
    raw_seen=[];aug_seen=[];view_counts=[]
    for board in population:
        transformed=coordinate_views(board)
        assert all(view.tobytes()in population_keys for view in transformed)
        raw_seen.append(literal_offsets([board]));aug_seen.append(literal_offsets(transformed));view_counts.append(len(transformed))
    check('all D4 views independently in complete valid population',True)
    check('deduplicated view counts',validation['unique_views_per_board_histogram']=={str(k):view_counts.count(k)for k in sorted(set(view_counts))})
    supports={r['group']:np.load(P/f"support{r['group']}/F.npz")['F']for r in source['supports']};members={group:[]for group in supports};cache={};bank_masks={}
    for bank,cfg in enumerate(p['models']):
        b=source['banks'][bank];model_path=R/cfg['model_path'].replace('artifacts/research/20261114/loki','results/research/loki');design_path=R/cfg['source_design_path'].replace('artifacts/research/20261114/loki','results/research/loki');check(f'bank{bank}inherited input identities',sha(model_path)==p['source_data_sha256'][cfg['model_path']]and sha(design_path)==p['source_data_sha256'][cfg['source_design_path']]);base=np.load(model_path)['F'];finite={tuple(x)for x in np.load(design_path)['offsets']};seen=set().union(*aug_seen);classes=sorted((dr,dc)for dr,dc in seen&finite if base[max(0,-dc),15*dr+max(0,dc)]);m=len(classes);check(f'bank{bank}classes',classes==[tuple(x)for x in b['classes']]and m==b['class_count']<=6)
        raw_masks=np.array([sum(1<<i for i,o in enumerate(classes)if o in seen)for seen in raw_seen]);aug_masks=np.array([sum(1<<i for i,o in enumerate(classes)if o in seen)for seen in aug_seen]);pairs=raw_masks|(aug_masks<<m);frequency=np.bincount(pairs,minlength=1<<(2*m));patterns=np.load(P/f'b{bank}/source-patterns.npz')
        check(f'bank{bank}all independent paired masks',np.array_equal(raw_masks,patterns['raw_masks'])and np.array_equal(aug_masks,patterns['augmented_masks'])and np.array_equal(pairs,patterns['paired_masks'])and np.array_equal(frequency,patterns['frequency'])and frequency.tolist()==b['frequency']and np.array_equal(patterns['unique_views_per_board'],view_counts))
        check(f'bank{bank}patterns hash',sha(P/f'b{bank}/source-patterns.npz')==b['source_patterns_sha256'])
        key=tuple(frequency)
        if key not in cache:cache[key]=core.grouped_union_dp(frequency.tolist(),32)[32]
        masses=cache[key];check(f'bank{bank}all paired masses match independent DP',masses==list(map(int,b['draw_masses']))and all(weight>=0 for weight in masses)and sum(masses)==int(b['total_draws'])==math.comb(1932,32))
        check(f'bank{bank}raw union subset augmented for every positive mass',all(not weight or ((pair&((1<<m)-1))|(pair>>m))==pair>>m for pair,weight in enumerate(masses)))
        for mask,group in enumerate(b['mask_to_support']):
            F=base.copy();deleted={classes[i]for i in range(m)if mask&(1<<i)}
            for a,c in itertools.combinations(range(225),2):
                if(c//15-a//15,c%15-a%15)in deleted:F[a,c]=F[c,a]=0
            check(f'bank{bank}mask{mask}complete repaired support',np.array_equal(F,supports[group]));members[group].append(dict(bank=bank,mask=mask))
        bank_masks[bank]=(raw_masks,aug_masks,finite)
    check('all1792paired masks and32supports retained',sum(len(b['draw_masses'])for b in source['banks'])==source['total_paired_bank_masks']==1792 and len(supports)==source['distinct_full_supports']==32 and len({F.tobytes()for F in supports.values()})==32)
    for record in source['supports']:
        group=record['group'];check(f'support{group}identity and members',record['members']==members[group]and record['F_npz_sha256']==sha(P/f'support{group}/F.npz')and record['support_sha256']==hashlib.sha256(supports[group].tobytes()).hexdigest())
    for i,fixture in enumerate(validation['concrete32_calls']):
        bank=fixture['bank'];indices=np.random.default_rng(fixture['seed']).choice(1932,32,replace=False);raw_masks,aug_masks,finite=bank_masks[bank];augment=fixture['augment'];boards=[view for board in population[indices]for view in coordinate_views(board)]if augment else list(population[indices]);unique=list({view.tobytes():view for view in boards}.values());mask=int(np.bitwise_or.reduce((aug_masks if augment else raw_masks)[indices]));cost=fixture['cost'];check(f'concrete call{i}independent draw, views and witness union',indices.tolist()==fixture['indices']and mask==fixture['union_mask']and cost['raw_positive_labels']==32 and cost['attempted_views']==(256 if augment else 32)and cost['unique_views']==len(unique)and [list(o)for o in sorted(literal_offsets(unique)&finite)]==cost['witnessed_finite_classes'])
    check('all24 concrete learner calls retained',len(validation['concrete32_calls'])==24)
    ref=R/p['evaluation']['reference_root'];Ftrue=np.load(ref/'truth/F.npy');truth=np.load(ref/'truth/cache.npz')['columns'];u=np.load(ref/'input-universe.npz');keys=u['eligible_keys'];clues=u['eligible_clues'];cells=15*np.arange(15)[None,:]+truth.astype(np.int64);ai,aj=np.triu_indices(225,1)
    check('all support inclusions before filtering',all((F>=Ftrue).all()for F in supports.values())and seal['all_repaired_F_contain_true_F']and seal['supports']==32)
    capacities={}
    for group,F in supports.items():
        counts=np.zeros(len(truth),np.int64)
        for i,j in itertools.combinations(range(15),2):counts+=F[cells[:,i],cells[:,j]]
        valid=counts==0;h=core.histogram(truth[valid]);flags=h[keys]>0;cap=np.load(P/f'support{group}/capacity.npz');r=raw['support_results'][group];covered=int(flags.sum());false=int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum())
        check(f'support{group}all packed states and integer key census',np.array_equal(valid,np.unpackbits(cap['valid_state_bits'],bitorder='little')[:len(valid)])and np.array_equal(flags,np.unpackbits(cap['eligible_capacity_bits'],bitorder='little')[:len(flags)]))
        check(f'support{group}metrics',r['valid_zero_F_states']==int(valid.sum())and r['covered_eligible_inputs']==covered and r['capacity_pct']==100*float(flags.mean())and r['state_retention_pct']==100*float(valid.mean())and r['meets_95_percent']==(20*covered>=19*len(keys))and r['false_forbidden_edges']==false and r['missed_true_edges']==0 and r['capacity_sha256']==sha(P/f'support{group}/capacity.npz'))
        for k in(1,2,3):part=clues==k;check(f'support{group}clue{k}',r['per_clue'][str(k)]==dict(eligible_inputs=int(part.sum()),covered_eligible_inputs=int(flags[part].sum())))
        capacities[group]=(covered,false)
    prior=read(R/p['evaluation']['unaugmented_reference_summary'])
    for row in raw['rows']:
        b=source['banks'][row['bank']];m=b['class_count'];mapping=b['mask_to_support'];masses=cache[tuple(b['frequency'])];den=math.comb(1932,32);cats={k:0 for k in('neither','raw_only','augmented_only','both')};rawexpected=augexpected=rawexact=augexact=0
        for pair,weight in enumerate(masses):
            a=capacities[mapping[pair&((1<<m)-1)]];bcap=capacities[mapping[pair>>m]];agood=20*a[0]>=19*len(keys);bgood=20*bcap[0]>=19*len(keys);key=('both'if bgood else'raw_only')if agood else('augmented_only'if bgood else'neither');cats[key]+=weight;rawexpected+=weight*a[0];augexpected+=weight*bcap[0];rawexact+=weight*(a[1]==0);augexact+=weight*(bcap[1]==0)
        rawgood=cats['both']+cats['raw_only'];auggood=cats['both']+cats['augmented_only'];gain=auggood-rawgood;rawprob=Fraction(rawgood,den);augprob=Fraction(auggood,den);gainprob=Fraction(gain,den)
        check(f"bank{row['bank']}paired categories and integer counts",cats=={k:int(v)for k,v in row['paired_categories'].items()}and cats['raw_only']==0 and int(row['raw_favorable_draws'])==rawgood and int(row['augmented_favorable_draws'])==auggood and int(row['gain_draws'])==gain and int(row['total_draws'])==den)
        for key,prob in [('raw_probability',rawprob),('augmented_probability',augprob),('gain_probability',gainprob)]:check(f"bank{row['bank']}{key}exact fraction",row[key]==dict(numerator=str(prob.numerator),denominator=str(prob.denominator)))
        check(f"bank{row['bank']}all probability metrics",row['raw_success_probability_pct']==100*float(rawprob)and row['augmented_success_probability_pct']==100*float(augprob)and row['gain_percentage_points']==100*float(gainprob)and row['raw_expected_capacity_pct']==100*float(Fraction(rawexpected,den*len(keys)))and row['augmented_expected_capacity_pct']==100*float(Fraction(augexpected,den*len(keys)))and row['raw_exact_graph_probability_pct']==100*float(Fraction(rawexact,den))and row['augmented_exact_graph_probability_pct']==100*float(Fraction(augexact,den)))
        reliable=10*auggood>=9*den;improved=10*gain>=den;check(f"bank{row['bank']}both rational thresholds",row['augmented_reliability_at_least90_percent']==reliable and row['gain_at_least10_percentage_points']==improved and row['decision']==('supported'if reliable and improved else'falsified'))
        old=next(r for r in prior['rows']if r['bank']==row['bank']and r['dose']==32);check(f"bank{row['bank']}prior raw probability",rawprob==Fraction(int(old['success_probability_numerator']),int(old['success_probability_denominator'])))
    check('all four decisions',len(raw['rows'])==4 and raw['decision']==('supported'if all(r['decision']=='supported'for r in raw['rows'])else'falsified'))
    if require_summary:
        summary=read(P/'summary.json');check('summary values',summary['status']=='validated'and summary['rows']==raw['rows']and summary['support_results']==raw['support_results']and summary['protocol_sha256']==sha(S/'protocol.json'))
        check('audit identities',summary['audit_source_sha256']==sha(S/'analyze.py')and summary['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in summary['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        check('figure identities',all(sha(R/rel)==digest for rel,digest in read(P/'plot-provenance.json')['input_sha256'].items()))
    return dict(status='passed',checks=len(checks),inherited_complete_evidence_audit=inherited_audit,all_D4_views_independently_reconstructed=True,all1792_paired_integer_masses_match_grouped_DP=True,all32_supports_and_integer_key_censuses_rebuilt=True,all24_concrete_learner_calls_replayed=True,every_paired_category_and_rational_threshold_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
