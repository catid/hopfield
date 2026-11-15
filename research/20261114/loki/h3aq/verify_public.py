"""Replay the metadata fix, source constructions and every complete capacity set."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
def load(path,name):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def views(board):
    cells=np.flatnonzero(board);r,c=cells//12,cells%12;values=[]
    for rr,cc in[(r,c),(c,11-r),(11-r,11-c),(11-c,r),(r,11-c),(11-c,11-r),(11-r,c),(c,r)]:v=np.zeros(144,bool);v[12*rr+cc]=True;values.append(v)
    return list({v.tobytes():v for v in values}.values())
def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3aq';P=R/'results/research/loki/h3aq';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,ok):assert ok,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');source=read(P/'source-outcomes.json');safety=read(P/'construction-safety-seal.json');pre=read(P/'preflight.json');core=load(R/'research/20261114/loki/h3ao/verify_public.py','h3aq_census_helpers');fix=load(S/'preflight.py','h3aq_preflight_replay');replayed=fix.inner();check('exact metadata-only AST/text and synthetic schema replay',all(pre[k]==v for k,v in replayed.items())and pre==p['metadata_fix_preflight'])
    check('preserved unevaluated predecessor',read(R/'results/research/loki/h3ap/summary.json')['scientific_decision']=='inconclusive_not_evaluated'and p['technical_predecessor']['scientific_decision']=='not_evaluated'and sha(R/'research/20261114/loki/h3ap/protocol.json')==p['technical_predecessor']['frozen_protocol_sha256'])
    check('preflight and new run fully charged',g['aggregate_cpu_seconds']==g['preflight_CPU_seconds']+g['new_run_CPU_seconds']<=120 and g['wall_seconds']==g['preflight_charged_wall_seconds']+g['new_run_wall_seconds']<=180 and g['preflight_CPU_seconds']==pre['measured_CPU_seconds']and g['preflight_charged_wall_seconds']==pre['charged_wall_seconds']and pre['charged_wall_seconds']>=pre['measured_child_wall_seconds'])
    check('completed sealed source before target',g['status']==raw['status']=='complete_verified'and not g['monitor_expired']and p['frozen_utc']<source['utc']<safety['utc']<raw['utc']and sha(S/'protocol.json')==g['protocol_sha256']==source['protocol_sha256'])
    check('stage and source receipts',all(g[k]==raw[k]for k in raw if k!='utc')and raw['source_outcomes_sha256']==sha(P/'source-outcomes.json'))
    for category in('source_sha256','source_data_sha256','target_data_sha256'):
        for rel,digest in p[category].items():
            path=P/'preflight.json'if rel=='artifacts/research/20261114/loki/h3aq/preflight.json'else R/rel;check(category+' '+rel,sha(path)==digest)
    supports={r['group']:np.load(P/f"support{r['group']}/F.npz")['F']for r in source['supports']};members={group:[]for group in supports};prior=read(R/p['evaluation']['published_filtered_summary'])
    for index,cfg in enumerate(p['banks']):
        d=np.load(R/cfg['design_path']);canonical=set(map(tuple,d['offsets']));lines=d['lines'];sourceF=core.graph(12,tuple(p['families'][cfg['family']]['leaper']));check(f'bank{index}fixed16positive cardinality and inherited queen lines',d['labels'].shape==(16,144)and(d['labels'].sum(1)==12).all()and{(0,1),(1,-1),(1,0),(1,1)}<=set(map(tuple,lines)))
        for arm in('published_filtered','raw_unfiltered','D4_unfiltered'):
            model=np.load(P/f'b{index}/{arm}/model.npz');receipt=read(P/f'b{index}/{arm}/construction.json');row=next(r for r in source['rows']if r['index']==index and r['arm']==arm);group=row['group']
            if arm=='published_filtered':F=np.load(R/cfg['filtered_model_path'])['F'];check(f'bank{index}published filtered origin',receipt['identity_imported'])
            else:
                boards=[v for board in d['labels']for v in views(board)]if arm=='D4_unfiltered'else list(d['labels']);unique=list({v.tobytes():v for v in boards}.values());seen=set()
                for board in unique:
                    cells=np.flatnonzero(board)
                    for a,b in itertools.combinations(cells,2):assert not sourceF[a,b];seen.add((int(b)//12-int(a)//12,int(b)%12-int(a)%12))
                zero=canonical-seen;F=np.zeros((169,169),np.int32)
                for a,b in itertools.combinations(range(169),2):dr,dc=b//13-a//13,b%13-a%13;F[a,b]=F[b,a]=(dr,dc)in zero or any(r*dr+c*dc==0 for r,c in lines)
                check(f'bank{index}{arm}all witness and view records',receipt['raw_labels']==16 and receipt['attempted_views']==(128 if arm=='D4_unfiltered'else 16)and receipt['unique_views']==len(unique)and receipt['observed_finite_offsets']==[list(o)for o in sorted(seen&canonical)]and receipt['retained_zero_offsets']==[list(o)for o in sorted(zero)])
            check(f'bank{index}{arm}complete uniform model and support identity',np.array_equal(F,model['F'])and np.array_equal(F.astype(np.float32),model['J'])and not model['h'].any()and model['q']==13 and model['unit']==1 and np.array_equal(F,supports[group])and receipt['model_sha256']==row['model_sha256']==sha(P/f'b{index}/{arm}/model.npz'));members[group].append(dict(index=index,arm=arm))
    check('all24outcomes and8primary banks retained',len(source['rows'])==24 and sum(r['primary']for r in source['rows'])==8 and len(supports)<=24)
    for support in source['supports']:
        group=support['group'];check(f'support{group}hash and all logical members',support['members']==members[group]and support['F_npz_sha256']==sha(P/f'support{group}/F.npz')and support['support_sha256']==hashlib.sha256(supports[group].tobytes()).hexdigest())
    check('verified conditional safety construction before filtering',safety['status']=='verified_construction_condition'and safety['not_an_independent_empirical_safety_hypothesis']and all((supports[r['group']]>=core.graph(13,tuple(p['families'][r['family']]['leaper']))).all()for r in source['rows']if r['arm']!='published_filtered')and all((F>=core.graph(13)).all()for F in supports.values()))
    with tempfile.TemporaryDirectory(prefix='h3aq-public-')as tmp:
        tmp=Path(tmp);binary=tmp/'reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,name):
            inp=tmp/f'{name}.bin';out=tmp/f'{name}cols.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert len(columns)==int(result.stdout);return columns
        references={}
        for family,cfg in enumerate(p['families']):
            Ftrue=core.graph(13,tuple(cfg['leaper']));truth=enumerate_F(Ftrue,f'family{family}');check(f'family{family}complete truth',np.array_equal(truth,np.load(R/cfg['target_path'])));h=core.hist(truth);keys=np.flatnonzero(h);u=np.load(P/f'family{family}/input-universe.npz');check(f'family{family}complete key census',np.array_equal(keys,u['keys'])and np.array_equal(h[keys],u['counts']));references[family]=(truth,Ftrue,keys,{r.tobytes()for r in truth})
        for support in source['supports']:
            group=support['group'];F=supports[group];truth,Ftrue,keys,truekeys=references[support['family']];columns=enumerate_F(F,f'support{group}');cap=np.load(P/f'support{group}/capacity.npz');valid=np.array([r.tobytes()in truekeys for r in columns]);h=core.hist(columns[valid]);flags=h[keys]>0;r=g['support_results'][group];ai,aj=np.triu_indices(169,1)
            check(f'support{group}complete states including invalid baseline',np.array_equal(columns,cap['columns'])and np.array_equal(valid,cap['valid_state_mask'])and r['complete_zero_F_states']==len(columns)and r['valid_zero_F_states']==int(valid.sum())and r['invalid_zero_F_states']==int((~valid).sum()))
            check(f'support{group}all capacities and excess exclusions',np.array_equal(flags,np.unpackbits(cap['key_capacity_bits'],bitorder='little')[:len(keys)])and r['covered_keys']==int(flags.sum())and r['capacity_pct']==100*float(flags.mean())and r['true_state_retention_pct']==100*float(valid.sum()/len(truth))and r['false_forbidden_edges']==int(((F[ai,aj]>0)&(Ftrue[ai,aj]==0)).sum())and r['missed_true_edges']==int(((F[ai,aj]==0)&(Ftrue[ai,aj]>0)).sum())and r['meets_95_percent']==(20*int(flags.sum())>=19*len(keys))and r['capacity_sha256']==sha(P/f'support{group}/capacity.npz'))
    for row in g['rows']:
        info=g['support_results'][row['group']];check(f"bank{row['index']}{row['arm']}metrics and frozen decision",all(row[k]==v for k,v in info.items()if k not in('group','family'))and row['decision']==(('supported'if info['meets_95_percent']else'falsified')if row['primary']else'descriptive'))
        if row['arm']=='published_filtered':old=next(r for r in prior['rows']if r['index']==row['index']and r['dose']==16);check(f"bank{row['index']}published baseline values",all(row[k]==old[k]for k in('capacity_pct','covered_keys','invalid_zero_F_states','false_forbidden_edges','missed_true_edges')))
    check('all8capacity conjunction',g['primary']==[r for r in g['rows']if r['primary']]and len(g['primary'])==8 and g['decision']==('supported'if all(r['meets_95_percent']for r in g['primary'])else'falsified'))
    partial=[]
    for rel in('b0/published_filtered/model.npz','b0/raw_unfiltered/model.npz','support0/F.npz','support1/F.npz'):
        old=R/'results/research/loki/h3ap'/rel;new=P/rel;check('recomputed predecessor partial '+rel,sha(old)==sha(new));partial.append(dict(path=rel,predecessor_sha256=sha(old),successor_sha256=sha(new),byte_identical=True))
    if require_summary:
        s=read(P/'summary.json');check('summary identities',s['status']=='validated'and s['rows']==g['rows']and s['primary']==g['primary']and s['support_results']==g['support_results']and s['protocol_sha256']==sha(S/'protocol.json')and s['audit_source_sha256']==sha(S/'analyze.py')and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        check('figure inputs',all(sha(R/rel)==digest for rel,digest in read(P/'plot-provenance.json')['input_sha256'].items()))
    return dict(status='passed',checks=len(checks),metadata_only_AST_and_full_schema_preflight_replayed=True,all24source_models_and_view_receipts_reconstructed=True,all18_complete_supports_and_integer_key_censuses_rebuilt=True,conditional_safety_construction_verified=True,all_primary_decisions_recomputed=True,recomputed_predecessor_partial_files=partial,elapsed_seconds=time.perf_counter()-start)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
