"""Revalidate all population masks and scores before any fresh source draw."""
import hashlib
import itertools
import math
import shutil
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,load


def run(p):
    rule=load(SRC.parent/'h3at/source.py','h3bx_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']]
    choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)]
    old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];h12=rule.incidence(ordinary,orbits)
    B=ROOT/'results/research/loki/h3bc';boards=np.load(B/'ordinary13.npy');bits=np.load(B/'predictions-bits.npy');oldprep=read(B/'preparation.json');h13=rule.incidence(boards,orbits);sourcepred=[];sourcepop={};scores={8:[],16:[]}
    assert read(B/'summary.json')['audit']['all1541N12_masks_and_N13_literal_populations_verified']
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[m12];F12=rule.rule_graph(12,[orbits[j] for j in chosen])
        assert np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
        sourcepred.append(m12);sourcepop[cid]=columns
        for m in [8,16]:scores[m].append(str(math.comb(55,len(chosen))*math.comb(len(columns),m)) if len(columns)>=m else None)
        m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(m13,bitorder='little');F13=rule.rule_graph(13,[orbits[j] for j in chosen])
        assert np.array_equal(packed,bits[cid]) and oldprep['populations'][cid]==dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
    assert scores[8]==read(ROOT/'results/research/loki/h3bm/score-values8.json') and scores[16]==read(ROOT/'results/research/loki/h3bj/score-values.json')
    shutil.copyfile(B/'predictions-bits.npy',OUT/'predictions-bits.npy');save(OUT/'score-values8.json',scores[8]);save(OUT/'score-values.json',scores[16])
    history={}
    for r in read(ROOT/'results/research/loki/h3bm/source-pairs.json'):history.setdefault(r['unordered_sample_sha256']['8'],[]).append(dict(study='H3bm',case=r['case'],candidate=r['candidate']))
    for r in read(ROOT/'results/research/loki/h3bq/source-banks.json'):history.setdefault(r['unordered_sample_sha256'],[]).append(dict(study='H3bq',case=r['case'],candidate=r['candidate']))
    for r in read(ROOT/'results/research/loki/h3br/source-banks.json'):history.setdefault(r['unordered_sample_sha256'],[]).append(dict(study='H3br',case=r['case'],candidate=r['candidate']))
    from prior import prior_rows
    expected_prior=prior_rows([len(sourcepop[cid]) for cid in range(1541)])
    assert p['truths']==expected_prior and sum(r['prior_integer_weight'] for r in expected_prior)==4436
    save(OUT/'historical-eight-banks.json',history)
    save(OUT/'population-parity.json',dict(utc=now(),all1541_source_target_filters_and_scores_verified=True,census_summary_sha256=sha(B/'summary.json'),score8_sha256=sha(OUT/'score-values8.json'),score16_sha256=sha(OUT/'score-values.json'),historical_eight_banks_sha256=sha(OUT/'historical-eight-banks.json')))
    return ordinary,np.array(sourcepred,bool),sourcepop,boards,bits,scores[8],history
