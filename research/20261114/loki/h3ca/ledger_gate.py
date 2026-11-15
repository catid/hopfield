"""Synthetic archival parity and durability gate before any new source draw."""
import hashlib
import json
import os
import time
from common import ROOT,SRC,OUT,read,save,sha,now
from ledger import Ledger,audit_bytes,canonical


def run(protocol,select):
    archive=[(read(ROOT/q),read(ROOT/a)) for q,a in protocol['ledger_gate']['archival_query_answer_paths']];legacy=OUT/'legacy-format-gate';legacy.mkdir();new=OUT/'synthetic-ledger.jsonl';ledger=Ledger(new);callbacks=0;equivalent=[];old_seconds=0.;new_seconds=0.
    def old_seal(path,value):
        with path.open('xb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    for i,(q,a) in enumerate(archive):
        selection=select(q['request']);assert selection==q['selection'];query=dict(index=i,request=q['request'],selection=selection,board_zero_based=q['board_zero_based']);t=time.perf_counter();old_seal(legacy/f'q{i}.json',query);old_seconds+=time.perf_counter()-t;t=time.perf_counter();qrecord=ledger.append('query',query);new_seconds+=time.perf_counter()-t
        # The oracle callback checks the actual durable file before exposing its label.
        committed=audit_bytes(new.read_bytes());assert committed[-1]['sha256']==qrecord['sha256'] and committed[-1]['kind']=='query';callback_time=time.monotonic_ns();assert callback_time>qrecord['monotonic_ns'];label=a['label'];callbacks+=1
        logical_answer=dict(index=i,label=label,candidates_before=a['candidates_before'],candidates_after=a['candidates_after']);t=time.perf_counter();old_seal(legacy/f'a{i}.json',logical_answer);old_seconds+=time.perf_counter()-t;t=time.perf_counter();arecord=ledger.append('answer',dict(logical_answer,query_sha256=qrecord['sha256']));new_seconds+=time.perf_counter()-t
        assert read(legacy/f'q{i}.json')==qrecord['data'] and read(legacy/f'a{i}.json')=={k:v for k,v in arecord['data'].items() if k!='query_sha256'};equivalent.append(dict(archival_query_sha256=sha(ROOT/protocol['ledger_gate']['archival_query_answer_paths'][i][0]),archival_answer_sha256=sha(ROOT/protocol['ledger_gate']['archival_query_answer_paths'][i][1]),new_query_record_sha256=qrecord['sha256'],new_answer_record_sha256=arecord['sha256'],logical_fields_identical=True,oracle_callback_monotonic_ns=callback_time))
    ledger.close();records=audit_bytes(new.read_bytes());assert len(records)==16;mutants={}
    def encode(rows):return b''.join(canonical(row)+b'\n' for row in rows)
    clone=lambda:json.loads(json.dumps(records))
    x=clone();x[1]['data']['label']=not x[1]['data']['label'];mutants['changed_answer_without_rehash']=encode(x)
    x=clone();x[0],x[1]=x[1],x[0];mutants['reordered_query_answer']=encode(x)
    x=clone();x[1]['previous_sha256']='1'*64;mutants['broken_hash_chain']=encode(x)
    x=clone();x[1]['data']['query_sha256']='2'*64
    # Rechain an internally hashed but semantically invalid answer reference.
    for i,row in enumerate(x):row.pop('sha256');row['previous_sha256']=x[i-1]['sha256'] if i else '0'*64;row['sha256']=hashlib.sha256(canonical(row)).hexdigest()
    mutants['rehashed_wrong_query_reference']=encode(x);rejected={}
    for name,data in mutants.items():
        try:audit_bytes(data)
        except AssertionError:rejected[name]=True
        else:raise AssertionError(name+' was not rejected')
    result=dict(utc=now(),all8_prior_format_logical_queries_and_answers_identical=True,oracle_callbacks_observed_durable_query_before_label=callbacks,tamper_cases_rejected=rejected,paired_format_parity=equivalent,legacy16files_write_fsync_wall_seconds=old_seconds,ledger16appends_write_fsync_wall_seconds=new_seconds,synthetic_ledger_sha256=sha(new),source_sha256=sha(__file__),ledger_source_sha256=sha(SRC/'ledger.py'));save(OUT/'ledger-gate.json',result);return result
