"""Portable metadata/ledger checks; full optimality replay requires external proofs."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from verify_complete import verify as verify_full_archive
from audit_costs import choose


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_manifest(root):
    root=Path(root).resolve();S=root/'research/20261114/loki/h3bw';P=root/'results/research/loki/h3bw';sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();read=lambda path:json.loads(path.read_text());s=read(P/'summary.json');g=read(P/'capacity-gate.json');p=read(S/'protocol.json');boundary=read(P/'archive-boundary.json');checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    check('complete_summary',s['status']=='validated' and s['decision']==s['audit']['primary_decision'] and s['cases']==4096 and s['new_policy_calls']==g['new_policy_calls'] and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check('frozen '+rel,sha(root/rel)==digest)
    for rel,digest in s['verifier_dependency_sha256'].items():check('verifier '+rel,sha(root/rel)==digest)
    for rel,digest in s['public_sha256'].items():check('public '+rel,sha(P/rel)==digest)
    check('proof_payload_boundary_explicit',boundary['public_package']=='compact_complete_study' and boundary['full_proof_payloads_in_git'] is False and boundary['full_verification_requires_external_archive'] and boundary['primary_decision']==s['decision'] and boundary['proof_manifest_sha256']==sha(P/'proof-export.json') and not any(P.rglob('proof.json.gz')) and not any(P.rglob('proof.json')))
    proof_manifest=read(P/'proof-export.json');check('all_omitted_payload_hashes_preserved',len(proof_manifest['proofs'])==proof_manifest['proof_files']==boundary['omitted_compressed_proof_files'] and sum(r['raw_bytes'] for r in proof_manifest['proofs'])==boundary['omitted_raw_proof_bytes'] and sum(r['compressed_bytes'] for r in proof_manifest['proofs'])==boundary['omitted_compressed_proof_bytes'])
    chain=load(root/'research/20261114/loki/h3bj/verify_public.py','h3bw_portable_ledger').chain;data=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(data);receipt=read(P/'ledger-export.json');check('ledger_bytes',hashlib.sha256(data).hexdigest()==receipt['raw_sha256']==s['ledger_sha256'] and len(data)==receipt['raw_bytes'] and sha(P/'query-ledger.jsonl.gz')==receipt['gzip_sha256'] and len(events)==s['ledger_records']);bits=np.load(root/'results/research/loki/h3bc/predictions-bits.npy');boards=np.load(root/'results/research/loki/h3bc/ordinary13.npy');pred=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);scores=read(P/'score-values8.json');sources=read(root/p['archive']['sources']);active={};queried={};paid=0;completed=[]
    for i,event in enumerate(events):
        d=event['data']
        if event['kind']=='archived_source_bound':
            case=d['case'];source=sources[case];check('source_input_binding',d==dict(case=case,candidate=source['candidate'],bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=source['initial_candidates']));active[case]=list(d['initial_candidates']);queried[case]=[]
        elif event['kind']=='query':
            case=d['case'];truth=sources[case]['candidate'];index=d['selection']['query_index'];candidates=active[case];answer=events[i+1];a=answer['data'];label=bool(pred[truth,index]);remaining=[c for c in candidates if bool(pred[c,index])==label];selection=d['selection'];expected={str(c):bool(pred[c,index]) for c in candidates};check('paid_query_label_and_candidate_filter',answer['kind']=='answer' and index not in queried[case] and d['step']==len(queried[case]) and d['request']['candidates']==candidates and d['request']['queried']==queried[case] and d['board_zero_based']==boards[index].tolist() and selection['candidate_predictions']==expected and a==dict(case=case,phase=d['phase'],step=d['step'],query_sha256=event['sha256'],label=label,candidates_before=candidates,candidates_after=remaining) and event['monotonic_ns']<answer['monotonic_ns'])
            if d['phase']=='map':check('independent_MAP_selection',selection==choose(pred,scores,candidates,queried[case]))
            else:check('tree_query_selection_fields',selection==dict(query_index=index,candidate_predictions=expected))
            active[case]=remaining;queried[case].append(index);paid+=1
        elif event['kind']=='case_result':
            case=d['case'];check('complete_case_ledger_count',d['remaining_candidates']==active[case] and d['membership_calls']==len(queried[case])<=8 and d['exact_true_singleton']==(active[case]==[sources[case]['candidate']]));completed.append(d)
    check('all_paid_queries_and_prefix_results',paid==g['new_policy_calls']==s['new_policy_calls'] and completed==read(P/'case-results.json') and len(completed)==4096 and read(P/'case-statuses.json')==[dict(case=i,status='complete') for i in range(4096)])
    old=read(root/p['archive']['cases']);old_calls=sum(row['arms']['posterior']['policy_calls'] for row in old);check('whole_cohort_primary_and_exact_flags',old_calls==10355 and s['decision']==('supported' if 100*paid<=98*old_calls else 'falsified') and all(new['exact_true_singleton']==prior['arms']['posterior']['exact_true_singleton'] for new,prior in zip(completed,old)) and sum(row['exact_true_singleton'] for row in completed)==4083)
    audit_receipt=read(P/'full-archive-audit.json');check('full_archive_audit_receipt',audit_receipt['audit']==s['audit'] and audit_receipt['proof_manifest_sha256']==sha(P/'proof-export.json') and audit_receipt['protocol_sha256']==sha(S/'protocol.json') and all(sha(root/rel)==digest for rel,digest in audit_receipt['audit_source_sha256'].items()))
    plot=read(P/'plot-provenance.json')
    for rel,digest in plot['input_sha256'].items():check('figure_input',sha(root/rel)==digest)
    for rel,digest in plot['output_sha256'].items():check('figure_output',sha(P/rel)==digest)
    return dict(status='passed_portable_metadata_and_ledger_audit',checks=checks,scientific_status='complete',primary_decision=s['decision'],actual_paid_queries_replayed=paid,completed_case_records_verified=len(completed),full_optimality_certificates_replayed=False,limitation='Full normalized optimality and counterfactual exact-flag proof replay requires the separately retained external archive; its prior complete audit receipt and every raw/compressed proof hash are published')


def main(root,archive_dir=None):
    portable=verify_manifest(root);result=dict(portable=portable)
    if archive_dir is not None:result['external_full_archive_audit']=verify_full_archive(root,archive_dir)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);parser.add_argument('--archive-dir',type=Path,help='Optional separately retained full export, required to replay omitted optimality proofs');args=parser.parse_args();main(args.root,args.archive_dir)
