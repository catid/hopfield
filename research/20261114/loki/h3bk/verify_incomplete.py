"""Integrity-only check of a retained failed run; no scientific optimality audit."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    R=Path(root);S=R/'research/20261114/loki/h3bk';P=R/'results/research/loki/h3bk';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=read(S/'protocol.json');s=read(P/'summary.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');checks=0
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():assert sha(R/rel)==digest,rel;checks+=1
    for rel,digest in s['public_sha256'].items():assert sha(P/rel)==digest,rel;checks+=1
    assert s['status']=='incomplete' and s['primary_decision'] is None and not s['partial_results_usable'] and s['integrity_only_no_optimality_audit'];assert g['status']==raw['status']=='failed_or_incomplete' and '4300 digits' in g['error'] and g['error']==raw['error']==s['error'];assert all(g[k]==raw[k] for k in raw if k!='utc');assert s['protocol_sha256']==g['protocol_sha256']==sha(S/'protocol.json');assert s['completed_group_receipts']==len(g['completed_groups'])==125
    for row in g['completed_groups']:
        folder=P/f'group{row["group"]}';assert read(folder/'result.json')==row
        for name in ['weights','catalog','compiler-input','proof']:assert sha(folder/f'{name}.json')==row[name.replace('-','_')+'_sha256'];checks+=1
    return dict(status='passed',checks=checks+5,integrity_only=True,scientific_endpoint=None,optimality_not_audited=True,failed_run_retained=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
