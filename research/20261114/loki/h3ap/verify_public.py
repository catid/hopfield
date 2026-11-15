"""Verify the retained failure and the absence of any completed scored evidence."""
import argparse,hashlib,json
from pathlib import Path
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);R=a.parse_args().root;S=R/'research/20261114/loki/h3ap';P=R/'results/research/loki/h3ap';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();s=read(P/'summary.json');g=read(P/'capacity-gate.json');r=read(P/'study-result.json');p=read(S/'protocol.json');assert s['status']=='verified_technical_failure'and s['scientific_decision']=='inconclusive_not_evaluated'and s['scored_rows']==0 and not s['partial_artifacts_usable'];assert g['status']==r['status']=='failed_or_incomplete'and r['completed_rows']==r['completed_supports']==[]and not(P/'source-outcomes.json').exists()and not(P/'construction-safety-seal.json').exists();assert sha(S/'protocol.json')==s['protocol_sha256']==g['protocol_sha256'];assert 'TypeError: Object of type int64 is not JSON serializable'in(P/'campaign.log').read_text()
    for rel,digest in p['source_sha256'].items():assert sha(R/rel)==digest,rel
    for rel,digest in s['public_sha256'].items():assert sha(P/rel)==digest,rel
    print(json.dumps(dict(status='passed',scientific_decision=s['scientific_decision'],source_hashes=len(p['source_sha256']),export_hashes=len(s['public_sha256']),target_stage_not_reached=True,zero_scored_rows=True)))
if __name__=='__main__':main()
