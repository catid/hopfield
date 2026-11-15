"""Bind the reused H1an sources before new permutation-noise observations."""
import json
from run import ROOT,HERE,sha,save,now
from permuted_noise_common import paths,load_protocol

def main():
    out,pp=paths();p=load_protocol(require_registration=False);assert not out.exists();out.mkdir()
    source=json.loads((ROOT/p['inputs']).read_text());assert len(source['inputs'])==32
    original=json.loads((ROOT/'artifacts/research/20261114/ripper/balanced_filter/audit.json').read_text());assert original['status']=='validated'
    for entry in source['inputs']:
        for role in ('calibration','training'):assert sha(ROOT/entry[role+'_path'])==entry[role+'_sha256']
    save(out/'input_registration.json',dict(status='registered_before_new_queries',utc=now(),protocol_sha256=sha(pp),preparer_sha256=sha(__file__),
        source_manifest_sha256=sha(ROOT/p['inputs']),source_banks=32,new_source_boards=0,new_solver_inputs=0,archived_h1an_audit_sha256=sha(ROOT/'artifacts/research/20261114/ripper/balanced_filter/audit.json')))
    print(json.dumps(dict(status='registered',source_banks=32,new_sources=0)))
if __name__=='__main__':main()
