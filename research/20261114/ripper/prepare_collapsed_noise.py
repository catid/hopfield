"""Register every reused source bank and archived iid control before new queries."""
import json
from run import ROOT, HERE, sha, save, now
from collapsed_noise_common import paths, load_protocol


def main():
    out,pp=paths();p=load_protocol(require_registration=False);assert not out.exists();out.mkdir()
    prior=ROOT/'artifacts/research/20261114/ripper/correlated_noise'
    original=json.loads((prior/'audit.json').read_text());assert original['status']=='validated'
    source=json.loads((ROOT/p['inputs']).read_text());assert len(source['inputs'])==32
    controls=[]
    for entry in source['inputs']:
        for role in ('calibration','training'):
            assert sha(ROOT/entry[role+'_path'])==entry[role+'_sha256']
        d=prior/f's{entry["seed"]}'/'iid'
        for name in ('receipt.json','audit.json'):
            assert sha(d/name)==original['artifact_sha256'][str((d/name).relative_to(ROOT))]
        receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['calibration_source_sha256']==entry['calibration_sha256'] and receipt['training_source_sha256']==entry['training_sha256']
        for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
        controls.append({'seed':entry['seed'],'directory':str(d.relative_to(ROOT)),
                         'receipt_sha256':sha(d/'receipt.json'),'audit_sha256':sha(d/'audit.json')})
    save(out/'input_registration.json',{'status':'registered_before_new_queries','utc':now(),'protocol_sha256':sha(pp),
        'preparer_sha256':sha(__file__),'source_manifest_sha256':sha(ROOT/p['inputs']),
        'archived_iid_controls':controls,'source_banks':32,'new_source_boards':0,'new_solver_inputs':0})
    print(json.dumps({'status':'registered','source_banks':32,'archived_iid_controls':32,'new_sources':0}),flush=True)


if __name__=='__main__':main()
