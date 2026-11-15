"""Export source identities and frozen pool provenance without board archives."""
from pathlib import Path
import json
import shutil
import numpy as np
from regularity_fresh_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,DESIGN,sha,save,verify


def main():
    p=json.loads(PROTOCOL.read_text());verify(p)
    summary=json.loads((PUBLIC/'summary.json').read_text());assert summary['status']=='validated'
    assert summary['protocol_sha256']==sha(PROTOCOL)
    source=summary['source_audit'];assert source['status']=='validated'
    ledger=json.loads((DESIGN/'ledger.json').read_text())
    assert sha(DESIGN/'ledger.json')==p['source_ledger_sha256']==source['source_ledger_sha256']
    assert sha(OUT/'source-draws.npz')==source['source_draws_sha256']
    with np.load(OUT/'source-draws.npz',allow_pickle=False) as z:rows=z['source_rows'].copy()
    excluded=np.load(DESIGN/'excluded.npy',allow_pickle=False);h4y=np.load(DESIGN/'h4y_ids.npy',allow_pickle=False)
    for name,array in [('excluded.npy',excluded),('h4y_ids.npy',h4y)]:assert sha(DESIGN/name)==ledger['array_sha256'][name]
    target=PUBLIC/'source-identities.npz';assert not target.exists()
    np.savez_compressed(target,source_rows=rows,excluded_ids=excluded,h4y_ids=h4y)
    shutil.copyfile(DESIGN/'ledger.json',PUBLIC/'source-ledger.json')
    save(PUBLIC/'source-provenance.json',dict(status='exported_from_validated_archive',protocol_sha256=sha(PROTOCOL),
        summary_sha256=sha(PUBLIC/'summary.json'),source_identities_sha256=sha(target),source_ledger_sha256=sha(PUBLIC/'source-ledger.json'),
        publisher_sha256=sha(Path(__file__)),raw_draws_sha256=source['source_draws_sha256'],
        scope='Public seeded source-ID, frozen exclusion-array and overlap replay. Original CSV labels, masks and full corruptions require the independent raw archive audit.'))
    print(json.dumps(dict(status='exported',source_rows=rows.size,unique_ids=len(np.unique(rows)),excluded_ids=len(excluded))))


if __name__=='__main__':main()
