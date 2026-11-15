"""Verify exported bytes independently against the immutable scientific source."""
import gzip
import hashlib
import json
from common import ROOT, SRC, OUT, frozen, read, sha, load


def main():
    frozen()
    public = ROOT/'results/research/loki/h3cb'
    adapter = load(SRC.parent/'archive_export_layer.py', 'h3cb_lossless_audit')
    archive = adapter.PublicArchive(public)
    validated = archive.validate_export()
    manifest = read(public/'proof-export.json')
    proofs = {row['logical_path']: row for row in manifest['proofs']}
    proof_seen, cost_seen = set(), set()
    metadata = source_files = source_bytes = 0
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or (path.parent == OUT and path.name in ['query-ledger.jsonl', 'predictions-bits.npy']):
            continue
        assert not path.is_symlink()
        relative = str(path.relative_to(OUT))
        digest = sha(path)
        source_files += 1
        source_bytes += path.stat().st_size
        if path.name == 'proof.json':
            row = proofs[relative]
            assert digest == row['raw_sha256'] and path.stat().st_size == row['raw_bytes']
            proof_seen.add(relative)
        elif adapter.is_counterfactual_json(relative):
            row = archive.entries[relative]
            assert digest == row['raw_sha256'] and path.stat().st_size == row['raw_bytes']
            cost_seen.add(relative)
        else:
            assert digest == sha(public/relative)
            metadata += 1
    assert proof_seen == set(proofs) and cost_seen == set(archive.entries)
    ledger_receipt = read(public/'ledger-export.json')
    if (OUT/'query-ledger.jsonl').exists():
        ledger = gzip.decompress((public/'query-ledger.jsonl.gz').read_bytes())
        assert hashlib.sha256(ledger).hexdigest() == ledger_receipt['raw_sha256'] == sha(OUT/'query-ledger.jsonl')
        assert ledger_receipt['gzip_sha256'] == sha(public/'query-ledger.jsonl.gz')
    else:
        result=read(OUT/'study-result.json')
        assert result['status']=='failed_or_incomplete' and result['actual_membership_calls']==0
        assert ledger_receipt==dict(status='no_ledger_created_before_failure',actual_membership_calls=0)
    provenance = read(public/'export-provenance.json')
    assert provenance['export_source_sha256'] == sha(SRC/'export.py')
    assert provenance['parallel_export_source_sha256'] == sha(SRC/'parallel_export.py')
    assert provenance['adapter_source_sha256'] == sha(SRC.parent/'archive_export_layer.py')
    assert provenance['proof_adapter_source_sha256'] == sha(SRC.parent/'export_layer.py')
    archive.close()
    result = dict(status='passed_lossless_scientific_export', source_files=source_files,
                  source_bytes=source_bytes, metadata_files_identical=metadata,
                  all_source_proofs_and_counterfactual_documents_identical=True,
                  original_query_ledger_exactly_preserved=True, export_validation=validated,
                  proof_manifest_sha256=sha(public/'proof-export.json'),
                  counterfactual_manifest_sha256=sha(public/'counterfactual-export.json'),
                  protocol_sha256=sha(SRC/'protocol.json'), audit_source_sha256=sha(__file__))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
