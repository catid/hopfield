"""Read raw or losslessly exported science while tracking proof reachability."""
import gzip
import hashlib
import json
from pathlib import Path
from common import load


class AuditIO:
    def __init__(self, folder, source, check):
        self.folder = Path(folder).resolve()
        self.check = check
        self.proofs_seen = set()
        self.costs_seen = set()
        self.hashes = {}
        self.archive = None
        if (self.folder/'counterfactual-export.json').exists():
            adapter = load(source.parent/'archive_export_layer.py', 'h3bz_archive_adapter')
            self.archive = adapter.PublicArchive(self.folder)

    def data(self, path):
        path = Path(path).resolve()
        return self.archive.logical_bytes(path) if self.archive else path.read_bytes()

    def read(self, path):
        path = Path(path).resolve()
        data = self.data(path)
        self.hashes[str(path)] = hashlib.sha256(data).hexdigest()
        if self.folder in path.parents:
            if path.name == 'proof.json':
                self.proofs_seen.add(str(path.relative_to(self.folder)))
            if path.parent.name.startswith('cost') and path.parent.name[4:].isdigit() and path.name in ['input.json', 'result.json']:
                self.costs_seen.add(str(path.relative_to(self.folder)))
        return json.loads(data)

    def sha(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = hashlib.sha256(self.data(path)).hexdigest()
        return self.hashes[key]

    def logical_cost_groups(self, prefix=None):
        prefix = Path(prefix).resolve() if prefix is not None else self.folder
        if self.archive:
            return self.archive.logical_cost_groups(prefix)
        return {int(p.parent.name[4:]) for p in prefix.glob('cost*/result.json')
                if p.parent.name[4:].isdigit()}

    def ledger(self):
        raw = self.folder/'query-ledger.jsonl'
        packed = self.folder/'query-ledger.jsonl.gz'
        self.check('single_unambiguous_query_ledger', raw.exists() != packed.exists())
        if raw.exists():
            return raw.read_bytes()
        data = gzip.decompress(packed.read_bytes())
        receipt = self.read(self.folder/'ledger-export.json')
        self.check('lossless_query_ledger_export', receipt['raw_sha256'] == hashlib.sha256(data).hexdigest() and
                   receipt['raw_bytes'] == len(data) and receipt['gzip_sha256'] == self.sha(packed))
        return data

    def finish(self):
        if self.archive:
            manifest = self.read(self.folder/'proof-export.json')
            self.check('every_exported_proof_replayed', self.proofs_seen ==
                       {row['logical_path'] for row in manifest['proofs']})
            self.check('every_counterfactual_document_replayed', self.costs_seen == set(self.archive.entries))
            result = self.archive.validate_export()
            self.archive.close()
            return result
        proofs = {str(p.relative_to(self.folder)) for p in self.folder.rglob('proof.json')}
        self.check('every_raw_proof_replayed', self.proofs_seen == proofs)
        costs = {str(p.relative_to(self.folder)) for p in self.folder.rglob('*.json')
                 if p.parent.name.startswith('cost') and p.parent.name[4:].isdigit() and
                 p.name in ['input.json', 'result.json']}
        self.check('every_raw_counterfactual_document_replayed', self.costs_seen == costs)
        return dict(status='passed_raw_scientific_artifact_audit', proof_files=len(proofs),
                    counterfactual_logical_files=len(costs), all_proofs_and_cost_documents_replayed=True)
