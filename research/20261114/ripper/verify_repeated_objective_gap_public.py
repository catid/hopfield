"""Portable H1bx manifest and exact objective-gap replay."""
import json
from repeated_objective_gap import ROOT, PUBLIC, sha
from audit_repeated_objective_gap import main as audit


def main():
    path = ROOT/'results/research/ripper/repeated_objective_gap_public_manifest.json'
    manifest = json.loads(path.read_text())
    for rel, digest in manifest['sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    assert sha(PUBLIC/'summary.json') == manifest['summary_sha256']
    audit()
    print(json.dumps(dict(status='passed', manifest_paths=len(manifest['sha256'])+1, raw_artifacts_required=False)))


if __name__ == '__main__':
    main()
