"""Public-only independent diff: five schema reads, plus declared provenance."""
import ast
import json
from run import ROOT, HERE, sha, save, now


def main():
    old = HERE/'audit_confidence_strict.py'
    new = HERE/'audit_confidence_strict_corrected.py'
    original = old.read_text()
    corrected = new.read_text()
    assert original.count("'weight 16'") == 2
    assert original.count("'weight 16_bytes_sha256'") == 3
    extra = "    amendment = json.loads((HERE/'confidence-strict-audit-correction.json').read_text())\n    assert amendment['corrected_auditor_sha256'] == sha(__file__)\n    assert amendment['original_auditor_sha256'] == sha(HERE/'audit_confidence_strict.py')\n    assert amendment['protocol_sha256'] == sha(PROTOCOL)\n"
    assert corrected.count(extra) == 1
    assert corrected.count(",'audit_correction':amendment") == 1
    numerical = corrected.replace(extra,'').replace(",'audit_correction':amendment",'')
    expected = original.replace("'weight 16'","'weight16'").replace("'weight 16_bytes_sha256'","'weight16_bytes_sha256'")
    assert numerical == expected
    assert ast.dump(ast.parse(numerical),include_attributes=False) == ast.dump(ast.parse(expected),include_attributes=False)
    amendment_path = HERE/'confidence-strict-audit-correction.json'
    amendment = json.loads(amendment_path.read_text())
    assert amendment['original_auditor_sha256'] == sha(old)
    assert amendment['corrected_auditor_sha256'] == sha(new)
    report = {'status':'validated','utc':now(),'verifier_sha256':sha(__file__),
              'original_auditor_sha256':sha(old),'corrected_auditor_sha256':sha(new),
              'addendum_sha256':sha(amendment_path),'literal_schema_fixes':5,
              'other_numeric_or_statistical_changes':False,
              'additional_changes':'Only addendum self-hash/original/protocol provenance checks and result provenance metadata.'}
    save(ROOT/'results/research/ripper/confidence_strict_audit_correction_check.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
