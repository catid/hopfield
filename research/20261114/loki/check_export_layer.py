"""Validate logical-byte parity, ambiguity and tamper rejection for proof exports."""
import json
from pathlib import Path
import tempfile
import time
from export_layer import export_tree,validate_export,logical_bytes,logical_json,logical_sha256


def main():
    started=time.perf_counter();fixtures=[b'{"states":{},"value":12345678901234567890}\n',b'{\n  "unicode": "\\u2264",\n  "text": "literal \\n"\n}\n']
    with tempfile.TemporaryDirectory(prefix='proof-export-gate-') as temporary:
        base=Path(temporary);source=base/'scientific';target=base/'public';source.mkdir()
        for i,data in enumerate(fixtures):folder=source/f'group{i}';folder.mkdir();(folder/'proof.json').write_bytes(data)
        (source/'summary.txt').write_bytes(b'Unmodified ordinary artifact\n');manifest=export_tree(source,target);audit=validate_export(target)
        for i,data in enumerate(fixtures):
            path=target/f'group{i}/proof.json';assert logical_bytes(path)==data and logical_json(path)==json.loads(data) and logical_sha256(path)==logical_sha256(source/f'group{i}/proof.json') and not path.exists()
        assert (target/'summary.txt').read_bytes()==(source/'summary.txt').read_bytes();first_bytes=[(target/row['compressed_path']).read_bytes() for row in manifest['proofs']];second=export_tree(source,target);assert second==manifest and first_bytes==[(target/row['compressed_path']).read_bytes() for row in second['proofs']]
        plain=target/'group0/proof.json';plain.write_bytes(fixtures[0]);rejected={}
        try:logical_bytes(plain)
        except ValueError:rejected['ambiguous_plain_and_compressed']=True
        else:raise AssertionError('Ambiguous proof representation accepted')
        plain.unlink();manifest_path=target/'proof-export.json';changed=json.loads(manifest_path.read_text());changed['proofs'][0]['raw_sha256']='0'*64;manifest_path.write_text(json.dumps(changed))
        try:validate_export(target)
        except AssertionError:rejected['wrong_raw_identity']=True
        else:raise AssertionError('Wrong logical proof identity accepted')
        export_tree(source,target);packed=target/'group0/proof.json.gz';packed.write_bytes(packed.read_bytes()[:-3])
        try:validate_export(target)
        except (AssertionError,EOFError):rejected['truncated_compressed_proof']=True
        else:raise AssertionError('Truncated proof accepted')
    print(json.dumps(dict(status='passed',fixtures=2,byte_exact_roundtrips=True,deterministic_repeat_export=True,ordinary_files_unchanged=True,rejected=rejected,audit=audit,elapsed_seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
