"""Exact identities, determinism, random access and fail-closed archive checks."""
import json
from pathlib import Path
import tempfile
import archive_export_layer as adapter


def must_reject(callback):
    try:callback()
    except (AssertionError,ValueError,KeyError,adapter.zipfile.BadZipFile):return
    raise AssertionError('Invalid archive was accepted')


def main():
    with tempfile.TemporaryDirectory(prefix='h3bu-archive-export-') as temporary:
        root=Path(temporary);source=root/'source';source.mkdir();files={'proof.json':b'{"states": [1, 2, 3]}\n','group0/proof.json':b'{"depth": 8, "state": "x"}\n','group0/result.json':b'{"receipt": true}\n','cost0/input.json':b'{"candidates": [0,1]}\n','cost0/result.json':b'{"calls": {"0": 1, "1": 2}}\n','cost-gate/fixture/cost1/input.json':b'{"toy": true}\n','cost-gate/fixture/cost1/result.json':b'{"result": 3}\n','unchanged.bin':bytes(range(256))}
        for rel,data in files.items():path=source/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        receipts=[]
        for name in ['one','two']:
            target=root/name;receipts.append(adapter.export_tree(source,target));reader=adapter.PublicArchive(target);result=reader.validate_export()
            for rel,data in files.items():assert reader.logical_bytes(target/rel)==data
            assert reader.logical_cost_groups()=={0} and reader.logical_cost_groups(target/'cost-gate/fixture')=={1};reader.close()
        assert receipts[0]==receipts[1]
        for path in (root/'one').rglob('*'):
            if path.is_file():assert path.read_bytes()==(root/'two'/path.relative_to(root/'one')).read_bytes()
        reader=adapter.PublicArchive(root/'one');plain=root/'one/cost0/input.json';plain.parent.mkdir();plain.write_bytes(files['cost0/input.json']);must_reject(lambda:reader.logical_bytes(plain));plain.unlink();reader.close()
        manifest=root/'one/counterfactual-export.json';original=manifest.read_text();bad=json.loads(original);bad['entries'][0]['raw_sha256']='0'*64;manifest.write_text(json.dumps(bad));reader=adapter.PublicArchive(root/'one');must_reject(reader.validate_export);reader.close();manifest.write_text(original)
        archive=root/'one/counterfactual-json.zip';archive.write_bytes(archive.read_bytes()[:20]);must_reject(lambda:adapter.PublicArchive(root/'one'))
        print(dict(status='passed',all8_logical_bytes_exact=True,both_exports_byte_identical=True,nested_random_access_verified=True,ambiguous_wrong_hash_and_truncated_exports_rejected=True))


if __name__=='__main__':main()
