"""Parallel deterministic gzip export; scientific source files are read-only."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def proof_export(pair):
    path, target, source, destination = pair
    raw = path.read_bytes()
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    assert gzip.decompress(packed) == raw
    target.parent.mkdir(parents=True, exist_ok=True)
    compressed = target.with_name(target.name+'.gz')
    assert not target.exists() and not compressed.exists()
    compressed.write_bytes(packed)
    return dict(logical_path=str(path.relative_to(source)), raw_sha256=digest(raw), raw_bytes=len(raw),
                compressed_path=str(compressed.relative_to(destination)), compressed_sha256=digest(packed),
                compressed_bytes=len(packed))


def export_tree(source, destination, adapter, workers=4):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source != destination and source not in destination.parents and not destination.exists()
    destination.mkdir(parents=True)
    started = time.perf_counter()
    metadata, pending = [], []
    archive_path = destination/'counterfactual-json.zip'
    with ThreadPoolExecutor(max_workers=workers) as pool, zipfile.ZipFile(
            archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or (path.parent == source and path.name in ['query-ledger.jsonl', 'predictions-bits.npy']):
                continue
            assert not path.is_symlink()
            relative = path.relative_to(source)
            target = destination/relative
            if adapter.is_counterfactual_json(relative):
                assert not target.exists()
                data = path.read_bytes()
                info = zipfile.ZipInfo(str(relative), (1980, 1, 1, 0, 0, 0))
                info.create_system, info.external_attr, info.compress_type = 3, 0o100644 << 16, zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                metadata.append(dict(logical_path=str(relative), raw_sha256=digest(data), raw_bytes=len(data)))
            elif path.name == 'proof.json':
                pending.append(pool.submit(proof_export, (path, target, source, destination)))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        proofs = [task.result() for task in pending]
    with zipfile.ZipFile(archive_path) as archive:
        for row in metadata:
            data = archive.read(row['logical_path'])
            assert len(data) == row['raw_bytes'] and digest(data) == row['raw_sha256']
    pm = dict(format='lossless-proof-gzip-v1', gzip_level=6, gzip_mtime=0, proof_files=len(proofs),
              raw_proof_bytes=sum(r['raw_bytes'] for r in proofs),
              compressed_proof_bytes=sum(r['compressed_bytes'] for r in proofs), all_export_roundtrips_exact=True, proofs=proofs)
    cm = dict(format='lossless-counterfactual-zip-v1', archive='counterfactual-json.zip', compression='deflate level6',
              entry_date_time=[1980, 1, 1, 0, 0, 0], archive_sha256=digest(archive_path.read_bytes()),
              archive_bytes=archive_path.stat().st_size, logical_files=len(metadata), raw_bytes=sum(r['raw_bytes'] for r in metadata),
              all_export_roundtrips_exact=True, entries=metadata)
    for filename, value in [('proof-export.json', pm), ('counterfactual-export.json', cm)]:
        (destination/filename).write_text(json.dumps(value, indent=2)+'\n')
    return dict(proofs=pm, counterfactual=cm), dict(workers=workers, elapsed_seconds=time.perf_counter()-started,
        compression='Identical gzip level6 mtime0 and ordered deflate6 ZIP bytes; independent files compressed in four threads')
