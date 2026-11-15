"""Lossless public proof storage with unchanged logical byte identities.

This module is an export/verification adapter, not scientific experiment code.
Only files named proof.json are represented as deterministic proof.json.gz.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(data):return hashlib.sha256(data).hexdigest()


def logical_bytes(path):
    path=Path(path)
    if path.name=='proof.json':
        compressed=path.with_name(path.name+'.gz')
        if path.exists() and compressed.exists():raise ValueError(f'Ambiguous plain/compressed proof: {path}')
        if not path.exists() and compressed.is_file():return gzip.decompress(compressed.read_bytes())
    return path.read_bytes()


def logical_sha256(path):return digest(logical_bytes(path))


def logical_json(path):return json.loads(logical_bytes(path))


def export_tree(source,destination,exclude_names=('query-ledger.jsonl','predictions-bits.npy')):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents:raise ValueError('Export destination must be separate from scientific artifacts')
    destination.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name in exclude_names:continue
        if path.is_symlink():raise ValueError(f'Scientific artifact must be a regular file: {path}')
        relative=path.relative_to(source);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if path.name=='proof.json':
            if target.exists():raise ValueError(f'Refusing to replace an existing plain proof export: {target}')
            raw=path.read_bytes();packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw;compressed=target.with_name(target.name+'.gz');compressed.write_bytes(packed)
            rows.append(dict(logical_path=str(relative),raw_sha256=digest(raw),raw_bytes=len(raw),compressed_path=str(compressed.relative_to(destination)),compressed_sha256=digest(packed),compressed_bytes=len(packed)))
        else:shutil.copyfile(path,target)
    manifest=dict(format='lossless-proof-gzip-v1',gzip_level=6,gzip_mtime=0,proof_files=len(rows),raw_proof_bytes=sum(r['raw_bytes'] for r in rows),compressed_proof_bytes=sum(r['compressed_bytes'] for r in rows),all_export_roundtrips_exact=True,proofs=rows)
    (destination/'proof-export.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def validate_export(root):
    root=Path(root);manifest=json.loads((root/'proof-export.json').read_text());assert manifest['format']=='lossless-proof-gzip-v1' and manifest['gzip_level']==6 and manifest['gzip_mtime']==0;seen=set();raw_total=0;compressed_total=0
    for row in manifest['proofs']:
        relative=Path(row['logical_path']);compressed_relative=Path(row['compressed_path']);assert not relative.is_absolute() and '..' not in relative.parts and relative.name=='proof.json' and str(compressed_relative)==str(relative)+'.gz';assert row['logical_path'] not in seen;seen.add(row['logical_path']);plain=root/relative;compressed=root/compressed_relative
        assert not plain.exists() and compressed.is_file();packed=compressed.read_bytes();raw=gzip.decompress(packed)
        assert len(packed)==row['compressed_bytes'] and digest(packed)==row['compressed_sha256'] and len(raw)==row['raw_bytes'] and digest(raw)==row['raw_sha256'];raw_total+=len(raw);compressed_total+=len(packed)
    actual={str(path.relative_to(root))[:-3] for path in root.rglob('proof.json.gz') if path.is_file()};assert actual==seen and not any(root.rglob('proof.json')) and manifest['proof_files']==len(seen) and manifest['raw_proof_bytes']==raw_total and manifest['compressed_proof_bytes']==compressed_total and manifest['all_export_roundtrips_exact']
    return dict(status='passed',proof_files=len(seen),raw_proof_bytes=raw_total,compressed_proof_bytes=compressed_total,all_raw_and_compressed_identities_verified=True)
