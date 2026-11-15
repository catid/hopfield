"""Lossless proof gzip plus deterministic indexed counterfactual JSON archive.

Separate from the already-published proof-only export adapter. Scientific inputs
and logical artifact hashes remain unchanged; only public storage is adapted.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

_ADAPTER=Path(__file__).with_name('export_layer.py')
_spec=importlib.util.spec_from_file_location('proof_only_export_adapter',_ADAPTER)
proof_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(proof_adapter)


def digest(data):return hashlib.sha256(data).hexdigest()


def is_counterfactual_json(relative):
    relative=Path(relative);name=relative.parent.name
    return relative.name in ['input.json','result.json'] and name.startswith('cost') and name[4:].isdigit()


def export_tree(source,destination,exclude_names=('query-ledger.jsonl','predictions-bits.npy')):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents:raise ValueError('Export destination must be separate from scientific artifacts')
    destination.mkdir(parents=True,exist_ok=True);proofs=[];metadata=[];archive_path=destination/'counterfactual-json.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.name in exclude_names:continue
            if path.is_symlink():raise ValueError(f'Scientific artifact must be a regular file: {path}')
            relative=path.relative_to(source);target=destination/relative
            if is_counterfactual_json(relative):
                if target.exists():raise ValueError(f'Refusing ambiguous plain/archive metadata: {target}')
                data=path.read_bytes();info=zipfile.ZipInfo(str(relative),(1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6);metadata.append(dict(logical_path=str(relative),raw_sha256=digest(data),raw_bytes=len(data)))
            elif path.name=='proof.json':
                if target.exists():raise ValueError(f'Refusing to replace plain proof export: {target}')
                target.parent.mkdir(parents=True,exist_ok=True);raw=path.read_bytes();packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw;compressed=target.with_name(target.name+'.gz');compressed.write_bytes(packed);proofs.append(dict(logical_path=str(relative),raw_sha256=digest(raw),raw_bytes=len(raw),compressed_path=str(compressed.relative_to(destination)),compressed_sha256=digest(packed),compressed_bytes=len(packed)))
            else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    with zipfile.ZipFile(archive_path) as archive:
        for row in metadata:
            data=archive.read(row['logical_path']);assert len(data)==row['raw_bytes'] and digest(data)==row['raw_sha256']
    proof_manifest=dict(format='lossless-proof-gzip-v1',gzip_level=6,gzip_mtime=0,proof_files=len(proofs),raw_proof_bytes=sum(r['raw_bytes'] for r in proofs),compressed_proof_bytes=sum(r['compressed_bytes'] for r in proofs),all_export_roundtrips_exact=True,proofs=proofs);metadata_manifest=dict(format='lossless-counterfactual-zip-v1',archive='counterfactual-json.zip',compression='deflate level6',entry_date_time=[1980,1,1,0,0,0],archive_sha256=digest(archive_path.read_bytes()),archive_bytes=archive_path.stat().st_size,logical_files=len(metadata),raw_bytes=sum(r['raw_bytes'] for r in metadata),all_export_roundtrips_exact=True,entries=metadata)
    for filename,value in [('proof-export.json',proof_manifest),('counterfactual-export.json',metadata_manifest)]:(destination/filename).write_text(json.dumps(value,indent=2)+'\n')
    return dict(proofs=proof_manifest,counterfactual=metadata_manifest)


class PublicArchive:
    def __init__(self,root):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/'counterfactual-export.json').read_text());self.entries={row['logical_path']:row for row in self.manifest['entries']};self.archive=zipfile.ZipFile(self.root/'counterfactual-json.zip')
        if len(self.entries)!=len(self.manifest['entries']):raise ValueError('Duplicate logical counterfactual paths')

    def close(self):self.archive.close()

    def logical_bytes(self,path):
        path=Path(path).resolve()
        try:relative=str(path.relative_to(self.root))
        except ValueError:return proof_adapter.logical_bytes(path)
        if relative in self.entries:
            if path.exists():raise ValueError(f'Ambiguous plain/archive metadata: {path}')
            data=self.archive.read(relative);row=self.entries[relative]
            if len(data)!=row['raw_bytes'] or digest(data)!=row['raw_sha256']:raise ValueError(f'Invalid archived logical identity: {relative}')
            return data
        if is_counterfactual_json(relative):raise ValueError(f'Unindexed counterfactual artifact: {relative}')
        return proof_adapter.logical_bytes(path)

    def logical_json(self,path):return json.loads(self.logical_bytes(path))
    def logical_sha256(self,path):return digest(self.logical_bytes(path))

    def logical_cost_groups(self,prefix=None):
        prefix=Path(prefix).resolve() if prefix is not None else self.root
        return {int((self.root/relative).parent.name[4:]) for relative in self.entries if (self.root/relative).parent.parent==prefix}

    def validate_export(self):
        m=self.manifest;assert m['format']=='lossless-counterfactual-zip-v1' and m['archive']=='counterfactual-json.zip' and m['compression']=='deflate level6' and m['entry_date_time']==[1980,1,1,0,0,0];path=self.root/'counterfactual-json.zip';assert path.stat().st_size==m['archive_bytes'] and digest(path.read_bytes())==m['archive_sha256'];names=self.archive.namelist();assert names==sorted(self.entries) and len(names)==len(set(names));raw=0
        for row in m['entries']:
            relative=Path(row['logical_path']);assert not relative.is_absolute() and '..' not in relative.parts and is_counterfactual_json(relative);data=self.logical_bytes(self.root/relative);info=self.archive.getinfo(str(relative));assert info.date_time==(1980,1,1,0,0,0) and info.compress_type==zipfile.ZIP_DEFLATED and info.create_system==3 and info.external_attr==0o100644<<16;raw+=len(data)
        assert m['logical_files']==len(names) and m['raw_bytes']==raw and m['all_export_roundtrips_exact'];assert not any(is_counterfactual_json(path.relative_to(self.root)) for path in self.root.rglob('*.json') if path.is_file());proofs=proof_adapter.validate_export(self.root)
        return dict(status='passed',counterfactual_logical_files=len(names),counterfactual_raw_bytes=raw,counterfactual_archive_bytes=m['archive_bytes'],proofs=proofs,all_logical_and_physical_identities_verified=True)
