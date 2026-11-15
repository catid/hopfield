"""Lossless public export, including unchanged partial evidence on a capped run."""
import gzip
import importlib.util
from common import ROOT,SRC,OUT,frozen,save,sha,now
from parallel_export import export_tree


def main():
    frozen();path=SRC.parent/'archive_export_layer.py';spec=importlib.util.spec_from_file_location('h3bz_archive_export_layer',path);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);public=ROOT/'results/research/loki/h3bz';manifest,timing=export_tree(OUT,public,adapter);raw=(OUT/'query-ledger.jsonl').read_bytes();packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw;(public/'query-ledger.jsonl.gz').write_bytes(packed);save(public/'ledger-export.json',dict(raw_sha256=sha(OUT/'query-ledger.jsonl'),raw_bytes=len(raw),gzip_sha256=sha(public/'query-ledger.jsonl.gz'),gzip_bytes=len(packed),compression='gzip level6 mtime0; exact original bytes preserved'));save(public/'export-provenance.json',dict(utc=now(),source_directory=str(OUT),export_source_sha256=sha(__file__),adapter_source_sha256=sha(path),proof_adapter_source_sha256=sha(SRC.parent/'export_layer.py'),raw_scientific_artifacts_unchanged=True,parallel_export_source_sha256=sha(SRC/'parallel_export.py'),export_timing=timing));print({family:{key:value for key,value in item.items() if key not in ['proofs','entries']} for family,item in manifest.items()})


if __name__=='__main__':main()
