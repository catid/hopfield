"""Verify public transfer statistics from the published sibling CSV/summary layout.

Example: artifacts/statistics-venv/bin/python research/20261114/work/verify_public_transfer_statistics.py results/research/work cap-transfer
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile


def main(public,campaign):
    if campaign=='cap-transfer':from verify_cap_transfer_statistics import main as check
    elif campaign=='density-transfer':from verify_density_transfer_statistics import main as check
    else:raise ValueError(campaign)
    with tempfile.TemporaryDirectory(prefix='hopfield-public-statistics-') as raw:
        temp=Path(raw)
        for source,target in [(public/campaign/'protocol.json','protocol.json'),(public/f'{campaign}-summary.json','audit.json'),(public/f'{campaign}-per-input.csv','per-input.csv')]:shutil.copyfile(source,temp/target)
        with contextlib.redirect_stdout(io.StringIO()):check(temp)
        result=json.loads((temp/'statistics-verification.json').read_text());result['public_layout']={'campaign':campaign,'protocol':str(public/campaign/'protocol.json'),'audit':str(public/f'{campaign}-summary.json'),'paired_rows':str(public/f'{campaign}-per-input.csv')};print(json.dumps(result,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('public',type=Path);a.add_argument('campaign',choices=['cap-transfer','density-transfer']);v=a.parse_args();main(v.public,v.campaign)
