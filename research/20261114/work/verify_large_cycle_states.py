"""Public-only H2aw five-arm original objective and shortest-path verification."""
import argparse
import json
from pathlib import Path
from verify_factorial_states import verify as base_verify,sha


def verify(directory):
    r=base_verify(directory);r['scope']='H2aw public first-input/source-nonsimple subset. Both new policies share the archived lexical256 start. Full 24576 new final states and all candidate geometry/costs audited separately. NumPy only; no Torch or decoder imports.';r['verifier_sha256']=sha(Path(__file__));return r


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);print(json.dumps(verify(a.parse_args().directory),indent=2))
