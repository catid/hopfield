"""Public-only original-energy/margin/shortest-path H2ba state verification."""
import argparse
import json
from pathlib import Path
from verify_factorial_states import verify as base_verify,sha


def verify(directory):
    r=base_verify(directory);r['scope']='H2ba first-input/all-source-cap fixtures. Inherited states, certificate-only boards and new16round boards compared with original objective independently. Full24476newstates/flags/geometry/terminalchecks audited separately. NumPy only; no Torch or decoder imports.';r['scope']=r['scope'].replace('24476','24576');r['verifier_sha256']=sha(Path(__file__));return r


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);print(json.dumps(verify(a.parse_args().directory),indent=2))
