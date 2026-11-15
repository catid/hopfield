"""Public-only H2av five-arm states without inference/framework imports."""
import argparse
import json
from pathlib import Path
from verify_factorial_states import verify as base_verify,sha


def verify(directory):
    r=base_verify(directory);r['scope']='H2av deterministicpublicsubset,240states/48labels. Complete61440states/allcandidatecostsandcycletransitions independentlyauditedseparately. NumPyonly; noTorch/decoder.';r['verifier_sha256']=sha(Path(__file__));return r


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);print(json.dumps(verify(a.parse_args().directory),indent=2))
