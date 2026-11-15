"""Public-only H2ap retained-state verification without inference imports."""
import argparse
import json
from pathlib import Path
from verify_factorial_states import verify as base_verify,sha


def verify(directory):
    r=base_verify(directory);r['scope']='H2apdeterministic publicsubset,240states/48labels. Full12288interventionfinalstates,63searchinputs andallcandidategeometry auditedseparately. NumPyonly;noTorch/decoder.';r['verifier_sha256']=sha(Path(__file__));return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);print(json.dumps(verify(p.parse_args().directory),indent=2))
