"""Independent graph/certificate verification of all published96 failures."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_confirmation96_states import verify as verify_states,sha
from verify_cycle_cut_geometry import cycle_rank


def verify(directory):
    r=verify_states(directory);m=json.loads((directory/'manifest.json').read_text());rows=[]
    for fixture,checked in zip(m['fixtures'],r['rows']):
        with np.load(directory/fixture['file']) as z:
            success=np.array(checked['graph_valid_shortest_by_arm'])&z['finite'][:,0]&z['stopped'][:,0]&~z['block_cap'][:,0]&~z['native_cap'][:,0]&(z['minimum_flip_margin64'][:,0]>=-1e-6)&z['hidden_good'][0];assert success.astype(int).tolist()==fixture['expected_success_by_arm'] and not success[3];assert [cycle_rank(b[0]) for b in z['predictions']]==fixture['cycle_ranks_by_arm'];assert z['block_cap'][3,0] and not z['native_cap'][3,0]
            rows.append({'input_index':fixture['input_index'],'model_seed':fixture['model_seed'],'success_by_arm':success.astype(int).tolist(),'total96_block_cap':True,'cycle_ranks_by_arm':fixture['cycle_ranks_by_arm'],'total_cycle_moves_by_arm':fixture['total_cycle_moves_by_arm']})
    return {**r,'rows':rows,'scope':'Explicit complete96-failure supplement; all published4-arm boards, originalenergies/margins,graphshortestness and native/blockflags checked. No new evidence or unbiasedsubsetclaim.','verifier_sha256':sha(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);print(json.dumps(verify(p.parse_args().directory),indent=2))
