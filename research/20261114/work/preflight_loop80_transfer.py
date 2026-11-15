"""H2be six-arm source parity before fresh loop80 draws."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import fast_unified_cycle_batch as Q
from unified_cycle import protocol,DEST as SOURCE
from loop80_transfer import load_models
from factorial_transfer import hidden

DEST=S.OUT/'loop80_transfer_preflight'


def main():
    gpu=0;DEST.mkdir(exist_ok=True)
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();started=time.perf_counter();p=protocol();old=(S.HERE/'unified_cycle_batch.py').read_text();new=(S.HERE/'fast_unified_cycle_batch.py').read_text();assert old[old.index('def solve('):]==new[new.index('def solve('):];assert json.loads((S.OUT/'geometry_speed/audit.json').read_text())['status']=='passed';ds=json.loads((SOURCE/'datasets.json').read_text())['datasets'][0];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
    rows=[]
    def watch():assert time.perf_counter()-started<120,'Fixed120s source-policy gate worker budget'
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed);run=SOURCE/'runs'/f'shard00-s{seed}';receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
            with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','stopped','finite','couplings','minimum_flip_margin64','energy64','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path']}
            trace_count=candidates=0
            for lo in range(0,32,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');a,hs,good=hidden(models[0],t);assert good.all() and np.array_equal(a.cpu().numpy(),old['couplings'][lo:lo+32]);r=Q.solve(models,doubles,t,a,watchdog=watch)
                for v in range(6):
                    for key in ['prediction','iterations','stopped','finite']:assert np.array_equal(r['outputs'][v][key].cpu().numpy(),old['predictions' if key=='prediction' else key][v,lo:lo+32]),(seed,lo,v,key)
                for v in range(2,6):
                    item=next(q for q in receipt['trace_archives'] if q['offset']==lo and q['arm']==Q.ARMS[v]);path=run/item['path'];assert S.sha(path)==item['sha256']
                    with np.load(path) as z:ix=z['valid_prefix_indices'];states=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());initial_term=z['initial_terminal'];pre_stop=z['pre_terminal_stopped'];pre_cap=z['pre_terminal_block_cap']
                    b=r['blocks'][v];raw=b['raw'];assert np.array_equal(ix,b['valid_prefix_indices']) and np.array_equal(initial_term,b['initial_terminal']);assert payload['initial_terminal_checks']==b['initial_terminal_records'];assert payload['trace']==(raw['trace'] if raw else []);assert np.array_equal(states,raw['states_before'] if raw else np.zeros((0,0,180,180),bool));assert payload['final_terminal_checks']==(raw.get('terminal_check_records',[]) if raw else [])
                    assert np.array_equal(pre_stop,raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool));assert np.array_equal(pre_cap,raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool))
                    for key,oldkey in [('block_moves','block_moves'),('post_block_sweeps','post_block_sweeps'),('native_cap','block_native_cap'),('block_cap','block_cap'),('initial_simple_path','initial_simple_path')]:assert np.array_equal(b[key],old[oldkey][v-2,lo:lo+32]),(seed,lo,v,key)
                    trace_count+=len(payload['trace']);candidates+=sum(len(q['candidates']) for q in payload['trace'])
                final=r['outputs'][5]['prediction'];margin=doubles[1].flip_delta(t,final,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();energy=doubles[1].energy(t,final,a.double()).cpu().numpy();assert np.allclose(margin,old['minimum_flip_margin64'][5,lo:lo+32],rtol=0,atol=1e-9) and np.allclose(energy,old['energy64'][5,lo:lo+32],rtol=0,atol=1e-6);watch()
            rows.append({'seed':seed,'inputs':32,'all_prefix_boards_iterations_flags_exact':True,'all_choices_traces_terminal_checks_exact':True,'trace_rounds':trace_count,'candidate_cost_records':candidates,'source_outputs_sha256':receipt['outputs_sha256']})
    result={'status':'passed','gpu':gpu,'inventory':inventory,'rows':rows,'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'dataset':ds,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['loop80_transfer.py','fast_unified_cycle_batch.py','preflight_loop80_transfer.py','bridge_cycle_geometry.py','bridge_lexical_moves.py','bridge_terminal_moves.py']},'scope':'All six arms on first32 H2bc shard00 inputs/all3models; exact source boards/iterations/flags/candidate costs/choices/traces/terminal checks. H2bd full768 source pipeline and35382 geometry states also hash-bound. Zero fresh loop80 inputs.'};watch();S.save(DEST/'numerical-gate.json',result)


if __name__=='__main__':
    main()
