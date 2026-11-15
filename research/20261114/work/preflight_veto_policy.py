"""H2bf optimized full-policy replay versus frozen H2be source archives."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import veto_policy_batch as Q
from loop80_transfer import protocol,load_models,DEST as SOURCE
from factorial_transfer import hidden

DEST=S.OUT/'veto_speed_preflight'


def main(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();started=time.perf_counter();p=protocol();frozen=S.HERE/'veto-speed-protocol.json';fp=json.loads(frozen.read_text());assert all(S.sha(S.ROOT/k)==h for k,h in fp['source_sha256'].items());assert json.loads((DEST/'synthetic-gate.json').read_text())['status']=='passed';ds=json.loads((SOURCE/'datasets.json').read_text())['datasets'][0];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
    rows=[]
    def watch():assert time.perf_counter()-started<120,'Fixed120s source-policy gate worker budget'
    with torch.no_grad():
        for seed in ([0,2] if gpu==0 else [1]):
            models,doubles=load_models(p,seed);run=SOURCE/'runs'/f'shard00-s{seed}';receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
            with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','stopped','finite','couplings','minimum_flip_margin64','energy64','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path']}
            trace_count=candidates=0
            for lo in range(0,256,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');a,hs,good=hidden(models[0],t);assert good.all() and np.array_equal(a.cpu().numpy(),old['couplings'][lo:lo+32]);r=Q.solve(models,doubles,t,a,fast=True,watchdog=watch)
                for v in range(1,6):
                    for key in ['prediction','iterations','stopped','finite']:assert np.array_equal(r['outputs'][v][key].cpu().numpy(),old['predictions' if key=='prediction' else key][v,lo:lo+32]),(seed,lo,v,key)
                for v in range(2,6):
                    item=next(q for q in receipt['trace_archives'] if q['offset']==lo and q['arm']==Q.ARMS[v]);path=run/item['path'];assert S.sha(path)==item['sha256']
                    with np.load(path) as z:ix=z['valid_prefix_indices'];states=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());initial_term=z['initial_terminal'];pre_stop=z['pre_terminal_stopped'];pre_cap=z['pre_terminal_block_cap']
                    b=r['blocks'][v];raw=b['raw'];assert np.array_equal(ix,b['valid_prefix_indices']) and np.array_equal(initial_term,b['initial_terminal']);assert payload['initial_terminal_checks']==b['initial_terminal_records'];assert payload['trace']==(raw['trace'] if raw else []);assert np.array_equal(states,raw['states_before'] if raw else np.zeros((0,0,180,180),bool));assert payload['final_terminal_checks']==(raw.get('terminal_check_records',[]) if raw else [])
                    assert np.array_equal(pre_stop,raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool));assert np.array_equal(pre_cap,raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool))
                    for key,oldkey in [('block_moves','block_moves'),('post_block_sweeps','post_block_sweeps'),('native_cap','block_native_cap'),('block_cap','block_cap'),('initial_simple_path','initial_simple_path')]:assert np.array_equal(b[key],old[oldkey][v-2,lo:lo+32]),(seed,lo,v,key)
                    trace_count+=len(payload['trace']);candidates+=sum(len(q['candidates']) for q in payload['trace'])
                final=r['outputs'][5]['prediction'];margin=doubles[1].flip_delta(t,final,a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();energy=doubles[1].energy(t,final,a.double()).cpu().numpy();assert np.allclose(margin,old['minimum_flip_margin64'][5,lo:lo+32],rtol=0,atol=1e-9) and np.allclose(energy,old['energy64'][5,lo:lo+32],rtol=0,atol=1e-6);watch()
            rows.append({'seed':seed,'inputs':256,'all_prefix_boards_iterations_flags_exact':True,'all_choices_traces_terminal_checks_exact':True,'trace_rounds':trace_count,'candidate_cost_records':candidates,'source_outputs_sha256':receipt['outputs_sha256']})
    result={'status':'passed','gpu':gpu,'inventory':inventory,'rows':rows,'frozen_protocol_sha256':S.sha(frozen),'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'dataset':ds,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['bridge_veto_geometry.py','bridge_veto_moves.py','veto_policy_batch.py','preflight_veto_policy.py','bridge_cycle_geometry.py','bridge_lexical_moves.py','bridge_terminal_moves.py']},'scope':'Complete optimized unified48 pipeline on fixed shard00/all3 models split across two workers; compare all prefixes/choices/traces/flags against archived H2be. Source verification only; no comparative speed endpoint.'};watch();S.save(DEST/f'policy-gate-gpu{gpu}.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);main(p.parse_args().gpu)
