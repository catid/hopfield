"""H2bh frozen integrated fast-prefix and archived continuation parity gates."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import confirmation96_batch as Q
from confirmation96_study import PP,DEST,protocol,load_models
from factorial_transfer import hidden

SOURCE=S.OUT/'loop80_transfer';CONT=S.OUT/'loop80_continuation'


def compare_raw(raw,path,source_item=None):
    with np.load(path) as z:states=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());prestop=z['pre_terminal_stopped'];precap=z['pre_terminal_block_cap'];indices=z['indices'] if 'indices' in z else z['valid_prefix_indices']
    assert raw is not None and raw['trace']==payload['trace'] and np.array_equal(raw['states_before'],states);assert np.array_equal(raw['pre_terminal_stopped'],prestop) and np.array_equal(raw['pre_terminal_block_cap'],precap);assert raw['terminal_check_records']==payload.get('terminal_checks',payload.get('final_terminal_checks'));return indices,len(payload['trace']),sum(len(r['candidates']) for r in payload['trace'])


def main(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();source_ds=json.loads((SOURCE/'datasets.json').read_text())['datasets'][0];assert S.sha(S.ROOT/source_ds['path'])==source_ds['sha256'];inp=json.loads((CONT/'inputs.json').read_text());assert S.sha(CONT/'protocol.json')==p['parent_protocol_sha256']['research/20261114/work/continuation-protocol.json'];rows=[];controws=[]
    with np.load(S.ROOT/source_ds['path']) as z:tokens=z['tokens']
    def watch():assert time.perf_counter()-started<120,'Frozen120s source gate workerbudget'
    with torch.no_grad():
        for seed in ([0,2] if gpu==0 else [1]):
            models,doubles=load_models(p,seed);run=SOURCE/'runs'/f'shard00-s{seed}';receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
            with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','stopped','finite','couplings','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path']}
            trace_count=candidates=0
            for lo in range(0,256,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');a,hs,good=hidden(models[0],t);assert good.all() and np.array_equal(a.cpu().numpy(),old['couplings'][lo:lo+32]);r=Q.solve(models,doubles,t,a,rotation=(lo//32+seed)%8,watchdog=watch)
                for v in range(6):
                    for name in ['prediction','iterations','stopped','finite']:assert np.array_equal(r['outputs'][v][name].cpu().numpy(),old['predictions' if name=='prediction' else name][v,lo:lo+32]),(seed,lo,v,name)
                for v in [2,3,4,5]:
                    item=next(q for q in receipt['trace_archives'] if q['offset']==lo and q['arm']==Q.ARMS[v]);path=run/item['path'];assert S.sha(path)==item['sha256']
                    with np.load(path) as z:ix=z['valid_prefix_indices'];states=z['states_before'];payload=json.loads(z['trace_utf8'].tobytes());initial_term=z['initial_terminal'];prestop=z['pre_terminal_stopped'];precap=z['pre_terminal_block_cap']
                    block=r['blocks'][v];raw=block['raw'];assert np.array_equal(block['valid_prefix_indices'],ix) and np.array_equal(block['initial_terminal'],initial_term);assert payload['initial_terminal_checks']==block['initial_terminal_records'];assert payload['trace']==(raw['trace'] if raw else []);assert np.array_equal(states,raw['states_before'] if raw else np.zeros((0,0,180,180),bool));assert payload['final_terminal_checks']==(raw.get('terminal_check_records',[]) if raw else [])
                    assert np.array_equal(prestop,raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool)) and np.array_equal(precap,raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool))
                    for key,oldkey in [('block_moves','block_moves'),('post_block_sweeps','post_block_sweeps'),('native_cap','block_native_cap'),('block_cap','block_cap'),('initial_simple_path','initial_simple_path')]:assert np.array_equal(block[key],old[oldkey][v-2,lo:lo+32])
                    trace_count+=len(payload['trace']);candidates+=sum(len(q['candidates']) for q in payload['trace'])
            rows.append({'seed':seed,'inputs':256,'all_six_archived_prefix_boards_flags_choices_traces_exact':True,'trace_rounds':trace_count,'candidate_records':candidates,'source_outputs_sha256':receipt['outputs_sha256']})
            for job in inp['jobs']:
                if job['seed']!=seed:continue
                path=S.ROOT/job['path'];assert S.sha(path)==job['sha256'];run=CONT/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
                with np.load(path) as z:d={k:z[k] for k in z.files}
                with np.load(run/'outputs.npz') as z:expected={k:z[k] for k in z.files}
                t=torch.tensor(d['tokens'].astype(np.int64),device='cuda');a=torch.tensor(d['couplings'],device='cuda');r=Q.solve(models,doubles,t,a,watchdog=watch);assert np.array_equal(r['outputs'][5]['prediction'].cpu().numpy(),d['board']) and np.array_equal(r['outputs'][5]['iterations'].cpu().numpy(),d['iterations']);assert r['blocks'][5]['block_cap'].all()
                cc=rr=0
                for v,arm,name in [(6,0,'extra16_limit256'),(7,1,'extra32_limit256')]:
                    block=r['blocks'][v];raw=block['raw'];out=r['outputs'][v]
                    for key in ['prediction','iterations','stopped','finite']:assert np.array_equal(out[key].cpu().numpy(),expected[key][arm])
                    for key in ['native_cap','block_cap']:assert np.array_equal(block[key],expected[key][arm])
                    moves=sum(r['blocks'][vv]['block_moves'] for vv in range(6,v+1));sweeps=sum(r['blocks'][vv]['post_block_sweeps'] for vv in range(6,v+1));assert np.array_equal(moves,expected['extra_moves'][arm]) and np.array_equal(sweeps,expected['extra_sweeps'][arm]);item=next((item for item in receipt['trace_archives'] if item['stage']==name),None)
                    if item:
                        assert S.sha(run/item['path'])==item['sha256'];ix,nr,nc=compare_raw(raw,run/item['path']);assert np.array_equal(ix,block['valid_prefix_indices']);cc+=nc;rr+=nr
                    else:assert raw is None and not len(block['valid_prefix_indices'])
                controws.append({'job':job['id'],'seed':seed,'caps':job['n'],'all48_starts_and64_96_continuations_exact':True,'trace_rounds':rr,'candidate_records':cc,'source_outputs_sha256':receipt['outputs_sha256']});watch()
    S.save(DEST/f'source-gate-gpu{gpu}.json',{'status':'passed','gpu':gpu,'inventory':inventory,'protocol_sha256':S.sha(PP),'source_prefix_rows':rows,'source_continuation_rows':controws,'seconds':time.perf_counter()-started,'scope':'Integrated fast-veto/full lexical prefix exact against archived H2be all768 shard00modelcases; all148H2bgcap starts/64/96 choices/states/counters/flags exact, split byseed. Source verification only.'})


def collect():
    p=protocol();rows=[json.loads((DEST/f'source-gate-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' and r['protocol_sha256']==S.sha(PP) and r['seconds']<120 for r in rows);assert sum(x['inputs'] for r in rows for x in r['source_prefix_rows'])==768;assert sum(x['caps'] for r in rows for x in r['source_continuation_rows'])==148;S.save(DEST/'numerical-gate.json',{'status':'passed','protocol_sha256':S.sha(PP),'rows':rows,'receipt_sha256':{f'source-gate-gpu{g}.json':S.sha(DEST/f'source-gate-gpu{g}.json') for g in [0,1]}})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[0,1]);a=ap.parse_args();main(a.gpu) if a.gpu is not None else collect()
