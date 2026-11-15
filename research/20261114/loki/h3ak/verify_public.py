"""Public-only witness replay, model reconstruction and complete capacity census."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,hashlib,itertools,json,subprocess,tempfile,time
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);R=parser.parse_args().root;P=R/'results/research/loki/h3ak';S=R/'research/20261114/loki/h3ak';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();checks=[];start=time.perf_counter()
    def check(name,ok):assert ok,name;checks.append(name)
    def hist(columns):
        h=np.zeros(226**3,np.int64)
        for k in [1,2,3]:
            ix=np.array(list(itertools.combinations(range(15),k)));powers=np.array([226**(2-j)for j in range(k)],np.int64)
            for lo in range(0,len(columns),2048):
                cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];z=((cells[:,ix]+1)*powers).sum(-1);h+=np.bincount(z.reshape(-1),minlength=226**3)
        check('575subsets',h.sum()==575*len(columns));return h
    s=json.loads((P/'summary.json').read_text());p=json.loads((S/'protocol.json').read_text());repair=json.loads((P/'repair-seal.json').read_text());acq=json.loads((P/'source-positive-seal.json').read_text());compiled=json.loads((P/'learned-enumeration-seal.json').read_text());population=np.load(P/'source-population-n13.npy');check('protocol',sha(S/'protocol.json')==s['protocol_sha256']);check('primary supported',s['decision']=='supported'and len(s['primary'])==4 and all(r['capacity_pct']>=95 and r['invalid_zero_F_states']==0 for r in s['primary']));check('lower doses descriptive',all(r['decision']=='descriptive'for r in s['rows']if not r['primary']));check('population identity',sha(P/'source-population-n13.npy')==p['acquisition']['population_sha256']);check('budget',s['aggregate_CPU_seconds']<=120 and s['wall_seconds']<=180 and s['GPU_seconds']==s['target_cohorts_created']==0)
    for rel,h in p['source_sha256'].items():check('source '+rel,sha(R/rel)==h)
    for rel,h in s['public_sha256'].items():check('export '+rel,sha(P/rel)==h)
    allowed=set();models={}
    for bank,cfg in enumerate(p['models']):
        folder=P/f'b{bank}';indices=np.load(folder/'positive-population-indices.npy');boards=np.load(folder/'positive-boards.npy');expected=np.random.default_rng(p['acquisition']['seeds'][bank]).choice(1932,64,replace=False);check(f'bank{bank}draw replay',np.array_equal(indices,expected)and np.array_equal(boards,population[indices]));basepath=R/cfg['model_path'].replace('artifacts/research/20261114/loki','results/research/loki');designpath=R/cfg['source_design_path'].replace('artifacts/research/20261114/loki','results/research/loki');check(f'bank{bank}source identities',sha(basepath)==p['source_data_sha256'][cfg['model_path']]and sha(designpath)==p['source_data_sha256'][cfg['source_design_path']]);base=np.load(basepath);finite={tuple(o)for o in np.load(designpath)['offsets']};allowed|={cfg['model_path'],cfg['source_design_path'],f'artifacts/research/20261114/loki/h3ak/b{bank}/positive-boards.npy',f'artifacts/research/20261114/loki/h3ak/b{bank}/positive-population-indices.npy'}
        for board in boards:
            assert board.sum()==13
            for a,b in itertools.combinations(np.flatnonzero(board),2):dr=abs(int(a)//13-int(b)//13);dc=abs(int(a)%13-int(b)%13);assert dr and dc and dr!=dc and(dr,dc)not in[(1,3),(3,1)]
        check(f'bank{bank}all selected source positives valid',True)
        for dose in [1,8,32,64]:
            model=np.load(folder/f'd{dose}/model.npz');receipt=json.loads((folder/f'd{dose}/repair.json').read_text());seen={}
            for pos,board in enumerate(boards[:dose]):
                for a,b in itertools.combinations(np.flatnonzero(board),2):
                    o=(int(b)//13-int(a)//13,int(b)%13-int(a)%13)
                    if o in finite and o not in seen:seen[o]=(pos,int(a),int(b))
            F=base['F'].copy();J=base['J'].copy();removed=set()
            for a,b in itertools.combinations(range(225),2):
                o=(b//15-a//15,b%15-a%15)
                if F[a,b]and o in seen:F[a,b]=F[b,a]=0;J[a,b]=J[b,a]=0;removed.add(o)
            check(f'bank{bank}dose{dose}exact repair',np.array_equal(F,model['F'])and np.array_equal(J,model['J'])and np.array_equal(model['unit'],base['unit'])and np.array_equal(model['h'],base['h']));check(f'bank{bank}dose{dose}removed classes',receipt['removed_finite_offsets']==[list(o)for o in sorted(removed)]and len(receipt['witnesses'])==len(removed))
            for w in receipt['witnesses']:pos,a,b=seen[tuple(w['offset'])];assert w['positive_position']==pos and w['population_index']==indices[pos]and w['source_cells']==[a,b]
            check(f'bank{bank}dose{dose}all witness identities',True);models[bank,dose]=F;allowed.add(f'artifacts/research/20261114/loki/h3ak/b{bank}/d{dose}/model.npz')
    check('repair read guard',repair['artifact_read_allowlist_enforced']and set(repair['artifact_read_paths'])<=allowed and p['acquisition']['population_path']not in repair['artifact_read_paths']);ref=R/p['evaluation']['reference_root'];u=np.load(ref/'input-universe.npz');keys=u['eligible_keys'];clues=u['eligible_clues'];truth=np.load(ref/'truth/cache.npz')['columns'];truekeys={r.tobytes()for r in truth};check('same seen universe',len(keys)==s['same_seen_eligible_inputs']==831605)
    with tempfile.TemporaryDirectory(prefix='h3ak-public-census-')as temp:
        temp=Path(temp);binary=temp/'reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        for comp in compiled['supports']:
            group=comp['group'];F=models[comp['representative_bank'],comp['representative_dose']];folder=P/f'support{group}';cache=np.load(folder/'cache.npz');columns=cache['columns'];inp=temp/f'F{group}.bin';out=temp/f'cols{group}.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),'15',str(inp),str(out)],capture_output=True,text=True,check=True);check(f'support{group}complete public reenumeration',int(r.stdout)==len(columns)and np.array_equal(columns,np.fromfile(out,np.uint8).reshape(-1,15)));valid=np.array([r.tobytes()in truekeys for r in columns]);h=hist(columns[valid]);cap=np.load(folder/'capacity.npz');kh=np.load(folder/'key-histogram.npz');check(f'support{group}full census',np.array_equal(cap['eligible_keys'],keys)and np.array_equal(cap['compatible_valid_zero_F'],h[keys])and np.array_equal(cap['valid_state_mask'],valid)and np.array_equal(kh['keys'],np.flatnonzero(h))and np.array_equal(kh['counts'],h[h>0]))
            for member in comp['members']:
                bank,dose=member['bank'],member['dose'];r=next(r for r in s['rows']if r['bank']==bank and r['dose']==dose);check(f'bank{bank}dose{dose}fullsupport',np.array_equal(models[bank,dose],F));check(f'bank{bank}dose{dose}capacity',r['capacity_pct']==100*float((h[keys]>0).mean())and r['valid_state_retention_pct']==100*float(valid.sum()/len(truth)))
                for k in [1,2,3]:part=clues==k;check(f'bank{bank}dose{dose}clue{k}',r['per_clue'][str(k)]['capacity_pct']==100*float((h[keys][part]>0).mean()))
    check('audit source',sha(S/'analyze.py')==s['audit_source_sha256']);check('figure inputs',all(sha(R/rel)==h for rel,h in json.loads((P/'plot-provenance.json').read_text())['input_sha256'].items()));print(json.dumps(dict(status='passed',checks=len(checks),all_selected_draws_and_deletions_replayed=True,all_nine_supports_reenumerated=True,all_key_censuses_rebuilt=True,elapsed_seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
