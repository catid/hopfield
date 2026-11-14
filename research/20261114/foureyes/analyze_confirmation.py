"""Independent saved-board checks and paired uncertainty for H4."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/research/20261114/foureyes'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--confirmation',action='store_true');ap.set_defaults(confirmation=True);args=ap.parse_args()
    protocol_path=ROOT/'research/20261114/foureyes'/('confirmation.json' if args.confirmation else 'protocol.json')
    protocol=json.loads(protocol_path.read_text())
    directory=OUT/('confirmation/jobs' if args.confirmation else 'jobs')
    control=ROOT/'artifacts/research/20261114/ripper'/('confirmation/jobs' if args.confirmation else 'jobs')
    # Confirmation control layout is checked against its actual protocol before use.
    evidence={}
    def digest(p):
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        evidence[str(p.relative_to(ROOT))]=h;return h
    def read(p):digest(p);return json.loads(p.read_text())
    digest(protocol_path);digest(Path(__file__))
    records=[];matrices={};allboards=0
    for arm in ['original']+protocol['arms']:
        cells=[]
        for k in protocol['training_k']:
            seeds=[]
            for seed in protocol['training_seeds']:
                dest=(control if arm=='original' else directory)/f'k{k}-s{seed}-{arm}'
                rec=read(dest/'receipt.json');assert rec['status']=='complete'
                for name,h in rec['output_sha256'].items():assert digest(dest/name)==h
                if arm!='original':
                    assert rec['protocol_sha256']==digest(protocol_path)
                    for name,h in rec['source_sha256'].items():assert digest(ROOT/name)==h
                z=np.load(dest/'boards.npz');b=z['boards'].astype(np.int64);n=len(b);r=b.reshape(n,9,9)
                valid=np.ones(n,dtype=bool)
                for i in range(9):
                    valid&=(np.sort(r[:,i,:],axis=1)==np.arange(9)).all(1)
                    valid&=(np.sort(r[:,:,i],axis=1)==np.arange(9)).all(1)
                for i in range(3):
                    for j in range(3):valid&=(np.sort(r[:,i*3:(i+1)*3,j*3:(j+1)*3].reshape(n,9),axis=1)==np.arange(9)).all(1)
                valid&=((b==z['solutions'])|z['free']).all(1)
                exact=(b==z['solutions']).all(1)
                assert np.array_equal(valid,z['valid']) and np.array_equal(valid,exact) and np.array_equal(exact,z['exact'])
                assert n==protocol['rows_half_open'][1]-protocol['rows_half_open'][0]
                assert exact.sum()==rec['solved']
                assert ((~exact)&z['stopped']).sum()==rec['wrong_stops']
                assert (z['moves']<=protocol['move_cap_per_chain']).all()
                assert (z['moves'][~z['stopped']]==protocol['move_cap_per_chain']).all()
                if arm!='original':assert not ((~valid)&z['stopped']).any()
                records.append({'k':k,'seed':seed,'arm':arm,'n':n,'solved':int(exact.sum()),'wrong_stops':rec['wrong_stops'],
                                'mean_moves':float(z['moves'].mean()),'wall_s':rec['wall_s']})
                seeds.append(exact);allboards+=n
            cells.append(seeds)
        matrices[arm]=np.asarray(cells,dtype=float)
    rng=np.random.default_rng(20261114);nseed=len(protocol['training_seeds']);npuzzle=matrices['original'].shape[-1]
    seed_draws=rng.integers(nseed,size=(5000,nseed));puzzle_draws=rng.integers(npuzzle,size=(5000,npuzzle))
    comparisons=[]
    for arm in protocol['arms']:
        delta=matrices[arm]-matrices['original']
        boot=np.array([delta[:,s][:,:,p].mean()*100 for s,p in zip(seed_draws,puzzle_draws)])
        lo,hi=np.quantile(boot,[.025,.975]);decision='supported' if lo>=10 else ('falsified' if hi<10 else 'inconclusive')
        comparisons.append({'arm':arm,'original_accuracy_pct':float(matrices['original'].mean()*100),
                            'accuracy_pct':float(matrices[arm].mean()*100),'gain_pp':float(delta.mean()*100),
                            'ci95_pp':[float(lo),float(hi)],'threshold_pp':10,'decision':decision})
    receipt={'status':'validated','utc':datetime.now(timezone.utc).isoformat(),'phase':'confirmation' if args.confirmation else 'discovery',
             'comparisons':comparisons,'records':records,'saved_boards_checked':allboards,'input_sha256':evidence,
             'uncertainty':'Paired two-way seed/puzzle bootstrap; small number of training-seed clusters; shared k cells.',
             'scope':'Additional symbolic Sudoku validity gate. Same move cap, potentially higher realized compute. No cross-host timing claim. Baseline shared with H1.'}
    name='confirmation-audit.json' if args.confirmation else 'audit.json'
    (OUT/name).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'validated','comparisons':comparisons,'saved_boards':allboards}))


if __name__=='__main__':main()
