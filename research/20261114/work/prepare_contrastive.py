"""Hash actual original training arrays and exact frozen-readout caches, no fits."""
import argparse
import hashlib
import json
import numpy as np
import study as S
import selective_fallback as F

DEST=S.OUT/'contrastive_training'


def arrays():
    torch,_,_=S.imports();from cram.maze_learned import load,DATA
    DEST.mkdir(parents=True,exist_ok=True);tokens,labels=load('train');assert tokens.shape==labels.shape==(1000,30,30)
    assert tokens.dtype==np.int64 and labels.dtype==bool
    path=DEST/'training-arrays.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,tokens=tokens,labels=labels)
    S.save(DEST/'training-manifest.json',{'utc':S.utc(),'source':'Actual upstream data/maze-hard/train.csv loaded by original maze_learned.load; not recreated or generated.','source_csv_path':str((DATA/'train.csv').relative_to(S.ROOT)),'source_csv_sha256':S.sha(DATA/'train.csv'),'loader_sha256':S.sha(S.UP/'cram/maze_learned.py'),'n_training':1000,'shape':[1000,30,30],'arrays_sha256':S.sha(path),'tokens_sha256':hashlib.sha256(tokens.tobytes()).hexdigest(),'labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'script_sha256':S.sha(__file__),'numpy':np.__version__})
    S.emit('contrastive_actual_training_arrays',n=1000,sha256=S.sha(path))


def cache(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();manifest=json.loads((DEST/'training-manifest.json').read_text());assert S.sha(DEST/'training-arrays.npz')==manifest['arrays_sha256']
    with np.load(DEST/'training-arrays.npz') as x:tokens=x['tokens']
    for seed in range(3):
        if seed%2!=gpu:continue
        model=S.model_for(seed,30,'original','cuda');couplings=[];hidden_steps=[]
        with torch.no_grad():
            for lo in range(0,1000,64):
                t=torch.tensor(tokens[lo:lo+64],device='cuda');psi,steps=model.settle(t,tol=1e-6,max_steps=4000,check=8);couplings.append(model.couplings(t,psi).cpu().numpy());hidden_steps.append(steps.cpu().numpy())
        a=np.concatenate(couplings);path=DEST/f'train-readout-seed{seed}.npz'
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,couplings=a,hidden_steps=np.concatenate(hidden_steps));S.save(DEST/f'train-readout-seed{seed}.json',{'seed':seed,'gpu':gpu,'inventory':inventory,'arrays_sha256':manifest['arrays_sha256'],'checkpoint_sha256':S.sha(S.OUT/f'models/seed{seed}.pt'),'readout_sha256':S.sha(path),'coupling_array_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'hidden_batch_size':64,'tol':1e-6,'cap':4000,'check_every':8,'frozen_parameters':['mu','beta'],'script_sha256':S.sha(__file__)})
        S.emit('contrastive_readout_cached',seed=seed,shape=a.shape)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['arrays','cache']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='arrays':arrays()
    else:cache(args.gpu)
