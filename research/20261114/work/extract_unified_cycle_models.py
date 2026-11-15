"""Export immutable checkpoint coefficients for independent NumPy audits."""
import numpy as np
import study as S
import fast_npz as Z
from unified_cycle import PP,DEST,protocol,load_models,ARMS


def main():
    p=protocol();records=[]
    for seed in range(3):
        models,_=load_models(p,seed,'cpu')
        for arm,m in zip(ARMS[:2],models):
            path=DEST/f'model-seed{seed}-{arm}.npz';Z.save(path,theta=np.concatenate([m.T.numpy().ravel(),m.b.numpy(),m.kappa.numpy()]),mu_offsets=np.array(m.mu_offsets),aoffs=np.array(m.aoffs));records.append({'seed':seed,'arm':arm,'path':path.name,'sha256':S.sha(path),'checkpoint':p['models'][str(seed)][arm]})
    S.save(DEST/'model-extraction.json',{'status':'complete','protocol_sha256':S.sha(PP),'records':records,'script_sha256':S.sha(__file__)})


if __name__=='__main__':main()
