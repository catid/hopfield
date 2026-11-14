"""Frozen deduplication and nested raw-board draws, before support inspection."""
import csv
import hashlib
import json
import numpy as np
from run import ROOT,HERE,save,sha,now


def main():
    p=json.loads((HERE/'protocol-raw-count.json').read_text());src=ROOT/p['source'];assert sha(src)==p['source_sha256']
    out=ROOT/'artifacts/research/20261114/ripper/raw';out.mkdir(exist_ok=False)
    seen=set();answers=[];first_rows=[];digest=hashlib.sha256();total=0
    with src.open(newline='') as f:
        reader=csv.DictReader(f)
        for row_index,row in enumerate(reader):
            total+=1;answer=row['answer'];assert len(answer)==81 and all('1'<=v<='9' for v in answer)
            if answer in seen:continue
            seen.add(answer);answers.append(answer);first_rows.append(row_index);digest.update(row_index.to_bytes(8,'little'));digest.update(answer.encode('ascii'))
    del seen
    first_rows=np.array(first_rows,np.int64);wanted={};datasets={};indices={}
    for seed in p['seeds']:
        chosen=np.random.default_rng(20261116+seed).permutation(len(answers))[:1000];rows=first_rows[chosen];indices[seed]=chosen
        sol=np.array([np.frombuffer(answers[i].encode('ascii'),np.uint8).astype(np.int8)-ord('1') for i in chosen])
        datasets[seed]={'solutions':sol,'free':np.zeros((1000,81),bool),'source_row':rows,'unique_pool_index':chosen}
        for i,row in enumerate(rows):wanted.setdefault(int(row),[]).append((seed,i))
    del answers
    remaining=set(wanted)
    with src.open(newline='') as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            if row_index not in wanted:continue
            tokens=np.frombuffer(row['question'].replace('.','0').encode('ascii'),np.uint8).astype(np.int8)-ord('0')
            for seed,i in wanted[row_index]:
                d=datasets[seed];d['free'][i]=tokens==0;assert np.all((tokens==0)|(tokens-1==d['solutions'][i]))
            remaining.remove(row_index)
            if not remaining:break
    assert not remaining
    manifest=[]
    for seed,d in datasets.items():
        b=d['solutions'].reshape(-1,9,9);boxes=b.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert np.all((b==digit).sum(1)==1) and np.all((b==digit).sum(2)==1) and np.all((boxes==digit).sum(2)==1)
        dest=out/f'inputs/s{seed}';dest.mkdir(parents=True);np.savez_compressed(dest/'train.npz',**d)
        manifest.append({'seed':seed,'path':str((dest/'train.npz').relative_to(ROOT)),'sha256':sha(dest/'train.npz'),'n':1000})
    overlap=[{'seed_a':a,'seed_b':b,'shared_boards':int(len(np.intersect1d(indices[a],indices[b])))} for a in range(6) for b in range(a+1,6)]
    save(out/'input_manifest.json',{'status':'prepared','utc':now(),'protocol_sha256':sha(HERE/'protocol-raw-count.json'),'wrapper_sha256':sha(__file__),'source_rows':total,'unique_complete_boards':len(first_rows),'ordered_unique_pool_sha256':digest.hexdigest(),'inputs':manifest,'cross_seed_overlap':overlap,'support_inspection_performed':False})
    print(json.dumps({'source_rows':total,'unique_complete_boards':len(first_rows),'prepared_seeds':6,'overlap':overlap}))


if __name__=='__main__':main()
