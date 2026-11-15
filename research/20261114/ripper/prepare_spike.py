"""Fresh globally disjoint raw banks, prepared without acquisition outcomes."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/spike'
PROTOCOL=HERE/'protocol-spike-inputs.json'


def main():
    assert np.__version__=='2.3.4'
    p=json.loads(PROTOCOL.read_text());source=ROOT/p['source'];assert sha(source)==p['source_sha256']
    oldpath=ROOT/p['prior_raw_manifest'];assert sha(oldpath)==p['prior_raw_manifest_sha256']
    old=json.loads(oldpath.read_text());assert old['source_rows']==old['unique_complete_boards']==p['pool_size']
    exclusion=np.zeros(p['pool_size'],bool);exclusion_sources=[]
    for spec in p['exclusion_manifests']:
        ep=ROOT/spec['path'];assert sha(ep)==spec['sha256'];excluded=json.loads(ep.read_text())
        for e in excluded['inputs']:
            path=ROOT/e['completion_path'];assert sha(path)==e['completion_sha256'];z=np.load(path)
            exclusion[z['source_row']]=True;exclusion_sources.append({'path':e['completion_path'],'sha256':sha(path)})
    permutation=np.random.default_rng(20261202).permutation(p['pool_size'])
    selected=permutation[~exclusion[permutation]][:9600].reshape(32,300)
    assert len(np.unique(selected))==9600 and not exclusion[selected].any()
    OUT.mkdir(exist_ok=False);answers=np.zeros((32,300,81),np.int8)
    wanted={int(row):(s,k) for s in range(32) for k,row in enumerate(selected[s])};remaining=set(wanted)
    with source.open(newline='') as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            if row_index not in wanted:continue
            s,k=wanted[row_index]
            answer=np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1')
            assert len(answer)==81 and np.all((answer>=0)&(answer<9));answers[s,k]=answer
            remaining.remove(row_index)
            if not remaining:break
    assert not remaining
    flat=answers.reshape(9600,81);b=flat.reshape(-1,9,9);boxes=b.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    for digit in range(9):
        assert np.all((b==digit).sum(1)==1) and np.all((b==digit).sum(2)==1)
        assert np.all((boxes==digit).sum(2)==1)
    assert len(np.unique(np.ascontiguousarray(flat).view(np.dtype((np.void,81)))))==9600
    entries=[]
    for seed in range(32):
        d=OUT/'inputs';d.mkdir(exist_ok=True);path=d/f's{seed}.npz'
        np.savez_compressed(path,solutions=answers[seed],source_row=selected[seed])
        entries.append({'seed':seed,'n':300,'completion_path':str(path.relative_to(ROOT)),'completion_sha256':sha(path)})
    save(OUT/'input_manifest.json',{'status':'prepared_without_outcome_inspection','utc':now(),
        'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'source_sha256':sha(source),
        'numpy':np.__version__,'excluded_unique_source_rows':int(exclusion.sum()),'exclusion_sources':exclusion_sources,
        'fresh_unique_rows':9600,'cross_bank_overlap':0,'prior_bank_overlap':0,'inputs':entries})
    print(json.dumps({'prepared_banks':32,'rows_each':300,'excluded_rows':int(exclusion.sum()),'fresh_rows':9600}))


if __name__=='__main__':main()
