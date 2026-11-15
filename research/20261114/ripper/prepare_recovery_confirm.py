"""Independent fresh raw banks with possible cross-bank overlap, prepared without acquisition outcomes."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/recovery_confirm'
PROTOCOL=HERE/'protocol-recovery-confirm-inputs.json'


def main():
    start_utc=now()
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
    eligible=np.flatnonzero(~exclusion)
    selected=np.array([np.random.default_rng(20262000+s).choice(eligible,600,replace=False) for s in range(256)])
    assert all(len(np.unique(rows))==600 for rows in selected) and not exclusion[selected].any()
    OUT.mkdir(exist_ok=False);answers=np.zeros((256,600,81),np.int8)
    wanted={}
    for s in range(256):
        for k,row in enumerate(selected[s]):wanted.setdefault(int(row),[]).append((s,k))
    remaining=set(wanted)
    with source.open(newline='') as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            if row_index not in wanted:continue
            answer=np.frombuffer(row['answer'].encode('ascii'),np.uint8).astype(np.int8)-ord('1')
            assert len(answer)==81 and np.all((answer>=0)&(answer<9));
            for s,k in wanted[row_index]:answers[s,k]=answer
            remaining.remove(row_index)
            if not remaining:break
    assert not remaining
    flat=answers.reshape(153600,81);b=flat.reshape(-1,9,9);boxes=b.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    for digit in range(9):
        assert np.all((b==digit).sum(1)==1) and np.all((b==digit).sum(2)==1)
        assert np.all((boxes==digit).sum(2)==1)
    assert len(np.unique(np.ascontiguousarray(flat).view(np.dtype((np.void,81)))))==len(np.unique(selected))
    entries=[]
    for seed in range(256):
        d=OUT/'inputs';d.mkdir(exist_ok=True);path=d/f's{seed}.npz'
        np.savez_compressed(path,solutions=answers[seed],source_row=selected[seed])
        entries.append({'seed':seed,'n':600,'completion_path':str(path.relative_to(ROOT)),'completion_sha256':sha(path)})
    save(OUT/'input_manifest.json',{'status':'prepared_without_outcome_inspection','utc':now(),'preparation_started_utc':start_utc,
        'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),'source_sha256':sha(source),
        'numpy':np.__version__,'excluded_unique_source_rows':int(exclusion.sum()),'exclusion_sources':exclusion_sources,
        'fresh_unique_rows':int(len(np.unique(selected))),'total_bank_rows':153600,'cross_bank_duplicate_row_instances':int(153600-len(np.unique(selected))), 'within_bank_overlap':0,'prior_bank_overlap':0,'inputs':entries})
    print(json.dumps({'prepared_banks':256,'rows_each':600,'excluded_rows':int(exclusion.sum()),'fresh_unique_rows':int(len(np.unique(selected)))}))


if __name__=='__main__':main()
