"""Replay the prospective sample and independently match every CSV answer."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/online_confirm'


def main():
    assert np.__version__=='2.3.4'
    ip=HERE/'protocol-online-confirm-inputs.json';p=json.loads(ip.read_text())
    mp=OUT/'input_manifest.json';manifest=json.loads(mp.read_text())
    assert manifest['protocol_sha256']==sha(ip) and manifest['wrapper_sha256']==sha(HERE/'prepare_online_confirm.py')
    source=ROOT/p['source'];assert sha(source)==p['source_sha256']==manifest['source_sha256']
    old=ROOT/p['prior_raw_manifest'];assert sha(old)==p['prior_raw_manifest_sha256']
    oldmanifest=json.loads(old.read_text());assert oldmanifest['source_rows']==oldmanifest['unique_complete_boards']==p['pool_size']
    excluded=np.zeros(p['pool_size'],bool)
    for e in manifest['exclusion_sources']:
        path=ROOT/e['path'];assert sha(path)==e['sha256'];excluded[np.load(path)['source_row']]=True
    draw=np.random.default_rng(20261128).permutation(p['pool_size'])
    expected=draw[~excluded[draw]][:9600].reshape(32,300)
    wanted={};allsolutions=[]
    for seed in range(32):
        e=next(e for e in manifest['inputs'] if e['seed']==seed);path=ROOT/e['completion_path']
        assert sha(path)==e['completion_sha256'];z=np.load(path)
        assert np.array_equal(z['source_row'],expected[seed]) and not excluded[z['source_row']].any()
        for row,board in zip(z['source_row'],z['solutions']):
            assert int(row) not in wanted;wanted[int(row)]=board
        allsolutions.append(z['solutions'])
    solutions=np.concatenate(allsolutions)
    assert len(np.unique(np.ascontiguousarray(solutions).view(np.dtype((np.void,81)))))==9600
    remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);header=next(reader);answercol=header.index('answer')
        for index,row in enumerate(reader):
            if index not in wanted:continue
            board=np.array([int(c)-1 for c in row[answercol]],dtype=np.int8)
            assert np.array_equal(board,wanted[index]);remaining.remove(index)
            if not remaining:break
    assert not remaining
    rec={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'input_protocol_sha256':sha(ip),
        'input_manifest_sha256':sha(mp),'source_sha256':sha(source),'fresh_banks':32,'boards_per_bank':300,
        'unique_source_rows':9600,'unique_answer_strings':9600,'prior_rows_excluded':int(excluded.sum()),
        'cross_bank_overlap':0,'prior_bank_overlap':0,'exact_numpy_sample_replayed':True,
        'independent_csv_answers_verified':9600,'numpy':np.__version__}
    save(OUT/'input_audit.json',rec);print(json.dumps(rec))


if __name__=='__main__':main()
