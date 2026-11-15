"""Replay the prospective sample and independently match every CSV answer."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/noise_bound'


def main():
    assert np.__version__=='2.3.4'
    ip=HERE/'protocol-noise-bound-inputs.json';p=json.loads(ip.read_text())
    mp=OUT/'input_manifest.json';manifest=json.loads(mp.read_text())
    assert manifest['protocol_sha256']==sha(ip) and manifest['wrapper_sha256']==sha(HERE/'prepare_noise_bound.py')
    source=ROOT/p['source'];assert sha(source)==p['source_sha256']==manifest['source_sha256']
    old=ROOT/p['prior_raw_manifest'];assert sha(old)==p['prior_raw_manifest_sha256']
    oldmanifest=json.loads(old.read_text());assert oldmanifest['source_rows']==oldmanifest['unique_complete_boards']==p['pool_size']
    excluded=np.zeros(p['pool_size'],bool)
    for e in manifest['exclusion_sources']:
        path=ROOT/e['path'];assert sha(path)==e['sha256'];excluded[np.load(path)['source_row']]=True
    eligible=np.flatnonzero(~excluded)
    expected=np.array([np.random.default_rng(20272000+s).choice(eligible,1500,replace=False) for s in range(32)])
    wanted={};allsolutions=[]
    for seed in range(32):
        e=next(e for e in manifest['inputs'] if e['seed']==seed);path=ROOT/e['completion_path']
        assert sha(path)==e['completion_sha256'];z=np.load(path)
        assert np.array_equal(z['source_row'],expected[seed]) and not excluded[z['source_row']].any()
        for row,board in zip(z['source_row'],z['solutions']):
            if int(row) in wanted:assert np.array_equal(wanted[int(row)],board)
            else:wanted[int(row)]=board
        allsolutions.append(z['solutions'])
    solutions=np.concatenate(allsolutions)
    assert len(np.unique(np.ascontiguousarray(solutions).view(np.dtype((np.void,81)))))==len(wanted)
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
        'input_manifest_sha256':sha(mp),'source_sha256':sha(source),'fresh_banks':32,'boards_per_bank':1500,
        'unique_source_rows':len(wanted),'unique_answer_strings':len(wanted),'total_bank_rows':48000,'prior_rows_excluded':int(excluded.sum()),
        'cross_bank_duplicate_row_instances':48000-len(wanted),'within_bank_overlap':0,'prior_bank_overlap':0,'exact_numpy_sample_replayed':True,
        'independent_csv_answers_verified':len(wanted),'numpy':np.__version__}
    save(OUT/'input_audit.json',rec);print(json.dumps(rec))


if __name__=='__main__':main()
