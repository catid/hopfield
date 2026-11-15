"""Independent source sampling and CSV verification for calibration and training."""
import argparse
import csv
import json
import numpy as np
from run import ROOT,HERE,sha,save,now
from confidence_budget_common import paths,load_protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    assert np.__version__=='2.3.4';out,pp=paths(args.pilot);p=load_protocol(args.pilot);mp=out/'input_manifest.json';m=json.loads(mp.read_text())
    if args.pilot:
        path=ROOT/p['prior_input_manifest'];assert sha(path)==p['prior_input_manifest_sha256'];old=json.loads(path.read_text());expected={};pilotboards={}
        for seed in p['seeds']:
            e=next(e for e in old['inputs'] if e['seed']==seed)
            for role,n in [('training',1110),('calibration',1000)]:
                path=ROOT/e[role+'_path'];assert sha(path)==e[role+'_sha256'];z=np.load(path)
                expected[role,seed]=z['source_row'][:n];pilotboards[role,seed]=z['solutions'][:n]
        excluded=None
    else:
        ip=HERE/'protocol-confidence-budget-inputs.json';spec=json.loads(ip.read_text());assert m['protocol_sha256']==sha(ip)==p['input_protocol_sha256']
        raw=ROOT/spec['prior_raw_manifest'];assert sha(raw)==spec['prior_raw_manifest_sha256'];proof=json.loads(raw.read_text())
        assert proof['source_rows']==proof['unique_complete_boards']==spec['pool_size']
        excluded=np.zeros(spec['pool_size'],bool)
        for e in m['exclusion_sources']:
            assert sha(ROOT/e['path'])==e['sha256'];excluded[np.load(ROOT/e['path'])['source_row']]=True
        assert excluded.sum()==1729093
        pool=np.random.default_rng(p['streams']['pool_split']).permutation(np.flatnonzero(~excluded));half=len(pool)//2
        expected={}
        for seed in p['seeds']:
            expected['calibration',seed]=np.random.default_rng(p['streams']['calibration_source']+seed).choice(pool[:half],1000,replace=False)
            expected['training',seed]=np.random.default_rng(p['streams']['training_source']+seed).choice(pool[half:],1110,replace=False)
    wanted={};roleids={'calibration':set(),'training':set()}
    for e in m['inputs']:
        seed=e['seed']
        for role in roleids:
            path=ROOT/e[role+'_path'];assert sha(path)==e[role+'_sha256'];z=np.load(path);ids=z['source_row'];boards=z['solutions']
            assert np.array_equal(ids,expected[role,seed]) and len(np.unique(ids))==len(ids)
            assert len(boards)==(1000 if role=='calibration' else 1110)
            if excluded is not None:assert not excluded[ids].any()
            if args.pilot:assert np.array_equal(boards,pilotboards[role,seed])
            for index,board in zip(ids,boards):
                index=int(index);roleids[role].add(index)
                if index in wanted:assert np.array_equal(wanted[index],board)
                else:wanted[index]=board
    assert not roleids['calibration']&roleids['training']
    source=ROOT/'upstream/data/sudoku-extreme/train.csv';assert sha(source)=='64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e'
    remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for index,row in enumerate(reader):
            if index not in wanted:continue
            board=np.array([int(c)-1 for c in row[column]],np.int8);assert np.array_equal(board,wanted[index]);remaining.remove(index)
            if not remaining:break
    assert not remaining
    unique_boards=len(np.unique(np.array(list(wanted.values())).view(np.dtype((np.void,81)))))
    assert unique_boards==len(wanted)
    if not args.pilot:
        selection=np.load(out/'selected_sources.npz');assert sha(out/'selected_sources.npz')==m['selected_sources_sha256']
        assert np.array_equal(selection['all_source_row'],sorted(wanted))
        am=out/'all_source_manifest.json';assert sha(am)==m['all_source_manifest_sha256']
        for e in json.loads(am.read_text())['inputs']:assert sha(ROOT/e['completion_path'])==e['completion_sha256']
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'protocol_sha256':sha(pp),'input_manifest_sha256':sha(mp),
       'banks':len(p['seeds']),'calibration_unique_rows':len(roleids['calibration']),'training_unique_rows':len(roleids['training']),
       'calibration_training_overlap':0,'within_bank_overlap':0,'prior_bank_overlap':0 if not args.pilot else None,
       'exact_numpy_sample_replayed':True,'independent_csv_answers_verified':len(wanted),'unique_answer_strings':unique_boards,'numpy':np.__version__}
    save(out/'input_audit.json',r);print(json.dumps(r),flush=True)


if __name__=='__main__':main()
