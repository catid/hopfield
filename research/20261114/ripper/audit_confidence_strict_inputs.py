"""Frozen fresh-cohort identity and complete-board source overlap gate."""
import json
import numpy as np
from run import ROOT, sha, save, now
from confidence_strict_common import OUT, PROTOCOL, load_protocol


def main():
    p = load_protocol()
    path = ROOT/'upstream/data/full-test/full-all.npz'
    assert sha(path) == p['test_data_sha256']
    data = np.load(path)
    lo, hi = p['rows_half_open']
    target = data['solutions'][lo:hi].astype(np.int8)-1
    clues = data['clue_mask'][lo:hi]
    assert target.shape == clues.shape == (5000,81)
    assert set(np.unique(clues)) <= {0,1} and np.all((target >= 0) & (target < 9))
    boards = target.reshape(-1,9,9)
    boxes = boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    for digit in range(9):
        assert np.all((boards == digit).sum(1) == 1) and np.all((boards == digit).sum(2) == 1)
        assert np.all((boxes == digit).sum(2) == 1)
    source = ROOT/'artifacts/research/20261114/ripper/confidence_budget/input_manifest.json'
    manifest = json.loads(source.read_text())
    targetbytes = np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps = []
    for entry in manifest['inputs']:
        for role in ('training','calibration'):
            path = ROOT/entry[role+'_path']
            assert sha(path) == entry[role+'_sha256']
            solutions = np.load(path)['solutions']
            count = int(np.isin(np.ascontiguousarray(solutions).view(np.dtype((np.void,81))).ravel(),targetbytes).sum())
            assert count == 0
            overlaps.append({'seed':entry['seed'],'role':role,'shared_complete_boards':count})
    report = {'status':'validated','utc':now(),'protocol_sha256':sha(PROTOCOL),
              'auditor_sha256':sha(__file__),'test_data_sha256':p['test_data_sha256'],
              'rows_half_open':[lo,hi],'puzzles':5000,'source_manifest_sha256':sha(source),
              'complete_board_overlap':overlaps,'outcomes_accessed':0}
    save(OUT/'input_audit.json',report)
    print(json.dumps({'status':'validated','puzzles':5000,'complete_board_overlaps':0}),flush=True)


if __name__ == '__main__':
    main()
