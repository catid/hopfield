"""Resume a saved CRAM search with validated, exact-state GPU move sharding.

The caller owns migration of the old worker and checkpoint. This program
never stops another process and only writes its isolated migration run.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from benchmark_sharded_sudoku import ROOT, reserve, sha, state_digests, compare
import torch
import triton
from cram import cram_sudoku_general as GS
import data_efficiency_common as DC
from sharded_sudoku_moves import ShardedMoves


def write(path, value):
    temp = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.is_relative_to((ROOT / 'artifacts/data_efficiency/migrations').resolve())
    assert run.name == f'k1-s{args.seed}'
    assert not (run / 'test10k.json').exists(), 'Final output already exists; never rescore'
    devices = [int(x) for x in args.devices.split(',')]
    assert devices[0] == 0 and len(devices) >= 2
    proof = json.loads(args.proof.read_text())
    assert proof['status'] == 'validated' and proof['exact_state'] and proof['speedup'] > 1.15
    assert len(proof['devices']) == len(devices), 'A matching shard-count benchmark is required'
    assert proof['host'] == socket.gethostname() and proof['devices'] == devices
    assert proof['versions'] == {'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__}
    assert proof['matmul_flags'] == {'tf32': torch.backends.cuda.matmul.allow_tf32,
                                     'float32_precision': torch.get_float32_matmul_precision()}
    assert proof['sharding_sha256'] == sha(ROOT / 'scripts/sharded_sudoku_moves.py')
    assert proof['script_sha256'] == sha(ROOT / 'scripts/benchmark_sharded_sudoku.py')
    handoff = json.loads(args.handoff.read_text())
    assert handoff['seed'] == args.seed and handoff['k'] == 1
    assert handoff['original_writer_stopped'] is True
    if handoff['source_host'] != socket.gethostname():
        native_path = ROOT / handoff['native_resume_proof_path']
        assert sha(native_path) == handoff['native_resume_proof_sha256']
        native = json.loads(native_path.read_text())
        assert native['status'] == 'passed' and native['seed'] == args.seed
        assert native['from_host'] == handoff['source_host'] and native['to_host'] == socket.gethostname()
        assert native['static_model_and_signature_exact'] is True
        assert len(native['all_state_fields_except_elapsed_exact']) == 24
        assert all(v is True for v in native['all_state_fields_except_elapsed_exact'].values())
    config = json.loads((ROOT / 'configs/data_efficiency_jobs.json').read_text())
    assert handoff['config_sha256'] == sha(ROOT / 'configs/data_efficiency_jobs.json')
    for rel, digest in config['source_sha256'].items():
        assert sha(ROOT / 'dataeff_source' / rel) == digest, rel
    paths = ('model.pt', 'fit.json', 'train.npz', 'test10k-state/ckpt.pt')
    inputs = {name: sha(run / name) for name in paths}
    assert inputs == handoff['input_sha256']
    checkpoint_archive = (ROOT / handoff['checkpoint_archive_path']).resolve()
    assert checkpoint_archive.is_relative_to((ROOT / 'artifacts/data_efficiency/migrations').resolve())
    assert sha(checkpoint_archive) == inputs['test10k-state/ckpt.pt']
    os.environ['CRAM_DATA'] = str(ROOT / 'upstream/data')
    assert sha(ROOT / 'upstream/data/full-test/full-all.npz') == handoff['test_data_sha256']
    torch.set_num_threads(4)
    receipt_path = run.parent / f'sharded-search-s{args.seed}.json'
    admission = (run.parent / f'sharded-search-s{args.seed}.lock').open('a')
    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not receipt_path.exists(), 'Existing execution receipt requires explicit recovery'
    receipt = {'status': 'starting', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pid': os.getpid(), 'host': socket.gethostname(), 'argv': sys.argv, 'devices': devices,
        'seed': args.seed, 'k': 1, 'input_sha256': inputs, 'handoff_sha256': sha(args.handoff),
        'proof_sha256': sha(args.proof), 'config_sha256': handoff['config_sha256'],
        'execution_source_sha256': {name: sha(ROOT / 'scripts' / name) for name in
          ('resume_sharded_data_efficiency.py', 'sharded_sudoku_moves.py', 'benchmark_sharded_sudoku.py')},
        'protocol': 'Original source/checks/refresh/compaction/budget; independent move kernels sharded with original global RNG keys'}
    write(receipt_path, receipt)
    started = time.monotonic()
    try:
        with reserve(devices):
            saved = torch.load(run / 'test10k-state/ckpt.pt', map_location='cpu', weights_only=False)
            receipt.update(status='running', starting_move=saved['state']['t'],
                           starting_active_puzzles=len(saved['state']['act']),
                           prior_search_seconds=saved['state']['elapsed'], original_signature=saved['signature'])
            write(receipt_path, receipt)
            sharded = ShardedMoves([f'cuda:{device}' for device in devices])
            original_moves = GS.GeneralSearch._moves
            validated = False
            last_progress = time.monotonic()
            def dispatch(search, nmoves):
                nonlocal validated, last_progress
                if not validated:
                    # Verify this exact resumed checkpoint before making new
                    # progress. Replayed validation moves are restored first.
                    before = search.state_dict()
                    validation_start = time.monotonic()
                    original_moves(search, nmoves)
                    expected = state_digests(search)
                    search.load_state_dict(before)
                    sharded(search, nmoves)
                    receipt['handoff_parity'] = compare(expected, state_digests(search), 'actual_resume/pre_refresh')
                    receipt['handoff_validation_seconds'] = time.monotonic() - validation_start
                    write(receipt_path, receipt)
                    validated = True
                else:
                    sharded(search, nmoves)
                if time.monotonic() - last_progress >= 30:
                    receipt['progress'] = {'moves_before_check': search.t,
                        'active_puzzles': len(search.act),
                        'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    write(receipt_path, receipt)
                    last_progress = time.monotonic()
            GS.GeneralSearch._moves = dispatch
            DC.ROOT = run.parent
            DC.search(1, args.seed, 'test10k')
            result = json.loads((run / 'test10k.json').read_text())
            assert result['signature'] == saved['signature']
            assert result['resumes'] == saved['resumes'] + 1
            assert result['summary']['n'] == 10000
            assert result['per_puzzle']['puzzle'] == list(range(10000, 20000))
            receipt.update(status='complete', exit_code=0, elapsed_wall_s=time.monotonic()-started,
                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                result_sha256=sha(run / 'test10k.json'), summary=result['summary'],
                final_moves=result['per_puzzle']['moves'], search_seconds=result['per_puzzle']['seconds'])
            write(receipt_path, receipt)
    except BaseException:
        receipt.update(status='failed', elapsed_wall_s=time.monotonic()-started, error=traceback.format_exc())
        write(receipt_path, receipt)
        raise
    print(json.dumps({'status': receipt['status'], 'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()
