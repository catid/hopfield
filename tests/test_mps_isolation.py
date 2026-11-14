"""Independent MPS ownership checks using fake /proc files, no GPU calls."""
import itertools
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mps_probes import mps_conflicts

A='GPU-abcdef12-3456-789a-bcde-f0123456789a'
B='GPU-22222222-2222-2222-2222-222222222222'
COUNTER=itertools.count()


@pytest.fixture
def private_pipe():
    for number in COUNTER:
        path=Path('/tmp')/f'hopfield-probes-{os.getpid()}999{number}'
        try:path.mkdir(mode=0o700);break
        except FileExistsError:continue
    yield path
    path.rmdir()


def evaluate(tmp_path,visible,pipe,raw=None):
    proc=tmp_path/'123';proc.mkdir(exist_ok=True)
    (proc/'environ').write_bytes(raw if raw is not None else
        f'CUDA_VISIBLE_DEVICES={visible}\0CUDA_MPS_PIPE_DIRECTORY={pipe}\0'.encode())
    return mps_conflicts(A,proc_root=tmp_path,processes=[dict(pid=123,executable='nvidia-cuda-mps-server')])


def test_disjoint_explicit_uuid_and_private_pipe_are_eligible(tmp_path,private_pipe):
    assert evaluate(tmp_path,B,private_pipe)==[]


@pytest.mark.parametrize('visible',[A,A.upper(),'0','',f'{B},{A}','GPU-'+'z'*36])
def test_overlapping_numeric_unset_or_malformed_device_is_rejected(tmp_path,private_pipe,visible):
    assert evaluate(tmp_path,visible,private_pipe)


def test_shared_mode_is_rejected(tmp_path,private_pipe):
    private_pipe.chmod(0o755)
    assert evaluate(tmp_path,B,private_pipe)


def test_missing_or_non_task_pipe_is_rejected(tmp_path,private_pipe):
    assert evaluate(tmp_path,B,str(private_pipe)+'987654321')
    assert evaluate(tmp_path,B,tmp_path)


def test_symlink_is_rejected(tmp_path,private_pipe):
    target=tmp_path/'directory';target.mkdir(mode=0o700)
    private_pipe.rmdir();private_pipe.symlink_to(target,target_is_directory=True)
    try:assert evaluate(tmp_path,B,private_pipe)
    finally:private_pipe.unlink();private_pipe.mkdir(mode=0o700)


def test_unreadable_environment_is_rejected(tmp_path,private_pipe,monkeypatch):
    evaluate(tmp_path,B,private_pipe)
    original=Path.read_bytes
    def denied(path):
        if path==tmp_path/'123/environ':raise PermissionError('synthetic denied environment')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',denied)
    assert mps_conflicts(A,proc_root=tmp_path,processes=[dict(pid=123)])


def test_non_utf8_environment_is_rejected(tmp_path,private_pipe):
    assert evaluate(tmp_path,B,private_pipe,raw=b'CUDA_VISIBLE_DEVICES=\xff\0')


def test_foreign_owned_pipe_is_rejected(tmp_path,private_pipe,monkeypatch):
    original=Path.lstat
    def changed_owner(path):
        meta=original(path)
        if path==private_pipe:
            fields=list(meta);fields[4]=os.getuid()+1;return os.stat_result(fields)
        return meta
    monkeypatch.setattr(Path,'lstat',changed_owner)
    assert evaluate(tmp_path,B,private_pipe)


def test_exited_process_is_not_a_live_conflict(tmp_path):
    assert mps_conflicts(A,proc_root=tmp_path,processes=[dict(pid=99999)])==[]
