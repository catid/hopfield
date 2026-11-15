"""Synthetic H2bq unchanged answer, numerical, replay and lifecycle gates."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import study as S
from audit_bp_first_fresh_states import numeric_error
from audit_bp_first_fresh_costs import expected_batches
from verify_bp_first_fresh_replay import check_trace_pair
from preflight_bp_first_geometry import gate as inherited_gate
from freeze_far_placement import modules
from analyze_far_placement_bins import distance_bin
from bp_first_fresh_pipeline import diagnostic_source_parity


def prefix(path):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'diagnostic_started' for t in n.targets))
    return ast.dump(ast.Module(body=function.body[:end], type_ignores=[]), include_attributes=False)


def rejected(function, exception=AssertionError):
    try:
        function()
    except exception:
        return
    raise AssertionError('Malformed evidence unexpectedly accepted')


def lifecycle(tmp):
    env = {**os.environ, 'PYTHONPATH': str(S.HERE)}
    worker_code = ('import sys; from pathlib import Path; import far_placement_study as M; '
                   'M.DEST=Path(sys.argv[1]); M.PP=Path(sys.argv[2]); M.worker("gate",0)')
    out = tmp / 'failed-worker'
    command = [sys.executable, '-c', worker_code, str(out), str(tmp / 'missing-protocol')]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    receipt = out / 'failure-gate-gpu0.json'
    original = S.sha(receipt)
    assert json.loads(receipt.read_text())['status'] == 'incomplete'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'FileExistsError' in result.stderr and S.sha(receipt) == original
    fifo = tmp / 'blocked-protocol'
    os.mkfifo(fifo)
    out = tmp / 'signal-worker'
    child = subprocess.Popen([sys.executable, '-c', worker_code, str(out), str(fifo)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (out / 'gate-worker0.json').exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(.01)
        child.send_signal(signal.SIGALRM)
        child.communicate(timeout=5)
        failure = json.loads((out / 'failure-gate-gpu0.json').read_text())
        assert failure['status'] == 'incomplete' and 'TimeoutError' in failure['error']
    finally:
        if child.poll() is None:
            child.kill(); child.communicate()
    code = '''
import json, sys
from pathlib import Path
import study as S
import prepare_far_placement as G
import audit_far_placement_inputs as I
base=Path(sys.argv[1]);base.mkdir()
G.DEST=base;G.PP=base/'protocol.json';G.PP.write_text('{}')
G.protocol=lambda:{'cohort':{'base_seed':2026111531,'n':4096},'budgets':{'fresh_generation_wall_seconds':300,'fresh_generation_processes':8}}
for gpu in [0,1]:
 S.save(base/f'gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(G.PP),'source_model_inputs_per_arm':864})
class FailedPool:
 def __init__(self,**kwargs):pass
 def map(self,*args,**kwargs):raise RuntimeError('Synthetic failure before any draw')
 def shutdown(self,**kwargs):pass
G.ProcessPoolExecutor=FailedPool
for iteration in range(2):
 try:G.main()
 except (RuntimeError,FileExistsError) as e:
  assert isinstance(e,RuntimeError if iteration==0 else FileExistsError)
 else:raise AssertionError('Generation attempt accepted')
r=json.loads((base/'generation-worker.json').read_text())
assert r['status']=='incomplete' and r['inputs_drawn']==0
I.DEST=base;I.PP=base/'missing-protocol.json'
for iteration in range(2):
 try:I.main()
 except FileNotFoundError:assert iteration==0
 except FileExistsError:assert iteration==1
 else:raise AssertionError('Input audit failure not preserved')
assert json.loads((base/'input-audit-worker.json').read_text())['status']=='incomplete'
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp / 'cpu-lifecycle')], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return {'new_worker_failure_and_overwrite_refusal': True, 'new_worker_owned_signal_receipt': True,
            'generation_failure_zero_draw_and_overwrite_refusal': True,
            'input_audit_failure_and_overwrite_refusal': True}


def replay_fixture(tmp):
    control, subset = tmp / 'control', tmp / 'subset'
    control.mkdir(); subset.mkdir()
    reference = {'trace_files': [], 'trace_sha256': {}}
    actual = {'trace_files': [], 'trace_sha256': {}}
    arms = ['connectivity_veto', 'original_lexical16', 'unified16', 'unified48', 'unified64', 'unified96']
    for arm in arms:
        for folder, row, n in [(control, reference, 2), (subset, actual, 1)]:
            trace = {'protocol_sha256': 'synthetic', 'arm': arm, 'offset': 224,
                     'initial_terminal_checks': [], 'final_terminal_checks': [],
                     'trace': [{'round': 0, 'geometry': [{'local_input': i, 'candidate_count': 1} for i in range(n)],
                         'candidates': [{'local_input': i, 'cells': [2], 'delta': float('nan')} for i in range(n)],
                         'chosen': [], 'native_results': []}]}
            file = folder / (arm + '.npz')
            np.savez(file, trace_utf8=np.frombuffer(json.dumps(trace).encode(), np.uint8),
                     valid_prefix_indices=np.arange(n), states_before=np.zeros((1,n,5,5),bool),
                     initial_terminal=np.zeros(n,bool), pre_terminal_stopped=np.zeros(n,bool),
                     pre_terminal_block_cap=np.zeros(n,bool))
            row['trace_files'].append(file.name);row['trace_sha256'][file.name]=S.sha(file)
    counts = check_trace_pair(subset, actual, control, reference, np.array([1]), 2)
    assert counts == (6,6,0.)
    assert check_trace_pair(subset, {'trace_files': []}, control, reference, np.array([],int), 2) == (0,0,0.)
    rejected(lambda: check_trace_pair(subset, actual, control, reference, np.array([],int), 2))
    return {'normalized_nonfinite_subset_trace_comparisons': 6, 'empty_trace_cases': 2}


def diagnostic_fixture(tmp):
    source = tmp / 'diagnostic-source'
    source.mkdir()
    energy = np.arange(16, dtype=float).reshape(8,2)
    margin = energy / 10
    native = np.array([[1.,2.]])
    np.savez(source/'batch00.npz', energy64=energy, minimum_flip_margin64=margin,
             native_minimum_flip_margin32=native)
    answer = {'answer':{'prediction':np.zeros((2,5,5),bool)}, 'bp_prefix':{}, 'core':{},
              'subset_indices':np.array([1]), 'audit':{'bp_energy64':energy[0].copy(),
              'bp_minimum_flip_margin32':native[0].copy(), 'LP_minimum_flip_margin64':margin[1:,[1]].copy(),
              'LP_energy64':energy[1:,[1]].copy()}}
    assert diagnostic_source_parity(answer,source,0)['new_diagnostic_array_comparisons']==4
    answer['audit']['LP_energy64'][0,0] += 1
    rejected(lambda: diagnostic_source_parity(answer,source,0))
    return {'new_source_diagnostic_arrays_checked':4,'changed_diagnostic_rejected':True}


def main():
    started = time.perf_counter()
    assert [distance_bin(x) for x in [1,64,65,128,129,256,257,1000]] == ['1..64','1..64','65..128','65..128','129..256','129..256','>=257','>=257']
    for invalid in [0,-1,1.5]:
        rejected(lambda:distance_bin(invalid),ValueError)
    assert prefix(S.HERE / 'bp_first_geometry_batch.py') == prefix(S.HERE / 'bp_first_fresh_batch.py')
    cases = [([1.,2.],[1.,2.]),([np.nan,np.inf,-np.inf,1.],[np.nan,np.inf,-np.inf,1.]),([np.nan],[np.nan]),([],[])]
    for actual, expected in cases:
        assert numeric_error(actual, expected, 1e-8) == 0
    bad = [([np.nan],[0.]),([np.inf],[-np.inf]),([1.],[1.1]),([1.,2.],[1.])]
    for actual, expected in bad:
        rejected(lambda: numeric_error(actual, expected, 1e-8))
    selections = [({'shards':[0,15],'batch_selection':{'0':list(range(8)),'15':[7]}},864),
                  ({'shards':[0],'batch_selection':{'0':[0]}},96),
                  ({'shards':list(range(16)),'batch_selection':None},12288)]
    for selection, count in selections:
        assert len(expected_batches(selection))*32 == count
    for bad_selection in [{'shards':[0,0],'batch_selection':None}, {'shards':[0],'batch_selection':{'0':[0,0]}},
                          {'shards':[0],'batch_selection':{'1':[0]}}, {'shards':[0],'batch_selection':{'0':[8]}}]:
        rejected(lambda:expected_batches(bad_selection))
    # AST inspection binds the only fresh archive reads in the worker: tokens.
    tree = ast.parse((S.HERE / 'far_placement_study.py').read_text())
    archive_reads = [node.slice.value for node in ast.walk(tree) if isinstance(node,ast.Subscript)
                     and isinstance(node.value,ast.Name) and node.value.id=='archive' and isinstance(node.slice,ast.Constant)]
    assert archive_reads == ['tokens','tokens']
    inherited = inherited_gate()
    assert inherited['status'] == 'passed'
    with tempfile.TemporaryDirectory(prefix='h2bo-synthetic-') as temporary:
        tmp = Path(temporary)
        life = lifecycle(tmp)
        replay = replay_fixture(tmp)
        diagnostic = diagnostic_fixture(tmp)
    names = modules()
    for name in names:
        compile((S.HERE / name).read_text(), name, 'exec')
    result = {'status':'passed','synthetic_only':True,
        'complete_answer_segment_AST_identical_to_frozen_H2bn':True,
        'numeric_error_finite_and_nonfinite_cases':len(cases),'inconsistent_numeric_evidence_rejected':len(bad),
        'source_gate_warmup_full_cohort_counts':[x[1] for x in selections], 'malformed_selection_cases':4,'distance_bin_boundary_cases':8,'invalid_distance_cases':3,
        'worker_fresh_archive_reads':archive_reads, **life, **replay, **diagnostic,
        'inherited_complete_lifecycle_and_policy_gate':inherited,
        'fresh_inputs_drawn':0,'fresh_outcomes_read':False,'seconds':time.perf_counter()-started,
        'source_sha256':{name:S.sha(S.HERE/name) for name in names}}
    S.save(S.OUT/'bp_first_far_placement/synthetic-gate.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','inherited_complete_lifecycle_and_policy_gate']},indent=2))


if __name__ == '__main__':
    main()
