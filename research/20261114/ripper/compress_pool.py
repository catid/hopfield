"""Fixed greedy cover and bounded one-thread set-cover MILP; no peer geometry."""
import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
import numpy as np
import highspy
from run import ROOT,HERE,save,sha,now

OUT = ROOT/'artifacts/research/20261114/ripper/compression'
PROTOCOL = HERE/'protocol-compression.json'


def coverage(solutions):
    i,j = np.triu_indices(81,1)
    return solutions[:,i] == solutions[:,j]


def greedy(matrix, source_rows):
    universe = matrix.any(0); remaining = universe.copy(); chosen = []; trace = []
    while remaining.any():
        gains = matrix[:,remaining].sum(1)
        maximum = gains.max(); candidates = np.flatnonzero(gains == maximum)
        pick = candidates[np.argmin(source_rows[candidates])]
        assert maximum>0 and pick not in chosen
        chosen.append(int(pick)); remaining &= ~matrix[pick]
        trace.append({'step':len(chosen),'pool_index':int(pick),'source_row':int(source_rows[pick]),
            'new_pairs':int(maximum),'remaining_pairs':int(remaining.sum())})
    return chosen,trace


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); a = ap.parse_args()
    assert platform.node()=='ripper' and np.__version__=='2.3.4' and highspy.Highs().version()=='1.12.0'
    p = json.loads(PROTOCOL.read_text()); mp = ROOT/p['inputs']; assert sha(mp)==p['input_manifest_sha256']
    manifest = json.loads(mp.read_text()); e = next(e for e in manifest['inputs'] if e['seed']==a.seed)
    source = ROOT/e['path']; assert sha(source)==e['sha256']; z = np.load(source)
    solutions,source_rows = z['solutions'],z['source_row']; assert len(solutions)==1000
    d = OUT/f's{a.seed}'; d.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); matrix = coverage(solutions); universe = matrix.any(0)
    assert np.all(matrix.sum(1)==324)
    selected,trace = greedy(matrix,source_rows)
    reverse,_ = greedy(matrix[::-1],source_rows[::-1])
    assert np.array_equal(source_rows[selected],source_rows[::-1][reverse])
    rng = np.random.default_rng(20261123+a.seed)
    permutations = rng.permuted(np.broadcast_to(np.arange(9),(1000,9)).copy(),axis=1)
    relabeled = permutations[np.arange(1000)[:,None],solutions]
    assert np.array_equal(coverage(relabeled),matrix)
    assert np.array_equal(matrix[selected].any(0),universe)
    np.savez_compressed(d/'greedy_selection.npz',pool_indices=np.array(selected),source_rows=source_rows[selected],
                        solutions=solutions[selected],support=~universe)
    greedy_time = time.monotonic()-start
    pre = {'status':'greedy_complete','utc':now(),'seed':a.seed,'protocol_sha256':sha(PROTOCOL),
        'wrapper_sha256':sha(__file__),'training_sha256':sha(source),'pool_labels_read':1000,
        'universe_size':int(universe.sum()),'pairs_per_board':324,'counting_lower_bound':math.ceil(universe.sum()/324),
        'greedy_size':len(selected),'greedy_trace':trace,'greedy_wall_s':greedy_time,
        'support_bytes_sha256':hashlib.sha256((~universe).tobytes()).hexdigest(),
        'invariance':{'reverse_order_same_source_rows':True,'per_board_digit_relabeling_same_coverage':True,
            'selected_and_full_equality_union_identical':True}}
    save(d/'greedy.json',pre)
    # Sparse binary covering matrix; rows are observed equalities, columns are candidate boards.
    restricted = matrix[:,universe]
    lp = highspy.HighsLp(); lp.num_col_=1000; lp.num_row_=int(universe.sum())
    lp.col_cost_=np.ones(1000); lp.col_lower_=np.zeros(1000); lp.col_upper_=np.ones(1000)
    lp.row_lower_=np.ones(lp.num_row_); lp.row_upper_=np.full(lp.num_row_,highspy.kHighsInf)
    lp.integrality_=[highspy.HighsVarType.kInteger]*1000
    mat = highspy.HighsSparseMatrix(); mat.format_=highspy.MatrixFormat.kColwise
    mat.num_col_=1000; mat.num_row_=lp.num_row_
    mat.start_=np.arange(1001,dtype=np.int32)*324
    mat.index_=np.concatenate([np.flatnonzero(row) for row in restricted]).astype(np.int32)
    mat.value_=np.ones(len(mat.index_)); lp.a_matrix_=mat
    h = highspy.Highs()
    options = {'threads':1,'parallel':'off','time_limit':55.,'mip_rel_gap':0.,'mip_abs_gap':0.,
        'random_seed':20261123+a.seed,'mip_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,
        'log_file':str(d/'highs.log'),'log_to_console':False}
    for name,value in options.items(): assert h.setOptionValue(name,value)==highspy.HighsStatus.kOk,(name,value)
    assert h.passModel(lp)==highspy.HighsStatus.kOk
    warm = np.zeros(1000); warm[selected]=1
    assert h.setSolution(1000,np.arange(1000,dtype=np.int32),warm)==highspy.HighsStatus.kOk
    baseline_cpu = time.process_time(); baseline_wall = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU,(math.ceil(baseline_cpu)+60,math.ceil(baseline_cpu)+61))
    save(d/'solver_running.json',{'utc':now(),'pid':os.getpid(),'solver_options':options,
         'process_cpu_baseline_s':baseline_cpu,'wall_limit_s':55.,'cpu_budget_s':60.})
    call_status = h.run()
    elapsed_cpu = time.process_time()-baseline_cpu; elapsed_wall = time.monotonic()-baseline_wall
    info = h.getInfo(); solution = h.getSolution(); model_status = h.getModelStatus()
    x = np.array(solution.col_value); incumbent = np.flatnonzero(x>.5)
    assert solution.value_valid and np.max(np.abs(x-np.rint(x)))<1e-8
    assert np.array_equal(matrix[incumbent].any(0),universe)
    assert abs(info.objective_function_value-len(incumbent))<1e-7 and elapsed_cpu<=60.
    np.savez_compressed(d/'milp_selection.npz',pool_indices=incumbent,source_rows=source_rows[incumbent],
                        solutions=solutions[incumbent],support=~universe)
    result = dict(pre,status='complete',completed_utc=now(),highs_version=h.version(),numpy=np.__version__,
        solver_status=h.modelStatusToString(model_status),solver_call_status=str(call_status),
        solver_wall_s=elapsed_wall,solver_process_cpu_s=elapsed_cpu,solver_options=options,
        incumbent_size=len(incumbent),solver_dual_bound=float(info.mip_dual_bound),solver_relative_gap=float(info.mip_gap),
        mip_nodes=int(info.mip_node_count),solver_objective=float(info.objective_function_value),
        optimum_claim=model_status==highspy.HighsModelStatus.kOptimal,
        output_sha256={n:sha(d/n) for n in ['greedy_selection.npz','greedy.json','milp_selection.npz','highs.log']})
    save(d/'receipt.json',result)
    print(json.dumps({k:result[k] for k in ['seed','greedy_size','incumbent_size','solver_dual_bound','solver_relative_gap','solver_status','solver_process_cpu_s']}),flush=True)


if __name__ == '__main__': main()
