"""Deterministic single-thread HiGHS 1.12.0 incidence LP, no geometry input."""
import time
import numpy as np
import highspy


def solve(n,degree,ii,jj,cost,seed,log_path):
    assert highspy.Highs().version()=='1.12.0' and np.__version__=='2.3.4'
    count=len(cost);assert ii.shape==jj.shape==cost.shape and count>0
    lp=highspy.HighsLp();lp.num_col_=count;lp.num_row_=n
    lp.col_cost_=cost.astype(np.float64);lp.col_lower_=np.zeros(count);lp.col_upper_=np.ones(count)
    lp.row_lower_=np.full(n,degree,dtype=np.float64);lp.row_upper_=np.full(n,degree,dtype=np.float64)
    matrix=highspy.HighsSparseMatrix();matrix.format_=highspy.MatrixFormat.kColwise
    matrix.num_col_=count;matrix.num_row_=n;matrix.start_=np.arange(count+1,dtype=np.int32)*2
    matrix.index_=np.column_stack((ii,jj)).ravel().astype(np.int32);matrix.value_=np.ones(2*count)
    lp.a_matrix_=matrix
    options=dict(threads=1,parallel='off',solver='simplex',simplex_strategy=1,presolve='off',
                 time_limit=1.,random_seed=2026111502+seed,primal_feasibility_tolerance=1e-9,
                 dual_feasibility_tolerance=1e-9,log_to_console=False,log_file=str(log_path))
    h=highspy.Highs()
    for name,value in options.items():assert h.setOptionValue(name,value)==highspy.HighsStatus.kOk,(name,value)
    assert h.passModel(lp)==highspy.HighsStatus.kOk
    started=time.monotonic();cpu=time.process_time();call=h.run()
    wall=time.monotonic()-started;elapsed_cpu=time.process_time()-cpu
    info=h.getInfo();solution=h.getSolution();status=h.getModelStatus()
    value=float(info.objective_function_value)
    metadata=dict(highs_version=h.version(),numpy=np.__version__,solver_options=options,
                  solver_status=h.modelStatusToString(status),solver_call_status=str(call),
                  solver_optimal=status==highspy.HighsModelStatus.kOptimal,
                  value_valid=bool(solution.value_valid),dual_valid=bool(solution.dual_valid),
                  solver_objective=value if np.isfinite(value) else None,
                  simplex_iterations=int(info.simplex_iteration_count),solver_wall_s=wall,solver_cpu_s=elapsed_cpu)
    return np.asarray(solution.col_value,np.float64),np.asarray(solution.row_dual,np.float64),metadata
