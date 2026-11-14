"""Audit saved CRAM evidence without fitting, inference, or GPU imports.

Board checks use NumPy and independent enumerators. Sudoku checks are limited
to saved per-puzzle correctness flags, as final Sudoku boards were not retained.
The only outputs are this audit's JSON and Markdown; live result files are read.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import statistics
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream'
EVIDENCE = {}
ROWS = []


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    value = h.hexdigest()
    EVIDENCE[str(path.relative_to(ROOT))] = value
    return value


def read(path):
    path = Path(path)
    data = path.read_bytes()
    EVIDENCE[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def arrays(path):
    sha(path)
    return np.load(path)


def close(a, b):
    assert np.isclose(a, b, rtol=1e-11, atol=1e-8), (a, b)


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else None,
                per_seed=values)


def check(name, fn):
    begin = time.monotonic()
    try:
        row = dict(name=name, status='validated', **fn())
    except FileNotFoundError as exc:
        row = dict(name=name, status='missing', error=str(exc))
    except Exception as exc:
        row = dict(name=name, status='invalid', error=repr(exc), traceback=traceback.format_exc())
    row['audit_seconds'] = time.monotonic() - begin
    ROWS.append(row)
    print(json.dumps(row), flush=True)
    return row


def sudoku(kind):
    learned = kind == 'learned'
    base = UP / ('experiments/runs/CRAM-S/full' if learned else 'experiments/runs/HAND-E1')
    counter = read(ROOT / ('artifacts/sudoku_operation_counts.json' if learned else 'artifacts/sudoku_audit/cram-hand-sudoku.json'))
    original = read(ROOT / f'artifacts/sudoku_audit/{kind}-full-test-audit.json')
    per_seed = []
    for seed in range(3):
        rec = read(base / (f'test/cram0-hb-fit-s{seed}.json' if learned else f'sudoku-test-s{seed}.json'))
        dev = read(base / (f'dev/cram0-hb-fit-s{seed}.json' if learned else f'sudoku-dev-s{seed}.json'))
        sig, pp = rec['signature'], rec['per_puzzle']
        assert sig['seed'] == seed and sig['C'] == 4 and sig['check'] == 4096 and sig['cap'] == 4e11
        assert sig['lo'] == 10000 and sig['n'] == 412786
        assert np.array_equal(pp['puzzle'], np.arange(10000,422786))
        assert dev['summary']['solved'] == dev['summary']['n'] == 10000
        assert dev['summary']['wrong_stops'] == dev['summary']['budget_failures'] == 0
        for key in set(sig)-{'data','lo','n'}:
            assert sig[key] == dev['signature'][key], key
        if learned:
            assert sig['model_sha256'] == sha(base / f'full-aug-s{seed}/model.pt')
            assert sig['test'] == 'fit' and sig['scale'] == 'hbar' and sig['rel'] == [.4,.31,.45]
            move = counter[f"sudoku_general_move_D{sig['D']}"]['total']
            start = counter['sudoku_general_overheads_s0']['init_per_puzzle']['total']
            poll = counter['sudoku_general_overheads_s0']['check_per_puzzle']['total']
        else:
            assert sig['test'] == 'zero' and sig['tau'] == 0 and sig['restart'] == 0
            move = counter['counted']['move_per_chain']['total']
            start = counter['counted']['start_per_puzzle']['total']
            poll = counter['counted']['check_per_puzzle']['total']
        right, stop, raw = np.asarray(pp['solved'],bool), np.asarray(pp['stopped'],bool), np.asarray(pp['flops'])
        assert right.shape == stop.shape == raw.shape == (412786,)
        assert np.isfinite(raw).all() and (raw>=0).all()
        counts = dict(n=len(right), solved=int(right.sum()), stopped=int(stop.sum()),
                      wrong_stops=int((stop & ~right).sum()), budget_failures=int((~stop).sum()))
        assert all(rec['summary'][k]==v for k,v in counts.items())
        steps = raw/(sig['C']*sig['fl_move'])
        assert np.allclose(steps,np.rint(steps),rtol=0,atol=1e-6)
        steps = np.rint(steps)
        flop = steps*sig['C']*move+start+np.ceil(steps/sig['check'])*poll
        close(flop.mean(),original['per_seed'][seed]['paper_flops']['mean'])
        per_seed.append(dict(seed=seed, **counts, percent=100*float(right.mean()), fc2_mean=float(flop.mean())))
    return dict(per_seed=per_seed, accuracy=stats([r['percent'] for r in per_seed]),
                fc2=stats([r['fc2_mean'] for r in per_seed]),
                limitation='Saved flags and identities audited; final Sudoku boards were not retained, so this is not an independent board validity check.')


def queens_solutions(n):
    found = []
    def rec(cols, used, plus, minus):
        row = len(cols)
        if row == n:
            found.append(cols); return
        for col in range(n):
            if col not in used and row+col not in plus and row-col not in minus:
                rec(cols+[col],used|{col},plus|{row+col},minus|{row-col})
    rec([],set(),set(),set())
    out=np.zeros((len(found),n,n),bool)
    out[np.arange(len(found))[:,None],np.arange(n),np.asarray(found)]=True
    assert len(out)=={8:92,10:724}[n]
    return out.reshape(-1,n*n)


def queens(n):
    data = UP / f'data/gram/nqueens-{n}/test'
    given = arrays(data/'all__inputs.npy') == 2
    unique, inverse = np.unique(given,axis=0,return_inverse=True)
    all_solutions = queens_solutions(n)
    counts = np.array([all_solutions[:,row].all(1).sum() for row in unique])
    assert np.array_equal(counts[inverse],arrays(data/'all__num_solutions.npy').ravel())
    caps = np.minimum(counts,20)
    published = read(ROOT/'artifacts/multi_solution/nqueens-paper-results.json')
    rows=[]
    for kind in ['fresh','hand','free','precondition-on','precondition-off']:
        if kind=='precondition-on' and n==10: continue
        for seed in range(3):
            if kind=='fresh': folder=UP/f'experiments/runs/NQ/nq11/n{n}'/('shared' if n==8 else 'final')/f'pl-ising-s{seed}'
            elif kind=='hand': folder=UP/f'experiments/runs/NQ/alt/n{n}/hand/test-s{seed}'
            elif kind=='free': folder=UP/f'experiments/runs/NQ/alt/n{n}/final/{"mpl" if n==8 else "cd"}-s{seed}'
            else:
                suffix='final/mpl-table' if kind=='precondition-on' else ('ablation/mpl-table-noprecond' if n==8 else 'ablation/pl-ising-noprecond')
                folder=UP/f'experiments/runs/NQ/nq11/n{n}/{suffix}-s{seed}'
            if not (folder/'final.json').exists():
                rows.append(dict(kind=kind,seed=seed,status='missing')); continue
            rec=read(folder/'final.json'); z=arrays(folder/'records.npz')
            b=np.unpackbits(z['boards'],axis=-1)[...,:n*n].astype(bool)
            assert b.shape==(len(given),20,n*n)
            sq=b.reshape(len(given),20,n,n)
            ok=(sq.sum(-1)==1).all(-1)&(sq.sum(-2)==1).all(-1)&(b>=given[:,None]).all(-1)
            for delta in range(-n+1,n):
                ok &= np.diagonal(sq,delta,axis1=-2,axis2=-1).sum(-1)<=1
                ok &= np.diagonal(sq[...,::-1],delta,axis1=-2,axis2=-1).sum(-1)<=1
            selected=np.argmin(z['energy'],axis=1)
            found=np.array([len({row.tobytes() for row in b[i,ok[i]]}) for i in range(len(given))])
            found_unique=np.bincount(inverse,weights=np.minimum(found,caps[inverse]))/np.bincount(inverse)
            metrics=dict(acc_n1=100*float(ok[:,0].mean()),acc_n20=100*float(ok[np.arange(len(given)),selected].mean()),
                         cov20_per_puzzle=100*float(found_unique.sum()/caps.sum()),
                         cov20_per_row=100*float(np.minimum(found,caps[inverse]).sum()/caps[inverse].sum()),
                         wrong_stops=int((z['stopped'] & ~ok).sum()),budget_failures=int((~z['stopped']).sum()))
            assert rec['test']['rows']==len(given) and rec['test']['chains']==20
            for k in ['wrong_stops','budget_failures']: assert rec['test'][k]==metrics[k]
            if kind in ['fresh','hand','free']:
                ref=next((r for r in published if r['task']==f'nqueens-{n}' and r['kind']==kind),None)
                if ref:
                    previous=next((r for r in ref['per_seed'] if r['seed']==seed),None)
                    if previous:
                        for k in metrics: close(metrics[k],previous[k])
            rows.append(dict(kind=kind,seed=seed,status='validated',**metrics,records_sha256=sha(folder/'records.npz')))
    return dict(n=n,test_rows=len(given),distinct_inputs=len(unique),independently_enumerated_full_solutions=len(all_solutions),
                stored_solution_counts_verified=True,rows=rows)


def canonical_colorings(n):
    rows=[]
    def rec(prefix,maximum):
        if len(prefix)==n: rows.append(prefix); return
        for x in range(min(2,maximum+1)+1): rec(prefix+[x],max(maximum,x))
    rec([0],0)
    return np.asarray(rows,np.int8)


def coloring(n):
    raw=arrays(UP/f'data/gram/graph-coloring-{n}/test/all__inputs.npy')
    i,j=np.triu_indices(n); keep=i!=j; i,j=i[keep],j[keep]
    edges=raw[:,keep]==2
    canon=canonical_colorings(n)
    total=[]
    for graph in edges:
        total.append(int((~((canon[:,i]==canon[:,j])&graph).any(1)).sum()))
    published=read(ROOT/'artifacts/multi_solution/graph-coloring-paper-results.json')
    paper=read(UP/'results/multi-solution-table.json')
    results=[]
    for kind in ['learned','hand']:
        reference=next(r for r in published if r['task']==f'graph-coloring-{n}' and r['kind']==kind)
        target=next(r for r in paper['rows'] if r['method']==('CRAM (hand-set)' if kind=='hand' else 'CRAM (free pair tables)'))['cells'][f'gc{n}']
        for seed in range(3):
            tag,T=('n8-full-u1024-lr10-lam0.0001',2) if n==8 else ('n10-full-u256-lr10-lam0.001',3)
            folder=UP/(f'experiments/runs/CRAM-GC/hand/n{n}/test-s{seed}' if kind=='hand' else f'experiments/runs/CRAM-GC/final/{tag}/test-terms-T{T}-S20-cap10000-s{seed}')
            rec=read(folder/'final.json'); z=arrays(folder/'outputs.npz'); b=z['boards']
            assert b.shape==(len(raw),20,n) and ((b>=0)&(b<3)).all()
            conflicts=((b[:,:,i]==b[:,:,j])&edges[:,None]).sum(-1)
            canonical=np.empty_like(b)
            for g in range(len(b)):
                for k in range(20):
                    mapping={}
                    canonical[g,k]=[mapping.setdefault(int(v),len(mapping)) for v in b[g,k]]
            proper=conflicts==0; strict=proper & (b==canonical).all(-1)
            strict_count=sum(len({tuple(x) for x in b[g,strict[g]]}) for g in range(len(b)))
            canonical_count=sum(len({tuple(x) for x in canonical[g,proper[g]]}) for g in range(len(b)))
            close(strict_count/sum(total),rec['coverage_20'])
            close(canonical_count/sum(total),rec['coverage_canonicalized_20'])
            assert sum(total)==rec['solutions_total']
            assert int((z['stopped'] & ~proper).sum())==rec['wrong_stops_all']
            assert int((~z['stopped']).sum())==rec['budget_failures_all']
            selected=np.argmin(z['energy'],axis=1)
            metrics=dict(conflicts_n1=int(conflicts[:,0].sum()),conflicts_n20=int(conflicts[np.arange(len(b)),selected].sum()),cov20=100*strict_count/sum(total))
            for k,v in metrics.items():
                close(v,reference['metrics'][k]['per_seed'][seed]); close(v,target[k]['per_seed'][seed])
            results.append(dict(kind=kind,seed=seed,**metrics,cov20_canonicalized=100*canonical_count/sum(total)))
    return dict(n=n,graphs=len(raw),independently_enumerated_canonical_solutions=sum(total),
                all_published_quality_cells_match=True,coverage_rule='Strict proper canonical outputs; canonicalized coverage is a separate diagnostic.',rows=results)


def mazes():
    # Reuse only the independent BFS/path checker, not any model or solver.
    from run_maze_reproduction import path_audit
    rows=[]
    for task in ['maze_unique','maze_hard','hand_maze_unique']:
        for seed in range(3):
            base=ROOT/f'artifacts/fresh_maze/{task}'
            folder=base if task.startswith('hand') else base/f'seed{seed}'
            rec=read(folder/(f'maze-test-s{seed}.json' if task.startswith('hand') else 'test.json'))
            z=arrays(folder/(f'test_predictions-s{seed}.npz' if task.startswith('hand') else 'test_predictions.npz'))
            audit=path_audit(z['inputs'],z['predictions'],z['labels'])
            assert audit['exact_solved']==rec['solved']
            stop=z['stopped'].astype(bool); right=(z['predictions'].astype(bool)==z['labels'].astype(bool)).all((1,2))
            assert int((stop & ~right).sum())==rec['wrong_stops'] and int((~stop).sum())==rec['budget_failures']
            costs=read(folder/'fc2_flops.json')
            if not task.startswith('hand'):
                assert costs['source_test_sha256']==sha(folder/'test.json')
                assert costs['source_model_sha256']==sha(folder/'model.pt')
            if task!='maze_hard':
                fit=costs['fit']['total']; flop=fit['setup']+fit['per']*np.asarray(rec['per_maze']['steps'])
                expected=costs['summaries'][seed]['fc2_flops_mean'] if task.startswith('hand') else costs['fc2_flops_mean']
            else:
                descent=np.asarray(z['steps']); old=np.asarray(rec['per_maze']['flop'])
                # Four latent neighbors; five visible offsets include the center.
                latent=(old-31500-30600*(1+1/16)*descent)/19800
                assert np.array_equal(latent,np.rint(latent))
                fz,fd=costs['latent_fit']['total'],costs['descent_fit']['total']
                flop=fz['setup']+fz['per']*latent+costs['attention']['total']+fd['setup']+fd['per']*descent
                expected=costs['fc2_flops_mean']
            close(float(flop.mean()),expected)
            if task=='maze_unique':
                refdir=ROOT/f'artifacts/reference_maze_unique/seed{seed}'
                ref=read(refdir/'result.json'); other=arrays(refdir/'predictions.npz')
                assert ref['prediction_sha256']==sha(refdir/'predictions.npz')
                assert ref['model_sha256']==sha(refdir/'model.pt')
                assert np.array_equal(other['predictions'].reshape(1000,30,30),z['labels'].astype(bool))
                assert len(ref['losses'])==20 and ref['solved']==1000
            if task=='maze_hard':
                refpath=ROOT/f'artifacts/reference_maze_hard/seed{seed}.json'; ref=read(refpath)
                npz=refpath.with_name(f'seed{seed}-predictions.npz'); other=arrays(npz)
                assert ref['output_sha256']==sha(npz) and ref['checkpoint_sha256']==sha(folder/'model.pt')
                for key in ['predictions','steps','stopped']: assert np.array_equal(other[key],z[key]),key
            rows.append(dict(task=task,seed=seed,solved=int(right.sum()),wrong_indices=np.flatnonzero(~right).tolist(),
                             fc2_mean=float(flop.mean()),bfs_path_audit=audit))
    return dict(rows=rows)


def markers():
    from main_followups import validate_marker
    rows=[]
    for seed in range(3):
        controller=read(ROOT/f'artifacts/main_followups/seed{seed}.json')
        for phase in controller['phases']:
            if 'marker' not in phase:
                rows.append(dict(seed=seed,name=phase['name'],status='not_completed',controller_status=controller['status'])); continue
            marker=Path(phase['marker'])
            assert phase['exit_code']==0 and phase['marker_sha256']==sha(marker)
            validate_marker(phase['name'],marker,seed)
            # Include the external provenance receipt in the audit's evidence list.
            if phase['name'].startswith('nqueens-free-'):
                n=phase['name'].split('-')[-1]; read(ROOT/f'artifacts/multi_solution/free-nq{n}-s{seed}-provenance.json')
            elif phase['name'].startswith('nqueens-'):
                n=phase['name'].split('-')[1]; side=phase['name'].split('-')[-1]
                read(ROOT/f'artifacts/multi_solution/precond-{side}-nq{n}-s{seed}-provenance.json')
            for name in ['model.pt','records.npz']:
                if (marker.parent/name).exists(): sha(marker.parent/name)
            rows.append(dict(seed=seed,name=phase['name'],status='validated',marker=str(marker.relative_to(ROOT))))
    return dict(rows=rows,validated_complete_markers=sum(r['status']=='validated' for r in rows),
                rule='Revalidate completed receipts using current validator; controller PID exit alone is not success.')


def recorded_evidence():
    tests=read(ROOT/'artifacts/reference-tests.json')
    for path,expected in tests['source_sha256'].items(): assert sha(ROOT/path)==expected
    xml=ET.parse(ROOT/'artifacts/reference-tests.xml').getroot()
    cases=xml.findall('.//testcase')
    assert len(cases)==27 and not xml.findall('.//failure') and not xml.findall('.//error')
    sha(ROOT/'artifacts/reference-tests.xml')
    up=read(ROOT/'artifacts/upstream_test_summary.json')
    assert up['combined_passed']==91 and up['combined_failed']==0
    figure=read(ROOT/'artifacts/figures/sudoku-compute-comparison.json')
    for key in ['fresh_cram_source','baseline_source','published_cram_source']:
        assert figure[key+'_sha256']==sha(ROOT/figure[key])
    ratios=[]
    for row in figure['comparisons']:
        ratio=row['baseline_best_setting']['mean_flop']/row['fresh_cram_matching_point']['mean_flop']
        close(ratio,row['baseline_mean_flops_over_fresh_cram'])
        assert row['fresh_cram_matching_point']['solved_mean']>=row['baseline_best_setting']['solve_rate_mean']
        if row['paper_main_figure_comparison']: ratios.append((ratio,row['method']))
    assert len(ratios)==7
    return dict(reference_tests=27,upstream_recorded_passes=91,main_baseline_minimum_ratio=min(ratios),
                limitation='Checks retained test receipts and source hashes without rerunning tests; imported baseline curves are not new baseline results.')


def objectives():
    summary=read(ROOT/'artifacts/objectives/summary.json')
    rows=[]
    for job in summary['jobs']:
        if job['task']!='maze' or job['status']!='complete': continue
        receipt=read(ROOT/job['execution_path'])
        assert receipt['status']=='complete'
        assert receipt['job']['seed']==job['seed'] and receipt['job']['method']==job['method']
        assert job['result']['test']['n']==job['result']['test']['solved']==1000
        assert job['result']['test']['wrong_stops']==job['result']['test']['budget_failures']==0
        assert job['result']['selected_update']==job['final']['selected']['update']
        folder=ROOT/'artifacts/objectives'/job['lane'].split(':')[0]/job['run_relative']
        validated=[]
        for name,expected in receipt['outputs'].items():
            target=folder/name
            assert sha(target)==expected; validated.append(name)
        rows.append(dict(id=job['id'],seed=job['seed'],method=job['method'],solved=1000,
                         local_output_hashes_verified=validated,recorded_output_hashes=receipt['outputs']))
    assert len(rows)==36 and len({(r['method'],r['seed']) for r in rows})==36
    return dict(complete_maze_objective_runs=36,rows=rows,
                limitation='Checks evaluator summaries, completed receipts, and available collected raw hashes; per-maze boards were not retained by this objective evaluator.')


def halting():
    result=read(ROOT/'artifacts/halting-comparison.json')
    for path,expected in result['sources'].items(): assert sha(ROOT/path)==expected
    sudoku_row=next(r for r in ROWS if r['name']=='sudoku_learned')
    maze_row=next(r for r in ROWS if r['name']=='mazes')
    assert sudoku_row['status']==maze_row['status']=='validated'
    for row in result['local']:
        if row['task']=='Sudoku-Extreme':
            audited=sudoku_row['per_seed']
            n=sum(r['n'] for r in audited); solved=sum(r['solved'] for r in audited)
            halted=sum(r['stopped'] for r in audited); wrong=sum(r['wrong_stops'] for r in audited)
        else:
            audited=[r for r in maze_row['rows'] if r['task']=='maze_unique']
            assert len(audited)==3; n=3000; solved=sum(r['solved'] for r in audited); halted=3000; wrong=0
        assert (row['n'],row['solved'],row['halted'],row['wrong_halts'])==(n,solved,halted,wrong)
        close(row['wrong_halts_percent'],100*wrong/halted)
        close(row['halted_percent'],100*halted/n); close(row['solved_percent'],100*solved/n)
    return dict(local=result['local'],published_rows=len(result['published']),
                rule='Wrong halts divided by halted boards; seed evaluations are repeated test inputs.')


def timings():
    sudoku={}
    for device in ['cpu','cuda']:
        row=read(UP/f'experiments/runs/CRAM-S/full/timing/cram0-fit-s0-{device}.json')
        assert len(row['warm_s'])==7 and row['same_support_as_saved']
        close(statistics.median(row['warm_s']),row['warm_median_s'])
        sudoku[device]=row['warm_median_s']
    mapping=read(ROOT/'artifacts/multi_solution/nq8-fit-validation.json')
    for row in mapping: close(row['warm_index_seconds']/row['warm_dense_seconds'],row['dense_vs_index_speedup'])
    counts=read(ROOT/'artifacts/multi_solution/nq-metric-count-validation.json')
    for row in counts:
        close(row['reference_all_inputs_seconds']/(row['enumeration_seconds']+row['all_input_count_seconds']),row['full_count_speedup'])
        assert row['all_reference_counts_match'] and row['all_stored_counts_match']
    return dict(sudoku_warm_median_seconds=sudoku,nq_dense_speedup=[r['dense_vs_index_speedup'] for r in mapping],
                count_speedup=[r['full_count_speedup'] for r in counts],
                limitation='Timing arithmetic verified from saved receipts; no timing rerun or H200 hardware equivalence.')


def main():
    begin=time.monotonic()
    report_path=ROOT/'REPRODUCTION.md'; captured_report=report_path.read_text(); report_sha=sha(report_path)
    for kind in ['learned','hand']: check('sudoku_'+kind,lambda kind=kind:sudoku(kind))
    check('mazes',mazes)
    for n in [8,10]: check(f'nqueens_{n}',lambda n=n:queens(n))
    for n in [8,10]: check(f'graph_coloring_{n}',lambda n=n:coloring(n))
    check('completed_controller_markers',markers)
    check('tests_and_compute_comparison_receipts',recorded_evidence)
    check('maze_objective_receipts',objectives)
    check('table4_halting',halting)
    check('timing_and_optimization_receipts',timings)
    editorial=[]
    multi=(ROOT/'docs/MULTI_SOLUTION_REPRODUCTION.md').read_text()
    if 'after canonical relabeling' in multi:
        editorial.append('Graph-coloring Table 3 uses coverage_20, counting outputs already canonical; wording saying after canonical relabeling should instead reserve that description for coverage_canonicalized_20.')
    result=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),
                source_commit='62bbe830c7d6f87f2ba18f30a28039258b986454',script_sha256=sha(Path(__file__)),
                report_snapshot_sha256=report_sha,report_changed_during_audit=hashlib.sha256(report_path.read_bytes()).hexdigest()!=report_sha,
                status_counts=dict(Counter(r['status'] for r in ROWS)),checks=ROWS,editorial_findings=editorial,
                evidence_sha256=EVIDENCE,elapsed_seconds=time.monotonic()-begin,
                scope='Completed main CRAM results and available controls. Pending runs are not failures or zero-accuracy results. No training or inference was rerun.')
    out=ROOT/'artifacts/completed_controls_audit.json'; out.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Completed result audit','',f"Audit snapshot: {result['generated_utc']}. The captured `REPRODUCTION.md` SHA-256 is `{report_sha}`.",'',
           'This CPU-only audit independently checks saved N-Queens and graph-coloring boards, enumerates exact solution counts, repeats the maze BFS/path checks, and recomputes Sudoku status/FLOP aggregates. It revalidates completed older-controller markers against their model and output hashes. No fitting or inference was rerun.','',
           '| Evidence family | Audit result |','|---|---|']
    lines += [f"| {r['name']} | {r['status']} |" for r in ROWS]
    lines += ['',f"Runtime: {result['elapsed_seconds']:.2f} seconds. Full metrics, errors, per-seed control availability and hash-linked evidence are in [completed_controls_audit.json](../artifacts/completed_controls_audit.json).",'',
              'Sudoku full-test records retain correctness flags but no final boards; aggregate agreement does not provide independent Sudoku board verification. Objective runs retain evaluator summaries and receipts but no saved per-maze boards. Test receipts are checked against source hashes without rerunning their tests. Published baseline comparisons remain imported evidence.','']
    if editorial: lines += ['Editorial findings:','']+[f'- {x}' for x in editorial]+['']
    invalid=[r for r in ROWS if r['status']!='validated']
    if invalid: lines += ['Unresolved checks:','']+[f"- {r['name']}: `{r['error']}`" for r in invalid]+['']
    (ROOT/'docs/COMPLETED_RESULTS_AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out),status_counts=result['status_counts'],elapsed_seconds=result['elapsed_seconds'])))
    return int(any(r['status']=='invalid' for r in ROWS))


if __name__=='__main__': sys.exit(main())
