"""Publish the hand-engineered supplied-graph BFS comparator without study-count inflation."""
import csv
import json
import shutil
import numpy as np
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_global_bound.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_global_bound
    export_global_bound.main();source=S.OUT/'bfs_scope_control';out=S.ROOT/'results/research/work';a=json.loads((source/'audit.json').read_text());assert a['status']=='validated';S.save(out/'bfs-scope-control-summary.json',a)
    with np.load(source/'outputs.npz') as x:distances=x['distances'];status=x['status']
    with (out/'bfs-scope-control-per-input.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['input_index','shortest_edges','independently_validated_shortest_path'])
        writer.writerows((i,int(d),int(ok)) for i,(d,ok) in enumerate(zip(distances,status)))
    shutil.copyfile(source/'outputs.npz',out/'bfs-scope-control-outputs.npz')
    summary=json.loads((out/'summary.json').read_text());summary['hand_engineered_bfs_scope_control']={'reused_h2u_inputs':16384,'independently_correct_shortest_paths':16384,'new_independent_inputs':0,'new_model_evaluations':0,'median_cpu_solver_seconds':a['solver']['median_solve_and_cpu_output_seconds'],'single_threaded':True,'complete_timing_repetitions':3,'scope':'Hand-engineered exact supplied-graph solver, outside learned/CRAMbaseline and never used in its inference.'};summary['updated_utc']=S.utc();S.save(out/'summary.json',summary);manifest(out);S.emit('bfs_scope_public_export',shortest_paths=16384,model_evaluation_total_unchanged=summary['model_maze_evaluations'])


if __name__=='__main__':main()
