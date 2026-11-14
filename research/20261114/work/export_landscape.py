"""Publish the audited exhaustive shortest-path energy counterexample."""
import csv
import json
import numpy as np
import study as S


def main():
    import export_fork_angle
    export_fork_angle.main();out=S.ROOT/"results/research/work";source=S.OUT/"path_landscape";audit=json.loads((source/"audit.json").read_text());assert audit["status"]=="validated"
    S.save(out/"shortest-path-energy-summary.json",{k:v for k,v in audit.items() if k!="evidence_sha256"}|{"full_local_audit_sha256":S.sha(source/"audit.json")})
    rows=[];histograms=[];extrema=[]
    for r in audit["rows"]:
        rows.append({k:r[k] for k in ["seed","path_count","min_valid_energy","max_valid_energy","endpoint_only_energy","retained_eta05_energy","endpoint_minus_min_valid","retained_minus_min_valid","cpu_full_grid_max_error","energy_evaluation_seconds","full_grid_over_sparse_timing_ratio"]})
        energies=np.load(source/f"seed{r['seed']}/energies.npy");counts,bins=np.histogram(energies-r["min_valid_energy"],bins=80)
        histograms.extend({"seed":r["seed"],"delta_energy_left":float(bins[i]),"delta_energy_right":float(bins[i+1]),"paths":int(n)} for i,n in enumerate(counts))
        paths=np.load(source/f"seed{r['seed']}/extrema-paths.npz")
        extrema.append({"seed":r["seed"],"shape":list(paths["tokens"].shape),"minimum_rank":int(paths["min_rank"]),"maximum_rank":int(paths["max_rank"]),"minimum_path_flat_indices":paths["min_path"].tolist(),"maximum_path_flat_indices":paths["max_path"].tolist()})
    for name,values in [("shortest-path-energy-ranges.csv",rows),("shortest-path-energy-histogram.csv",histograms)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    S.save(out/"shortest-path-energy-extrema.json",{"graph":"cyclic-failure-case.json","flat_index_convention":"row*90+column","models":extrema})
    summary=json.loads((out/"summary.json").read_text());summary["post_outcome_energy_diagnostics"]={"unique_graphs":1,"shortest_paths_per_model":352212,"frozen_models":3,"model_energy_evaluations":1056636,"additional_neural_rollouts":0,"summary":"shortest-path-energy-summary.json","figure":"shortest-path-energy-landscape.png"}
    pp=S.HERE/"path_landscape-v2-protocol.json";protocol=json.loads(pp.read_text());summary["protocols"]["path_landscape-v2-protocol"]={"sha256":S.sha(pp),"frozen_utc":protocol["frozen_utc"]}
    summary["audit_sha256"][str((source/"audit.json").relative_to(S.ROOT))]=S.sha(source/"audit.json");summary["updated_utc"]=S.utc()
    S.save(out/"summary.json",summary);manifest(out)


def manifest(out):
    S.save(out/"manifest.json",{"source_sha256":S.sha(__file__),"base_export_sha256":S.sha(S.HERE/"export_fork_angle.py"),"files":{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}})


if __name__=="__main__":main()
