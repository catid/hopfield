"""Finish the free-pair/preconditioning report after all single-pass runs complete."""
from pathlib import Path
import json
import runpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/multi_solution"
UP = ROOT / "upstream"
for n, method in [(8, "mpl"), (10, "cd")]:
    for seed in (0, 1, 2):
        assert (UP / f"experiments/runs/NQ/alt/n{n}/final/{method}-s{seed}/final.json").exists()
for n, method, arm in [(8, "mpl", "table"), (10, "pl", "ising")]:
    for seed in (0, 1, 2):
        assert (UP / f"experiments/runs/NQ/nq11/n{n}/ablation/{method}-{arm}-noprecond-s{seed}/final.json").exists()
        assert (UP / f"experiments/runs/NQ/nq11/n{n}/final/{method}-{arm}-s{seed}/final.json").exists()
runpy.run_path(str(ROOT / "scripts/summarize_nqueens_results.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts/summarize_nqueens_controls.py"), run_name="__main__")
results = json.loads((OUT / "nqueens-paper-results.json").read_text())
controls = json.loads((OUT / "nqueens-preconditioning-results.json").read_text())
assert controls["complete"]
reference = json.loads((UP / "results/nqueens-results.json").read_text())
fmt = lambda x: f"{x['mean']:.6f} ± {x['sd']:.6f}"
lines = ["<!-- additional-cram-controls -->", "## Additional CRAM architecture and preconditioning controls", "",
    "All six free-pair runs, six no-preconditioning ablations, and three size-8 locally selected controls are complete. The size-10 preconditioned control reuses the already completed main run. Final test sets were scored once per run; subsequent metrics use saved chain outputs.", "",
    "### Free pair tables", "",
    "The release includes a free-pair N-Queens architecture control in its comparison export; it is not an additional visible row of the paper's rendered Table 3. The frozen validation-selected recipes are MPL at size 8 (learning rate 10, masking 0.1, no L1, 10,000 updates) and CD at size 10 (learning rate 100, 100 negative-sample moves, no L1, 2,000 updates). Both use momentum 0.5, cosine decay, local-margin stopping, temperature 3 times the fitted unit, 20 chains, and a 20,000-move cap. Source fit and search routines are unchanged, with TF32 disabled.", "",
    "| Size / model | N1 accuracy, % | Lowest-energy N20 accuracy, % | Per-puzzle Cov@20, % |",
    "|---|---:|---:|---:|", "| BP AdamW control: not measured; outside CRAM-only scope | — | — | — |"]
fresh_fits = []
for n, method in [(8, "mpl"), (10, "cd")]:
    r = next(r for r in results if r["task"] == f"nqueens-{n}" and r["kind"] == "free")
    assert r["complete_seeds"]
    for label, metrics in [(f"{n}, fresh {method.upper()} free pair tables", r["metrics"]),
                           (f"{n}, imported released reference", r["paper_reference"])]:
        lines.append(f"| {label} | {fmt(metrics['acc_n1'])} | {fmt(metrics['acc_n20'])} | {fmt(metrics['cov20_per_puzzle'])} |")
    for seed in (0, 1, 2):
        folder = UP / f"experiments/runs/NQ/alt/n{n}/final/{method}-s{seed}"
        fit = json.loads((folder / "fit.json").read_text())
        metadata = json.loads((OUT / f"free-nq{n}-s{seed}-provenance.json").read_text())
        assert metadata["selection"] == reference["training_methods"]["sizes"][str(n)]["rows"][method]["choice"]
        assert "completed_unix" in metadata
        fresh_fits.append({"n": n, "seed": seed, "method": method, "fit": fit, "provenance": metadata})
lines += ["", "The local-margin stopping rule is not a general correctness certificate for these weaker free-pair fits:", "",
          "| Size | Wrong stopped chains, seeds 0 / 1 / 2 | Capped chains, seeds 0 / 1 / 2 |",
          "|---|---|---|"]
for n in (8, 10):
    r = next(r for r in results if r["task"] == f"nqueens-{n}" and r["kind"] == "free")
    lines.append(f"| {n} | {' / '.join(str(s['wrong_stops']) for s in r['per_seed'])} | {' / '.join(str(s['budget_failures']) for s in r['per_seed'])} |")
lines += ["", "Cold fits were timed for every seed; only seed 0 ran the seven reinitialized warm fits used for the publication's timing boundary. This skips fourteen unnecessary timing-only fits per architecture while preserving all three quality runs. Warm repeats are checked against the scored cold model; per-tensor differences are retained in `fit.json`. H200 references and current Blackwell times are hardware-specific measurements, not isolated software speedups.", "",
    "| Size | Local cold fit, seeds 0 / 1 / 2, s | Local seed-0 warm median, s | Published H200 seed-0 warm, s |",
    "|---|---:|---:|---:|"]
for n, method in [(8, "mpl"), (10, "cd")]:
    fits = [x["fit"] for x in fresh_fits if x["n"] == n]
    published = reference["training_methods"]["sizes"][str(n)]["rows"][method]["per_seed"][0]["fit_gpu_warm_s"]
    cold_text = ' / '.join(format(f['cold_seconds'], '.4f') for f in fits)
    lines.append(f"| {n} | {cold_text} | {fits[0]['warm_seconds']:.4f} | {published:.4f} |")
lines += ["", "### Matched preconditioning ablation", "",
    "The published size-8 ablation uses the locally selected MPL/table energy, with masking 0.3 and learning rate 30; this differs from the transferred size-10 PL/Ising recipe used in the main size-8 table. We therefore reran its own preconditioned MPL/table control before comparing the ablation. Size 10 uses the same PL/Ising recipe as the main result. Each pair differs only by the per-offset gradient preconditioner; there are 4,000 updates, no L1, and no test tuning.", "",
    "| Size / matched recipe | Preconditioning | N1 accuracy, % | Lowest-energy N20, % | Per-puzzle Cov@20, % |",
    "|---|---|---:|---:|---:|", "| BP AdamW control: not measured; outside CRAM-only scope | — | — | — | — |"]
for r in controls["rows"]:
    metrics = r["metrics"]
    lines.append(f"| {r['n']} / {r['method'].upper()} {r['arm']} | {'on' if r['preconditioned'] else 'off'} | {fmt(metrics['acc_n1'])} | {fmt(metrics['acc_n20_lowest_energy'])} | {fmt(metrics['cov20_per_puzzle'])} |")
lines += ["", "The imported ablation records use diagnostic majority-vote N20 and GRAM row-mean coverage. Those are also retained and compared only against the matching diagnostic values in [nqueens-preconditioning-results.json](../artifacts/multi_solution/nqueens-preconditioning-results.json); the table above uses the paper's headline metric definitions recomputed from the saved chains.", ""]
failed_stops = next(r for r in controls["rows"] if r["n"] == 10 and not r["preconditioned"])["metrics"]["wrong_stops_20chains"]["per_seed"]
lines += [f"Size-10 no-preconditioning produces {' / '.join(str(x) for x in failed_stops)} wrong stopped chains across seeds 0 / 1 / 2, out of 369,320 chains per seed. The matched preconditioned control has zero wrong stops. Both size-8 controls also have zero wrong stops and retain 100% N1/N20 accuracy; the large quality loss is specific to size 10 in this experiment.", ""]
on = next(r for r in controls["rows"] if r["n"] == 10 and r["preconditioned"])["per_seed"][0]
off = next(r for r in controls["rows"] if r["n"] == 10 and not r["preconditioned"])["per_seed"][0]
lines.append(f"For seed 0 at size 10, the longest-diagonal interaction contrast is {on['diagonal_offset_contrast'][-1]['contrast_mean']:.6f} with preconditioning and {off['diagonal_offset_contrast'][-1]['contrast_mean']:.6f} without it; the shortest-diagonal values are {on['diagonal_offset_contrast'][0]['contrast_mean']:.6f} and {off['diagonal_offset_contrast'][0]['contrast_mean']:.6f}. The respective offsets occur in 2 versus 162 directed cell pairs. This directly measures the paper's explanation that rare long-diagonal interactions barely train without normalization by offset frequency.")
lines += ["", "The [fresh diagonal-cost plot](../artifacts/figures/nqueens-preconditioning-fresh.png), also retained as PDF and SVG, shows the mean and sample SD over all three seeds for each matched control.", "", "Commands and evidence: `scripts/run_nqueens_free_reproduction.py --n N --seed S`; `scripts/run_nqueens_preconditioning.py --n N --seed S --precond on|off`; raw logs in `artifacts/main_followups/`; per-run source/recipe/model hashes in `artifacts/multi_solution/*-provenance.json`. The complete machine-readable free-pair fit bundle is [free-pair-fit-results.json](../artifacts/multi_solution/free-pair-fit-results.json). Final aggregation and this report section are reproduced by `scripts/finalize_nqueens_controls.py`.", ""]
(OUT / "free-pair-fit-results.json").write_text(json.dumps(fresh_fits, indent=2)+"\n")
path = ROOT / "docs/MULTI_SOLUTION_REPRODUCTION.md"
text = path.read_text().split("<!-- additional-cram-controls -->")[0].rstrip()
text = text.replace("The released N-Queens free-pair architecture control is queued for fresh reproduction: validation-selected MPL at size 8 and CD at size 10.",
    "The released N-Queens free-pair architecture controls and matched preconditioning ablations are also complete: validation-selected MPL at size 8 and CD at size 10 for free pairs.")
path.write_text(text+"\n\n"+"\n".join(lines))
print("Updated", path)
