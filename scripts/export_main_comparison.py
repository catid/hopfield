"""Export Tables 2 and 8 comparisons using saved measurements; no rescoring."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}
    def read(relative, json_value=True):
        path = ROOT / relative
        data = path.read_bytes()
        sources[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data) if json_value else data.decode()
    paper = read("references/2611.17309v1.md", False)
    sudoku = read("artifacts/sudoku_audit/learned-full-test-audit.json")
    hand = read("artifacts/sudoku_audit/hand-full-test-audit.json")
    maze = read("artifacts/fresh_maze/reproduction_summary.json")
    weights = read("artifacts/sudoku_fresh_vs_released.json")
    timing_text = read("artifacts/sudoku_gpu_timing.log", False)
    timing = next(json.loads(line) for line in reversed(timing_text.splitlines()) if line.startswith("{"))
    assert timing["split"] == 0 and timing["bitwise_equal_to_saved"] and len(timing["warm_s"]) == 7
    assert sudoku["complete"] and hand["complete"]
    rows = []
    for table, section, initial_task in (
        (2, paper.split('Table 2: ', 1)[1].split('<a id="S4">', 1)[0], None),
        (8, paper.split('Table 8: ', 1)[1].split('##### Data.', 1)[0], "Maze-Hard"),
    ):
        task = initial_task
        for line in section.splitlines():
            if line in ("**Sudoku-Extreme**", "**Maze-Unique**"):
                task = line.strip("*")
            if not line.startswith("| ") or "Model" in line:
                continue
            fields = [cell.strip() for cell in line.strip().strip("|").split("|")]
            assert len(fields) == (5 if table == 2 else 4) and task
            link = re.search(r"\[([^\]]+)\]\(", fields[0])
            model = link.group(1).rstrip("§") if link else None
            if model is None:
                model = "CRAM+z" if table == 8 else ("Hand-set CRAM" if "hand-set" in fields[0] else "CRAM")
            clean = lambda value: re.sub(r"<sup>.*?</sup>", "", value).replace("**", "").replace("$", "")
            rows.append(dict(table=table, task=task, model=model, provenance="published_only",
                             accuracy_display=clean(fields[1]), params_display=clean(fields[2]),
                             flops_display=clean(fields[3]), fit_display=clean(fields[4]) if table == 2 else None,
                             raw_paper_cells=fields))
    assert len(rows) == 24
    assert len({(r["task"], r["model"]) for r in rows}) == len(rows)
    for name, audit, params, fit in (("CRAM", sudoku, str(weights[0]["tensor_comparison"]["W"]["nonzero"]), timing["warm_median_s"]),
                                     ("Hand-set CRAM", hand, "1 distinct coefficient", None)):
        rows.append(dict(table=2, task="Sudoku-Extreme", model=name, provenance="fresh_local",
                         accuracy_display=f'{audit["solved_percent"]["mean"]:.6f} ± {audit["solved_percent"]["sd"]:.6f}',
                         params_display=params, flops_display=f'{audit["paper_mean_flops"]["mean"]:.8g}',
                         fit_display=f'{fit:.8f} s' if fit is not None else "none",
                         distinct_test_inputs=412786, training_or_inference_seeds=[0, 1, 2]))
    for item in maze["records"]:
        seed_records = item["fresh_seeds"]
        assert sorted(r["seed"] for r in seed_records) == [0, 1, 2]
        rates = [100 * r["solved"] / r["n"] for r in seed_records]
        hard = item["task"] == "maze_hard"
        is_hand = item["task"] == "hand_maze_unique"
        fit = seed_records[0].get("complete_fit_timing", {}).get("cuda", {}).get("warm_median_s")
        rows.append(dict(table=8 if hard else 2, task="Maze-Hard" if hard else "Maze-Unique",
                         model="CRAM+z" if hard else ("Hand-set CRAM" if is_hand else "CRAM"), provenance="fresh_local",
                         accuracy_display=f'{statistics.fmean(rates):.6f} ± {statistics.stdev(rates):.6f}',
                         params_display="29" if hard else ("3 distinct coefficients" if is_hand else "18"),
                         flops_display=f'{statistics.fmean(r["fc2_flops_mean"] for r in seed_records):.8g}',
                         fit_display=f'{fit:.8f} s' if fit is not None else (None if hard else "none"),
                         distinct_test_inputs=1000, training_or_inference_seeds=[0, 1, 2]))
    result = dict(paper="arXiv:2611.17309v1", rows=rows, source_sha256=sources,
                  timing="Published H200 timings and local Blackwell timings are separate; no locally matched baseline speed ratio",
                  params="Paper convention: nonzero contributing coefficients; local Sudoku row shows seed0 (7716); seeds1 and2 have7750 and7761. Hand models count distinct nonzero coefficients.")
    (ROOT / "artifacts/main-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Main results and published baselines", "",
             "Fresh CRAM measurements are compared with Tables 2 and 8 of the [reviewed paper](../references/2611.17309v1.md). Every baseline value is **published only**. None is a local baseline training, inference or timing measurement.", "",
             "Published fit times use H200 hardware; fresh fit times use RTX PRO 6000 Blackwell. Fresh times below are seed 0's median of seven complete warm fits, matching CRAM's timing boundary in the paper. The paper's ordinary baseline times include cold initialization and compilation. These columns do not establish a matched local training speedup.", ""]
    for task in ("Sudoku-Extreme", "Maze-Unique", "Maze-Hard"):
        timing_column = task != "Maze-Hard"
        lines += [f"## {task}", "",
                  "| Method / provenance | Solved, % ± sample SD | Params | Mean FLOPs/puzzle |" + (" Fit time |" if timing_column else ""),
                  "|---|---:|---:|---:|" + ("---:|" if timing_column else ""),
                  "| BP AdamW control: outside local training scope | — | — | — |" + (" — |" if timing_column else "")]
        for row in rows:
            if row["task"] != task:
                continue
            origin = "published" if row["provenance"] == "published_only" else "fresh local"
            lines.append(f'| {row["model"]} ({origin}) | {row["accuracy_display"]} | {row["params_display"]} | {row["flops_display"]} |' + (f' {row["fit_display"]} |' if timing_column else ""))
        lines.append("")
    lines += ["Sudoku uses all 412,786 test puzzles per seed; both maze tasks use all 1,000 test mazes per seed. All local means and sample standard deviations use seeds 0–2. FLOPs use the paper's arithmetic convention, including unsuccessful searches and all chains. They are logical counts, not measured latency.", "",
              "† The paper substitutes the original authors' GRAM Sudoku and EqR Maze-Unique accuracy where its own baseline reproduction scored lower. Other columns still describe the CRAM paper's runs. Its TRMP lattice footnote reports a separate 98.90% Sudoku model, 16k parameters and 13m28s warm fit; the table uses the main ACT model. Hand-set parameter counts mean distinct coefficient values. Fresh Sudoku has 7,716 / 7,750 / 7,761 contributing coefficients by seed; its row shows seed 0.", "",
              "Maze-Hard baseline accuracies were published by the original authors; counts were reported by the CRAM paper. NCA uses official A* targets, while fresh CRAM+z uses the released BFS target convention. Every local CRAM+z seed has one wrong stop on maze 106; see the [maze audit](MAZE_REPRODUCTION.md).", "",
              "Exact source cells, measured summaries and SHA-256 hashes are in [main-comparison.json](../artifacts/main-comparison.json). Regenerate from saved results with `python3 scripts/export_main_comparison.py`.", ""]
    (ROOT / "docs/MAIN_COMPARISON.md").write_text("\n".join(lines))
    print(json.dumps({"published_rows": 24, "fresh_rows": len(rows) - 24}))


if __name__ == "__main__":
    main()
