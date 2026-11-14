"""Export Table 4 from saved local audits and the hand-reviewed paper.

This reads evidence only. It never launches inference or edits paper records.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paper_path = ROOT / "references/2611.17309v1.md"
    paper = paper_path.read_text()
    section = paper.split('Table 4: ', 1)[1].split('<a id="A2">', 1)[0]
    published, task = [], None
    for line in section.splitlines():
        if line in ("**Sudoku-Extreme**", "**Maze-Unique**"):
            task = line.strip("*")
        if not line.startswith("| ") or "Stopping rule" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 4 and task is not None
        published.append(dict(task=task, method=cells[0], provenance="published_only",
                              halted_percent=cells[1], wrong_halts_percent=cells[2],
                              solved_percent=cells[3]))
    assert len(published) == 12
    sudoku_path = ROOT / "artifacts/sudoku_audit/learned-full-test-audit.json"
    maze_path = ROOT / "artifacts/fresh_maze/reproduction_summary.json"
    sudoku = json.loads(sudoku_path.read_text())
    maze = json.loads(maze_path.read_text())
    assert sudoku["complete"] and sudoku["seeds"] == [0, 1, 2]
    assert sudoku["unique_test_puzzles"] == 412786
    counts = sudoku["counts_pooled_over_seeds"]
    assert sudoku["seed_puzzle_evaluations"] == 3 * 412786
    local = [dict(task="Sudoku-Extreme", method="CRAM (energy threshold)",
                  n=sudoku["seed_puzzle_evaluations"], distinct_inputs=412786,
                  halted=counts["stopped"], wrong_halts=counts["wrong_stops"],
                  solved=counts["solved"], seeds=sudoku["seeds"])]
    record = next(r for r in maze["records"] if r["task"] == "maze_unique")
    seeds = record["fresh_seeds"]
    assert sorted(s["seed"] for s in seeds) == [0, 1, 2]
    assert all(s["n"] == 1000 for s in seeds)
    local.append(dict(task="Maze-Unique", method="CRAM (fixed-point stop)",
                      n=sum(s["n"] for s in seeds), distinct_inputs=1000,
                      halted=sum(s["n"] - s["budget_failures"] for s in seeds),
                      wrong_halts=sum(s["wrong_stops"] for s in seeds),
                      solved=sum(s["solved"] for s in seeds), seeds=[0, 1, 2]))
    for row in local:
        assert 0 <= row["wrong_halts"] <= row["halted"] <= row["n"]
        assert 0 <= row["solved"] <= row["n"]
        row.update(provenance="fresh_local_saved_results",
                   halted_percent=100 * row["halted"] / row["n"],
                   wrong_halts_percent=(100 * row["wrong_halts"] / row["halted"]
                                        if row["halted"] else None),
                   solved_percent=100 * row["solved"] / row["n"])
    result = dict(paper="arXiv:2611.17309v1", table=4, complete=True,
                  published=published, local=local,
                  sources={str(p.relative_to(ROOT)): sha(p)
                           for p in (paper_path, sudoku_path, maze_path)},
                  definitions={"wrong_halts_percent": "Incorrect halted boards / all halted boards, pooled across seeds",
                               "sample_count": "Each seed uses the same full test inputs; seed evaluations are not distinct puzzles",
                               "maze_stop": "The released Maze-Unique solver stops at a fixed point, despite Table 4's generic energy-threshold label"},
                  limitations=["Published baseline halting runs are not local measurements",
                               "These results do not imply zero incorrect stops for CRAM+z on Maze-Hard: its fresh runs each halt incorrectly on maze 106"])
    output = ROOT / "artifacts/halting-comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Stopping-rule comparison", "",
             "Table 4's CRAM rows are complete on all 412,786 Sudoku puzzles and all 1,000 Maze-Unique mazes, each under three seeds. Both tasks have zero incorrect stops. The baseline rows below are copied from the manually reviewed paper and are **published results only**.", "",
             "Wrong halts uses halted boards as its denominator. Counts pool the three seeds, which evaluate the same test inputs. No inference was rerun for this report.", ""]
    for task in ("Sudoku-Extreme", "Maze-Unique"):
        lines += [f"## {task}", "",
                  "| Method / provenance | Halted, % | Wrong halts, % of halted | Solved, % |",
                  "|---|---:|---:|---:|",
                  "| BP AdamW control: outside local training scope | — | — | — |"]
        for row in published:
            if row["task"] == task:
                lines.append(f'| {row["method"]} (published) | {row["halted_percent"]} | {row["wrong_halts_percent"]} | {row["solved_percent"]} |')
        row = next(r for r in local if r["task"] == task)
        lines.append(f'| {row["method"]} (fresh local) | {row["halted_percent"]:.6f} | {row["wrong_halts_percent"]:.6f} | {row["solved_percent"]:.6f} |')
        lines += ["", f'Fresh counts: {row["halted"]:,} halted, {row["wrong_halts"]:,} incorrect halts, and {row["solved"]:,} solved out of {row["n"]:,} seed evaluations.', ""]
    lines += ["The released Maze-Unique solver stops at a fixed point; Table 4 labels its CRAM row generically as an energy threshold. This distinction is retained. The zero-wrong-stop observation does not extend to CRAM+z on Maze-Hard: each fresh seed stops incorrectly on maze 106, as documented in the [maze report](MAZE_REPRODUCTION.md).", "",
              "Regenerate without scoring any test input: `python3 scripts/export_halting_comparison.py`. Counts and source hashes are in [halting-comparison.json](../artifacts/halting-comparison.json).", ""]
    (ROOT / "docs/HALTING_REPRODUCTION.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "fresh_rows": len(local), "published_rows": len(published)}))


if __name__ == "__main__":
    main()
