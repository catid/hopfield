"""Consolidate completed CRAM audit receipts; never launch fits or inference.

Run after the last collection, plots, reports, and independent audits are final.
This verifies completion gates and recorded evidence hashes, not new science.
An incomplete study produces no completion receipt.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final_reproduction_audit.json"
EVIDENCE = {}
EXPECTED_KS = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 100, 300, 1000, 3000, 10000]
MULTI_CHECKS = (
    "all_main_and_additional_nqueens_seed_cells_complete",
    "all_free_pair_timing_repeat_counts_match_frozen_protocol",
    "every_free_pair_warm_model_bitwise_equal_to_scored_cold_model",
    "all_free_pair_model_record_and_final_hashes_verified",
    "all_report_links_resolve",
    "upstream_tracked_tree_clean",
)


def digest(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Evidence escapes workspace: {relative}")
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    EVIDENCE[str(path.relative_to(ROOT))] = value
    return value


def read(relative):
    digest(relative)
    return json.loads((ROOT / relative).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "Missing evidence hash mapping")
    for path, expected in mapping.items():
        require(digest(path) == expected, f"Changed evidence: {path}")


def main():
    EVIDENCE.clear()
    de = read("artifacts/data_efficiency/collection.json")
    require(de["figure_complete"] and de["measured_jobs"] == de["expected_jobs"] == 102,
            f"Data efficiency incomplete: {de['measured_jobs']}/102 measured")
    require(de["missing_jobs"] == de["degenerate_jobs"] == 0 and de["status_counts"] == {"complete": 102},
            "Data efficiency includes missing, degenerate or invalid rows")
    require(len(de["rows"]) == 102 and len(de["points"]) == 17, "Wrong data-efficiency dimensions")
    require(sorted(point["k"] for point in de["points"]) == EXPECTED_KS,
            "Missing, duplicate, or unexpected data-efficiency k point")
    for point in de["points"]:
        rows = sorted((r for r in de["rows"] if r["k"] == point["k"]), key=lambda r: r["seed"])
        require([r["seed"] for r in rows] == list(range(6)), "Missing or duplicate seed")
        require(all(r["status"] == "complete" and r["test_n"] == r["summary"]["n"] == 10000 for r in rows), "Invalid test cohort")
        values = [100 * r["summary"]["solved"] / r["summary"]["n"] for r in rows]
        require(values == point["accuracy_pct_per_seed"], "Per-seed accuracy mismatch")
        require(statistics.median(values) == point["accuracy_pct_median"] and
                min(values) == point["accuracy_pct_min"] and max(values) == point["accuracy_pct_max"],
                "Data-efficiency aggregate mismatch")
    figure = read("artifacts/data_efficiency/figure3b.json")
    require(figure["status"] == "complete" and figure["figure_complete"], "Final plot incomplete")
    require(figure["source_sha256"] == digest("artifacts/data_efficiency/collection.json"), "Final plot is stale")
    dea = read("artifacts/data_efficiency/final-audit.json")
    require(dea["status"] == "validated", "Independent data-efficiency audit did not pass")
    de_audited_paths = {(ROOT / path).resolve() for path in dea["input_sha256"]}
    require(all((ROOT / f"artifacts/data_efficiency/{name}").resolve() in de_audited_paths
                for name in ("collection.json", "figure3b.json", "figure3b.png", "figure3b.pdf")),
            "Independent data-efficiency audit omits final collection or plot evidence")
    verify_hashes(dea["input_sha256"])
    for seed in (3, 4, 5):
        resumed = read(f"artifacts/orchestration_review/resumed-data-efficiency-seed{seed}-audit.json")
        require(resumed["status"] == "validated", "Continuation audit did not pass")
        require(resumed["seed"] == seed and resumed["k"] == 1, "Continuation audit seed/k mismatch")
        verify_hashes(resumed["sha256"])

    objective = read("artifacts/objective_studies/final-validation.json")
    require(objective["status"] == "passed" and objective["managed_jobs"] + objective["external_jobs"] == 84,
            "Objective fits incomplete")
    require(objective["d2_complete_schedules"] == 9 and objective["figure4_complete_curves"] == 100 and
            objective["figure4_complete_probes"] == 9740 and objective["checkpoint_comparisons"] == 228,
            "Supplemental schedules or clock checks incomplete")
    verify_hashes(objective["input_sha256"])
    require(digest(objective["finalization_script"]) == objective["finalization_script_sha256"],
            "Objective finalizer differs from recorded source")
    controls = read("artifacts/final_controls_audit.json")
    require(controls["status"] == "validated", "Final controls audit did not pass")
    verify_hashes(controls["evidence_sha256"])
    maze_dev = read("artifacts/maze_hard_development/validation.json")
    require(maze_dev["status"] == "validated" and maze_dev["complete_seeds"] == [0, 1, 2],
            "Maze-Hard frozen-recipe development controls incomplete")
    require(sorted(r["seed"] for r in maze_dev["runs"]) == [0, 1, 2] and
            all(r["status"] == "complete" and r["n"] == 100 and r["updates"] == 4000 for r in maze_dev["runs"]),
            "Maze-Hard development schedule or cohort mismatch")
    require(maze_dev["test_data_used"] is False and maze_dev["tuning"] is False,
            "Maze-Hard development check used test data or tuning")
    verify_hashes(maze_dev["evidence_sha256"])
    maze_review = read("artifacts/orchestration_review/maze-hard-development-independent-review.json")
    require(maze_review["status"] == "validated" and not maze_review["concrete_blockers"],
            "Independent Maze-Hard development review did not pass")
    require(maze_review["validation_receipt_sha256"] == digest("artifacts/maze_hard_development/validation.json"),
            "Independent Maze-Hard development review is stale")
    main_audit = read("artifacts/completed_controls_audit.json")
    require(len(main_audit["checks"]) == 12 and all(c["status"] == "validated" for c in main_audit["checks"]),
            "Main-result audit did not pass")
    require(not main_audit["report_changed_during_audit"] and not main_audit["editorial_findings"],
            "Main-result audit has unresolved report findings")
    verify_hashes(main_audit["evidence_sha256"])
    multi = read("artifacts/multi_solution/final-validation.json")
    require(all(multi.get(key) is True for key in MULTI_CHECKS), "Multi-solution audit omits or fails a required check")
    verify_hashes(multi["summary_sha256"])
    verify_hashes(read("artifacts/main-comparison.json")["source_sha256"])
    resources = read("artifacts/orchestration_review/final-resource-audit.json")
    require(resources["status"] == "validated" and resources["all_gpu_queries_succeeded"] is True and
            resources["all_gpu_compute_processes_absent"] is True and not resources["remaining_task_processes"],
            "Final task-resource cleanup audit incomplete")
    verify_hashes(resources["evidence_sha256"])

    reference = read("artifacts/reference-tests.json")
    require(reference["tests"] == 27 and reference["failures"] == reference["errors"] == reference["skipped"] == 0,
            "Independent test receipt did not pass")
    verify_hashes(reference["source_sha256"])
    upstream = read("artifacts/upstream_test_summary.json")
    require(upstream["combined_passed"] == 91 and upstream["combined_failed"] == 0, "Upstream test receipt did not pass")
    conversion = read("references/review/validation.json")
    for key, count in (("display_equation_tags", 15), ("tables", 8), ("figures", 6), ("bibliography_entries", 66)):
        require(conversion[key] == [str(i) for i in range(1, count + 1)], f"Paper structure mismatch: {key}")
    require(not any(conversion[k] for k in ("missing_internal_anchors", "duplicate_anchors", "missing_local_assets", "unconverted_markup")),
            "Paper conversion has structural errors")
    math = read("references/review/math_validation.json")
    require(math["count"] == 342 and not math["errors"], "Paper math validation did not pass")

    links = 0
    for path in [ROOT / "README.md", ROOT / "REPRODUCTION.md", *sorted((ROOT / "docs").glob("*.md"))]:
        digest(str(path.relative_to(ROOT)))
        for match in re.finditer(r"\]\(([^)]+)\)", path.read_text()):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(("https:", "http:", "mailto:")):
                continue
            # The report may link to the receipt this successful invocation creates.
            resolved = (path.parent / target).resolve()
            require(resolved == OUT or resolved.exists(), f"Broken link in {path.name}: {target}")
            links += 1
    scope = read("configs/study_scope.json")
    pin = read("upstream_pin.json")
    digest("references/CONVERSION_REVIEW.md")
    digest(str(Path(__file__).relative_to(ROOT)))
    result = {
        "status": "validated_complete_with_declared_limits",
        "utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope["confirmed_by_user"],
        "source_commit": pin["commit"],
        "completed": {"main_audit_families": 12, "table7_runs": 84, "d2_runs": 9,
                      "figure4_curves": 100, "figure4_probes": 9740, "data_efficiency_runs": 102,
                      "data_efficiency_six_seed_points": 17, "resumed_search_audits": 3,
                      "maze_hard_frozen_recipe_development_controls": 3},
        "saved_tests": {"independent": 27, "upstream": 91},
        "report_links_checked": links,
        "new_training_or_inference": False,
        "limits": scope["known_protocol_limits"] + [
            "Saved Sudoku outputs retain correctness flags, not final boards; these audits cannot independently reconstruct their final board validity."
        ],
        "input_sha256": EVIDENCE,
        "reproduction_command": "python3 scripts/finalize_reproduction.py",
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(OUT)
    print(json.dumps({"status": result["status"], "hashed_files": len(EVIDENCE), "receipt": str(OUT)}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "not_finalized", "reason": str(exc)}))
        sys.exit(1)
