"""Independent saved-board audit of the frozen paired eta experiment."""
import csv
import json
from pathlib import Path

import numpy as np

import audit as A
import study as S


def main():
    dest = S.OUT / "eta"
    pp = S.HERE / "eta-protocol.json"
    protocol = json.loads(pp.read_text())
    evidence = {}

    def verify(path, expected=None):
        path = Path(path)
        path = path if path.is_absolute() else S.ROOT / path
        actual = S.sha(path)
        if expected is not None:
            assert actual == expected, path
        evidence[str(path.relative_to(S.ROOT))] = actual

    verify(pp)
    verify(__file__)
    verify(A.__file__)
    verify(S.__file__, protocol["core_script_sha256"])
    verify(S.HERE / "eta_study.py", protocol["execution_script_sha256"])
    verify(dest / "preflight.json")
    preflight = json.loads((dest / "preflight.json").read_text())
    assert preflight["status"] == "passed"
    assert preflight["protocol_sha256"] == S.sha(pp)
    assert preflight["execution_sha256"] == protocol["execution_script_sha256"]
    for path, sha in protocol["source_sha256"].items():
        verify(path, sha)
    for model in protocol["models"].values():
        verify(model["path"], model["sha256"])

    records, saved, datasets, diagnostics = [], {}, {}, []
    for job in protocol["jobs"]:
        run = dest / "runs" / job["id"]
        record = json.loads((run / "result.json").read_text())
        assert record["status"] == "complete" and record["job"] == job
        assert record["protocol_sha256"] == S.sha(pp)
        assert record["script_sha256"] == protocol["execution_script_sha256"]
        assert record["core_sha256"] == protocol["core_script_sha256"]
        assert record["checkpoint_sha256"] == protocol["models"][str(job["seed"])]["sha256"]
        verify(run / "result.json")
        dataset = next(d for d in protocol["datasets"] if d["length"] == job["length"] and d["loop_percent"] == job["loop_percent"])
        verify(dataset["path"], dataset["sha256"])
        assert record["dataset_sha256"] == dataset["sha256"]
        if dataset["path"] not in datasets:
            datasets[dataset["path"]] = np.load(S.ROOT / dataset["path"])
        data = datasets[dataset["path"]]
        for eta in ("0.2", "0.5"):
            arm = record["arms"][eta]
            verify(run / f"eta{eta}.npz", arm["output_sha256"])
            output = np.load(run / f"eta{eta}.npz")
            valid = []
            for tok, pred, expected_distance in zip(data["tokens"], output["predictions"], data["shortest_edges"]):
                ok, distance = A.independent(tok, pred)
                assert distance == expected_distance
                valid.append(ok)
            valid = np.asarray(valid)
            success = valid & output["stopped"] & output["hidden_good"]
            prefix = success & (output["answer_steps"] <= 2048)
            assert np.array_equal(valid, output["shortest_path"])
            assert np.array_equal(success, output["success"])
            assert np.array_equal(prefix, output["success2048"])
            assert int(success.sum()) == arm["solved"]
            assert int(prefix.sum()) == arm["solved2048"]
            assert int((output["stopped"] & ~valid).sum()) == arm["wrong_stops"]
            assert int((~output["stopped"]).sum()) == arm["answer_budget_failures"]
            assert int((~output["hidden_good"]).sum()) == arm["hidden_failures"]
            steps = output["answer_steps"]
            assert np.all((steps >= 16) & (steps <= 8192) & (steps % 16 == 0))
            assert np.all(steps[~output["stopped"]] == 8192)
            assert np.isclose(steps.mean(), arm["answer_steps_mean"])
            saved[(job["length"], job["loop_percent"], job["seed"], eta)] = {
                "success": success, "success2048": prefix, "steps": steps,
            }
            records.append(dict(job=job, eta=float(eta), n=len(success), **arm,
                                hidden_seconds_shared=record["hidden_seconds_shared"],
                                wall_seconds_paired_job=record["wall_seconds"],
                                gpu=record["gpu"], host=record["host"]))
        if record["diagnostics_sha256"]:
            verify(run / "diagnostics.json", record["diagnostics_sha256"])
            for eta, diag in json.loads((run / "diagnostics.json").read_text()).items():
                assert diag["replay_native_parity"]
                energy = np.asarray([h["energy"] for h in diag["history"]])
                residual = np.asarray([h["residual"] for h in diag["history"][1:]])
                changes = np.diff(energy, axis=0)
                tolerance = 1e-6 * np.maximum(1, np.abs(energy[:-1]))
                diagnostics.append({
                    "job": job["id"], "eta": float(eta), "n_unique_traces": energy.shape[1],
                    "replay_native_parity": True, "recorded_energy_transitions": changes.size,
                    "positive_energy_transitions": int((changes > 0).sum()),
                    "energy_increases_over_relative1e6_tolerance": int((changes > tolerance).sum()),
                    "maximum_energy_increase": float(np.maximum(changes, 0).max()),
                    "maximum_energy_relative_increase": float(np.maximum(changes / np.maximum(1, np.abs(energy[:-1])), 0).max()),
                    "maximum_final_free_logit_residual": float(residual[-1].max()),
                    "two_cycle_flags": int(sum(diag["two_cycle_detected"])),
                    "recorded_updates": [h["update"] for h in diag["history"]],
                })

    rng = np.random.default_rng(2026111421)
    summaries, maze_rows, cells = [], [], []
    for length in (60, 90):
        strata, joint_by_eta, prefix_by_eta = [], {e: [] for e in ("0.2", "0.5")}, {e: [] for e in ("0.2", "0.5")}
        for rate in (0, 5, 20):
            steps = {e: np.stack([saved[(length, rate, s, e)]["steps"] for s in range(3)]).mean(0) for e in ("0.2", "0.5")}
            joint = {e: np.stack([saved[(length, rate, s, e)]["success"] for s in range(3)]).all(0) for e in ("0.2", "0.5")}
            prefix = {e: np.stack([saved[(length, rate, s, e)]["success2048"] for s in range(3)]).all(0) for e in ("0.2", "0.5")}
            strata.append(steps)
            for eta in ("0.2", "0.5"):
                joint_by_eta[eta].append(joint[eta])
                prefix_by_eta[eta].append(prefix[eta])
            cells.append({"length": length, "loop_percent": rate, "n_unique": len(steps["0.2"]),
                          "median_steps_eta02": float(np.median(steps["0.2"])),
                          "median_steps_eta05": float(np.median(steps["0.5"])),
                          "ratio_of_medians": float(np.median(steps["0.2"]) / np.median(steps["0.5"])),
                          "joint_solved_eta02": int(joint["0.2"].sum()), "joint_solved_eta05": int(joint["0.5"].sum()),
                          "joint_solved2048_eta02": int(prefix["0.2"].sum()), "joint_solved2048_eta05": int(prefix["0.5"].sum())})
            for i in range(len(steps["0.2"])):
                maze_rows.append({"length": length, "loop_percent": rate, "maze_index": i,
                                  "mean_model_steps_eta02": float(steps["0.2"][i]), "mean_model_steps_eta05": float(steps["0.5"][i]),
                                  "joint_success8192_eta02": int(joint["0.2"][i]), "joint_success8192_eta05": int(joint["0.5"][i]),
                                  "joint_success2048_eta02": int(prefix["0.2"][i]), "joint_success2048_eta05": int(prefix["0.5"][i])})
        boot = {e: [] for e in ("0.2", "0.5")}
        for stratum in strata:
            indices = rng.integers(len(stratum["0.2"]), size=(10000, len(stratum["0.2"])))
            for eta in boot:
                boot[eta].append(stratum[eta][indices])
        ratios = np.median(np.concatenate(boot["0.2"], axis=1), axis=1) / np.median(np.concatenate(boot["0.5"], axis=1), axis=1)
        ci = np.quantile(ratios, [.025, .975]).tolist()
        medians = {e: float(np.median(np.concatenate([s[e] for s in strata]))) for e in boot}
        joint = {e: np.concatenate(joint_by_eta[e]) for e in boot}
        prefix = {e: np.concatenate(prefix_by_eta[e]) for e in boot}
        lost = joint["0.2"] & ~joint["0.5"]
        gained = ~joint["0.2"] & joint["0.5"]
        loss_bounds = A.bounds(int(lost.sum()), len(lost))
        speed_decision = "supported" if ci[0] >= 2 else "falsified" if ci[1] < 2 else "inconclusive"
        accuracy_decision = "falsified" if lost.any() else "supported" if loss_bounds[1] <= .01 else "inconclusive"
        selected = [r for r in records if r["job"]["length"] == length]
        timing = {e: sum(r["answer_seconds"] for r in selected if r["eta"] == float(e)) for e in boot}
        whole_timing = {e: sum(r["hidden_seconds_shared"] + r["answer_seconds"] for r in selected if r["eta"] == float(e)) for e in boot}
        summaries.append({"length": length, "n_unique": len(lost), "model_repeats": 3,
                          "median_steps_eta02": medians["0.2"], "median_steps_eta05": medians["0.5"],
                          "ratio_of_medians": medians["0.2"] / medians["0.5"],
                          "paired_stratified_bootstrap95_ratio": ci, "speed_threshold2x_decision": speed_decision,
                          "joint_solved_eta02": int(joint["0.2"].sum()), "joint_solved_eta05": int(joint["0.5"].sum()),
                          "lost_joint_solves": int(lost.sum()), "gained_joint_solves": int(gained.sum()),
                          "one_sided95_loss_upper_pp": 100 * loss_bounds[1], "accuracy_decision": accuracy_decision,
                          "combined_hypothesis": "falsified" if "falsified" in (speed_decision, accuracy_decision) else "supported" if speed_decision == accuracy_decision == "supported" else "inconclusive",
                          "joint_solved2048_eta02": int(prefix["0.2"].sum()), "joint_solved2048_eta05": int(prefix["0.5"].sum()),
                          "answer_seconds_eta02": timing["0.2"], "answer_seconds_eta05": timing["0.5"],
                          "native_answer_runtime_ratio": timing["0.2"] / timing["0.5"],
                          "hidden_plus_answer_runtime_ratio": whole_timing["0.2"] / whole_timing["0.5"],
                          "censored_model_maze_arms": sum(r["answer_budget_failures"] for r in selected)})

    with (dest / "per-maze.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(maze_rows[0]))
        writer.writeheader()
        writer.writerows(maze_rows)
    verify(dest / "per-maze.csv")
    result = {"status": "validated", "utc": S.utc(), "campaign": "eta", "jobs": len(protocol["jobs"]),
              "evaluated_arms": len(records), "saved_board_checks": sum(r["n"] for r in records),
              "protocol_sha256": S.sha(pp), "records": records, "summaries": summaries, "cells": cells,
              "diagnostics": diagnostics, "evidence_sha256": evidence, "hashed_files": len(evidence),
              "new_inference": False,
              "limits": ["Step-ratio intervals resample unique mazes within the three frozen loop-rate strata, preserving all model repeats and both eta arms.",
                         "Runtime ratios compare summed synchronized native answer times on the same GPU within each job; each includes identical shared hidden preparation only in the separately reported full ratio.",
                         "No multiplicity adjustment: each per-size claim uses its declared 95 percent interval.",
                         "Energy changes are sampled every 16 updates on 48 unique model-seed0 mazes per eta; positive changes below relative 1e-6 are reported as numerical scale diagnostics, not a proof of monotonic descent.",
                         "The stopping test is discrete flip stability; final continuous-logit residual need not vanish."]}
    S.save(dest / "audit.json", result)
    S.emit("eta_audit_complete", jobs=result["jobs"], saved_board_checks=result["saved_board_checks"], summaries=summaries)


if __name__ == "__main__":
    main()
