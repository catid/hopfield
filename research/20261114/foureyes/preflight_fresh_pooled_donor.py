"""H4bc unrelated full-shape reconstruction, exact proofs and failure gates."""
import itertools
import tempfile
import time
import numpy as np
from fresh_pooled_donor_common import *
from freeze_fresh_pooled_donor import source_paths


def seed_audit(design):
    bound, explicit, bases = {}, set(), set()
    def scan(value, is_seed=False):
        if isinstance(value, dict):
            for key, v in value.items():
                if "seed" in key.lower() and isinstance(v, int):
                    if "base" in key.lower():
                        bases.add(v)
                scan(v, is_seed or "seed" in key.lower())
        elif isinstance(value, list):
            for v in value:
                scan(v, is_seed)
        elif is_seed and type(value) is int:
            explicit.add(value)
    for path in (HERE.parent).rglob("*.json"):
        if path == PROPOSAL:
            continue
        try:
            value = read(path)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(value, dict) or not value.get("frozen_utc"):
            continue
        bound[str(path.relative_to(ROOT))] = sha(path)
        scan(value)
    acquisition = {k: set(range(v, v+N)) for k, v in design["seeds"].items() if k != "solver"}
    solver = {design["seeds"]["solver"]+44*i+d for i in range(N) for d in range(1, 44)}
    all_new = set.union(solver, *acquisition.values())
    assert len(all_new) == len(solver)+sum(map(len, acquisition.values()))
    assert not all_new & explicit
    # Over-approximate recent solver grids by 100000 following integers, and
    # high-valued acquisition bases by 1000 (all recent streams have <=256 banks).
    collisions = [(v, base) for v in all_new for base in bases if base <= v <= base+(100000 if base < 2**31 else 1000)]
    assert not collisions, collisions[:10]
    record = dict(status="passed", frozen_protocol_sha256=bound, explicit_historical_seed_count=len(explicit),
                  conservative_base_intervals=len(bases), solver_base_interval_width=100000, acquisition_base_interval_width=1000,
                  new_LP_seed_count=len(solver), new_acquisition_and_augmentation_seed_count=sum(map(len, acquisition.values())),
                  solver_range=[min(solver), max(solver)], collisions=[],
                  scope="Explicit frozen seed fields plus conservative 100000 solver-base and 1000 acquisition-base intervals, covering recent root/Ripper formulas; not all historical implicit seeds or a PRNG independence proof.")
    save(HERE/"fresh-pooled-donor-seed-audit.json", record)
    return record


def main():
    started = time.monotonic()
    assert not OUT.exists() and not PROTOCOL.exists()
    design = read(PROPOSAL)
    assert design["cohort"]["banks"] == N and design["cohort"]["rows_per_bank"] == 1800
    assert design["budget"]["maximum_new_LPs"] == 43*N == 11008
    assert design["budget"]["noisy_cell_observations_union"] == N*1800*81
    seed_audit(design)
    from fresh_pooled_donor_source import source, audit_source
    from larger_donor_math import counts, independent_counts, validate_labels
    from conflict_mask_math import detector, independent_detector
    from corruption_censor_math import score, independent_score
    from audit_objective_gap import reconstruct
    import fresh_pooled_donor_fit as F
    import audit_fresh_pooled_donor as A
    p = dict(design)
    p["seeds"] = dict(graph=995100, coloring=995200, heatmap=995300, noise=995400, solver=995500, adamw_augmentation=995600)
    old = read(HERE/"pooled-donor.json")["solver"]
    p["solver_options"] = {k: v for k, v in old.items() if k not in ["highspy", "numpy", "random_seed_base"]}
    checks, source_seconds = [], []
    with tempfile.TemporaryDirectory(prefix="h4bc-preflight-") as temporary:
        directory = Path(temporary)
        for index, family in enumerate(FAMILIES):
            begin = time.monotonic()
            order = np.random.default_rng(p["seeds"]["graph"]+index).permutation(81)
            hot = np.sort(np.random.default_rng(p["seeds"]["heatmap"]+index).choice(81, 18, replace=False))
            arrays, receipt = source(index, p, order, hot)
            audit_source(index, p, arrays, receipt, order, hot)
            source_seconds.append(time.monotonic()-begin)
            rows = arrays["noisy"]
            for n in [600, 1200, 1800]:
                assert np.array_equal(counts(rows[:n]), independent_counts(rows[:n]))
            truth = reconstruct(order, family)[np.triu_indices(81, 1)]
            scores, mask = detector(rows[:600], 2*truth.astype(np.int64))
            expected_scores, expected_mask = independent_detector(rows[:600], 2*truth.astype(np.int64))
            assert np.array_equal(scores, expected_scores) and np.array_equal(mask, expected_mask)
            assert (mask.sum(1) == 9).all()
            for k, value in independent_score(rows[:600], mask).items():
                assert np.array_equal(score(rows[:600], mask)[k], value)
            for degree in [1, 2]:
                cost = counts(rows[:600])
                proof, record = F.lp(cost, degree, index, degree, directory, family+str(degree), p)
                objective = A.certificate(proof, record, cost, degree, index, degree, p)
                assert objective is not None
                result = A.outcome(proof["primal_twice"], truth, degree, cost, objective)
                if degree != {"cycle81": 2, "power2": 4, "power3": 6, "cliques9": 8}[family]:
                    assert result["truth_degree_feasible"] is False and result["truth_gap_twice"] is None
                checks.append(dict(family=family, degree=degree, independent_certificate=True, outcome=result))
        # Finite proof in a fractional donor must retain its literal values.
        fractional = np.ones(3240, np.int64)
        assert np.array_equal(detector(rows[:600], fractional)[1], independent_detector(rows[:600], fractional)[1])
        states = ["success", "resolved_failure_nonexact", "resolved_failure_fractional", "unresolved"]
        for a, b in itertools.product(states, repeat=2):
            actual = bounds([a], [b], minimum=1, maximum=0)["regressions"]
            outcomes = [int(y and not x) for x in ([True, False] if a == "unresolved" else [a == "success"])
                        for y in ([True, False] if b == "unresolved" else [b == "success"])]
            assert [actual["lower"], actual["upper"]] == [min(outcomes), max(outcomes)]
        records = [dict(certificate=dict(certified=True, primal_objective_twice=d*d)) for d in range(1, 41)]
        assert F.selected_degree(records)[0] == 1
        records[20]["certificate"]["certified"] = False
        assert F.selected_degree(records) == (None, None)
        # Actual downstream functions must skip uncertified grids without a solver call.
        saved = F.OUT, F.PROTOCOL, F.observations, F.lp
        work = directory/"skip-lifecycle"
        folder = work/"fits"/"b000"
        folder.mkdir(parents=True)
        dummy_protocol = work/"protocol.json"
        save(dummy_protocol, {})
        save(work/"grid-seal.json", {})
        save(work/"donor-seal.json", {})
        save(folder/"grid.json", dict(selected_degree=None))
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(1)
            raise AssertionError("Uncertified grid reached solver")
        try:
            F.OUT, F.PROTOCOL, F.observations, F.lp = work, dummy_protocol, lambda i: rows, forbidden
            F.donor_bank(0, p)
            F.masked_bank(0, p)
            assert not calls
            for arm in ["donor", "pooled", "masked"]:
                assert read(folder/f"{arm}.json")["executed"] is False
            try:
                F.donor_bank(0, p)
            except AssertionError:
                pass
            else:
                raise AssertionError("Existing fit overwritten")
        finally:
            F.OUT, F.PROTOCOL, F.observations, F.lp = saved
        import fresh_pooled_donor_common as C
        saved = C.OUT, C.PROTOCOL, C.N
        seals = directory/"seals"
        seals.mkdir()
        try:
            C.OUT, C.PROTOCOL, C.N = seals, dummy_protocol, 2
            try:
                C.seal("grid")
            except AssertionError:
                pass
            else:
                raise AssertionError("Missing phase files sealed")
            for name in C.files_for("grid"):
                path = seals/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic sealed bytes")
            C.seal("grid")
            C.check_seal("grid", seals)
            changed = seals/sorted(C.files_for("grid"))[0]
            changed.write_bytes(b"mutated")
            try:
                C.check_seal("grid", seals)
            except AssertionError:
                pass
            else:
                raise AssertionError("Mutated sealed file accepted")
            try:
                C.seal("grid")
            except AssertionError:
                pass
            else:
                raise AssertionError("Existing seal overwritten")
        finally:
            C.OUT, C.PROTOCOL, C.N = saved
        from run_fresh_pooled_donor import enforce_BP_deadline
        import subprocess
        child = subprocess.Popen(["python3", "-c", "import time; time.sleep(10)"], start_new_session=True)
        deadline_directory = directory/"deadline"
        deadline_directory.mkdir()
        try:
            active = deadline_directory/"bp-active0.json"
            save(active, dict(status="idle", pid=child.pid, gpu=0, index=0))
            assert enforce_BP_deadline(deadline_directory, 0, child) is False
            save(active, dict(status="running", pid=child.pid, gpu=0, index=0,
                              job_started_monotonic_ns=time.monotonic_ns()))
            assert enforce_BP_deadline(deadline_directory, 0, child) is False
            save(active, dict(status="running", pid=child.pid, gpu=0, index=0,
                              job_started_monotonic_ns=time.monotonic_ns()-31_000_000_000))
            assert enforce_BP_deadline(deadline_directory, 0, child) is True
            assert child.wait(timeout=2) == -9
            assert read(deadline_directory/"bp-watchdog-failure0.json")["status"] == "incomplete"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
        invalid = rows.astype(float)
        try:
            validate_labels(invalid, (1800, 81))
        except AssertionError:
            pass
        else:
            raise AssertionError("Noninteger labels accepted")
    bp = ROOT/"artifacts/research/20261114/foureyes/fresh_pooled_donor_prefreeze/bp_gate"
    gate_records = [read(bp/f"gpu{gpu}.json") for gpu in range(4)]
    assert all(r["status"] == "passed" and r["source_sha256"] == sha(HERE/"fresh_pooled_donor_bp.py") for r in gate_records)
    record = dict(status="passed", proposal_sha256=sha(PROPOSAL), cohort_draws=0,
                  synthetic_full1800row_families=4, independent_source_seconds=source_seconds,
                  new_synthetic_LPs=8, certificates=checks, mutations_rejected=24,
                  literal_fractional_donor_checked=True, paired_uncertainty_cases=16,
                  lower_degree_tie_and_uncertified_grid_checked=True,
                  downstream_solver_skip_overwrite_missing_phase_and_seal_mutation_checked=True,
                  independent_parent_BP_deadline_idle_live_and_owned_termination_checked=True,
                  seed_audit_sha256=sha(HERE/"fresh-pooled-donor-seed-audit.json"),
                  BP_gate_sha256={str((bp/f"gpu{gpu}.json").relative_to(ROOT)): sha(bp/f"gpu{gpu}.json") for gpu in range(4)},
                  source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_paths()}, wall_seconds=time.monotonic()-started)
    save(HERE/"fresh-pooled-donor-preflight.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ["source_sha256", "certificates"]}), flush=True)


if __name__ == "__main__":
    main()
