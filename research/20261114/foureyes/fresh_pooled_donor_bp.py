"""H4bc ordinary BP AdamW control and unrelated full-size feasibility gates."""
import time
START = time.monotonic()
from pathlib import Path
import argparse
import json
import os
import resource
import signal
from weak_weight_study import ROOT, HERE, setup, sha, save

GATE = ROOT / "artifacts/research/20261114/foureyes/fresh_pooled_donor_prefreeze/bp_gate"
CONFIG = dict(lr=.01, weight_decay=.01, betas=[.9, .999], eps=1e-8)


def independent_transform(np, weights):
    values = weights.astype(np.float64)
    diagonal = sum(values[:, d, d] for d in range(9)) / 9
    off = sum(values[:, a, b] for a in range(9) for b in range(9) if a != b) / 72
    coefficients = np.maximum(diagonal - off, 0)
    positive = sorted(float(v) for v in coefficients if v > 0)
    if len(positive) < 2 or not np.isfinite(coefficients).all():
        return np.zeros(3240, bool), coefficients, False
    differences = [b - a for a, b in zip(positive, positive[1:])]
    where = max(range(len(differences)), key=lambda i: (differences[i], -i))
    if differences[where] <= 0:
        return np.zeros(3240, bool), coefficients, False
    threshold = (positive[where] + positive[where + 1]) / 2
    return coefficients > threshold, coefficients, True


def train(torch, np, CF, noisy, augmentation_seed):
    import noisy_pl_long as base
    assert noisy.shape == (1800, 81) and noisy.dtype.kind in "iu"
    assert ((noisy >= 0) & (noisy <= 8)).all()
    begun = time.monotonic()
    observations = torch.as_tensor(noisy.astype(np.int64), device="cuda")
    free = torch.ones_like(observations, dtype=torch.bool)
    draws = torch.rand((32, 1800, 9), generator=torch.Generator().manual_seed(augmentation_seed))
    w, b, seconds, history = base.fit_adam(torch, CF, observations, free, draws, CONFIG)
    assert torch.isfinite(w).all() and torch.isfinite(b).all()
    support, coeff, record = base.transform(w, torch)
    if support is None:
        support = torch.zeros(3240, device="cuda", dtype=torch.bool)
    arrays = dict(W=w.cpu().numpy(), B=b.cpu().numpy(), support=support.cpu().numpy(),
                  coefficients=coeff.cpu().numpy())
    return arrays, dict(transform=record, optimizer_loop_seconds=seconds, history=history,
                       augmentation_seed=augmentation_seed, updates=32,
                       train_and_transform_wall_seconds=time.monotonic() - begun)


def gate(gpu):
    target = GATE / f"gpu{gpu}.json"
    assert not target.exists() and not (GATE / f"gpu{gpu}-started.json").exists()
    save(GATE / f"gpu{gpu}-started.json", dict(pid=os.getpid(), gpu=gpu, source_sha256=sha(__file__)))
    lock = None
    def timeout(signum, frame):
        raise TimeoutError("Synthetic full-size BP gate reached 30-second worker cap")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 30 - (time.monotonic() - START)))
    try:
        lock, torch, np, *_ = setup(gpu)
        import noisy_pl_long as base
        from audit_heterogeneous_noise import numpy_raw_loss_gradient
        _, _, CF = base.dependencies()
        rng = np.random.default_rng(202613760000 + gpu)
        noisy = rng.integers(0, 9, (1800, 81), dtype=np.int64)
        free = np.ones_like(noisy, dtype=bool)
        w = rng.normal(0, .03, (3240, 9, 9)).astype(np.float32)
        b = rng.normal(0, .03, (81, 9)).astype(np.float32)
        expected, dw, db = numpy_raw_loss_gradient(w.astype(np.float64), b.astype(np.float64), noisy[:17], free[:17])
        wt = torch.tensor(w, device="cuda", requires_grad=True)
        bt = torch.tensor(b, device="cuda", requires_grad=True)
        loss = CF.pl_loss(CF.full_S(wt, CF.pair_index("cuda")), bt,
                          torch.tensor(noisy[:17], device="cuda"), torch.tensor(free[:17], device="cuda"))
        loss.backward()
        errors = [abs(float(loss.detach()) - expected), float(abs(wt.grad.cpu().numpy() - dw).max()),
                  float(abs(bt.grad.cpu().numpy() - db).max())]
        assert errors[0] < 2e-6 and max(errors[1:]) < 2e-7
        arrays, receipt = train(torch, np, CF, noisy, 202613770000 + gpu)
        fixtures = [arrays["W"], np.zeros_like(w), np.ones_like(w)]
        # Tie at the largest gap: the first gap is the fixed rule.
        tied = np.zeros_like(w)
        for d in range(9):
            tied[:3, d, d] = [1, 2, 3]
        fixtures.append(tied)
        max_error = 0.
        for value in fixtures:
            observed, coeff, meta = base.transform(torch.tensor(value, device="cuda"), torch)
            actual = np.zeros(3240, bool) if observed is None else observed.cpu().numpy()
            support, independent, eligible = independent_transform(np, value)
            error = float(abs(coeff.cpu().numpy() - independent).max())
            max_error = max(max_error, error)
            assert error < 1e-12 and np.array_equal(actual, support) and meta["eligible"] == eligible
        elapsed = time.monotonic() - START
        assert elapsed < 30
        save(target, dict(status="passed", gpu=gpu, pid=os.getpid(), worker_wall_seconds=elapsed,
                          CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                          loss_gradient_errors=errors, transform_fixtures=len(fixtures), transform_max_error=max_error,
                          full_shape=[1800, 81], full_updates=32, fit=receipt,
                          synthetic_only=True, cohort_draws=0, source_sha256=sha(__file__)))
    except BaseException as exc:
        save(target, dict(status="failed", gpu=gpu, error=repr(exc), worker_wall_seconds=time.monotonic()-START,
                          synthetic_only=True, cohort_draws=0, source_sha256=sha(__file__)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["gate"])
    parser.add_argument("--gpu", type=int, required=True, choices=range(4))
    args = parser.parse_args()
    gate(args.gpu)
