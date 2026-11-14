"""Exact chain-range dispatch for the pinned CRAM Sudoku move kernel.

Kernel derived from Prefrontal-AI/CRAM, commit
62bbe830c7d6f87f2ba18f30a28039258b986454, cram/cram_sudoku_general.py.
Licensed under Apache-2.0; see upstream/LICENSE.
Modification: use the original global chain index for the random key while
addressing local chain slices. All arithmetic, updates and stopping tests
are unchanged. The original solver retains refresh, checks and compaction.
This optional execution optimization is not enabled by importing this file.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _moves_general_offset(nbr_ptr, R_ptr, Bc_ptr, y_ptr, G_ptr, flag_ptr, E_ptr, free_ptr, fmask_ptr, nfree_ptr,
                   temp_ptr, found_ptr, ycert_ptr, seed_ptr, nmoves, move0, thr, tau,
                   MODE: tl.constexpr, N: tl.constexpr, V: tl.constexpr, D: tl.constexpr,
                   NB: tl.constexpr, VB: tl.constexpr, DB: tl.constexpr, RNG_OFFSET: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    ks = tl.arange(0, DB)
    rmask = rows < N
    cmask = cols < V
    kmask = ks < D
    km2 = kmask[:, None] & cmask[None, :]
    ybase = y_ptr + pid * N
    gbase = G_ptr + pid * N * V
    fbase = flag_ptr + pid * N
    mbase = fmask_ptr + pid * N
    E = tl.load(E_ptr + pid)
    T = tl.load(temp_ptr + pid)
    nf = tl.load(nfree_ptr + pid)
    found = tl.load(found_ptr + pid)
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = ((pid + RNG_OFFSET) * 512).to(tl.int32)
        u0 = tl.rand(sk, base)
        idx = tl.minimum((u0 * nf).to(tl.int32), nf - 1)
        cell = tl.load(free_ptr + pid * N + idx)
        if MODE == 2:
            fl = tl.load(fbase + rows, mask=rmask, other=0) != 0
            score = tl.where(fl, tl.rand(sk, base + 64 + rows), -1.0)
            cc = tl.argmax(score, 0)
            cell = tl.where(tl.max(score, 0) >= 0.0, cc, cell)
        Li = tl.load(gbase + cell * V + cols, mask=cmask, other=0.0) + tl.load(Bc_ptr + cell * V + cols, mask=cmask, other=0.0)
        uu = tl.rand(sk, base + 16 + cols)
        gum = -tl.log(-tl.log(tl.maximum(uu, 1e-20)))
        lg = tl.where(cmask, -Li / T + gum, float('-inf'))
        new = tl.argmax(lg, 0)
        old = tl.load(ybase + cell)
        new = tl.where(found >= 0, old, new)
        d = (new != old).to(tl.float32)
        nb = tl.load(nbr_ptr + cell * D + ks, mask=kmask, other=0)
        rb = R_ptr + ((cell * D + ks[:, None]) * V + cols[None, :]) * V  # R[cell, k, a, :]
        delta = (tl.load(rb + new, mask=km2, other=0.0) - tl.load(rb + old, mask=km2, other=0.0)) * d
        pg = gbase + nb[:, None] * V + cols[None, :]
        Gn = tl.load(pg, mask=km2, other=0.0) + delta
        tl.store(pg, Gn, mask=km2)
        ynb = tl.load(ybase + nb, mask=kmask, other=0)
        own = tl.sum(tl.where(cols[None, :] == ynb[:, None], Gn, 0.0), 1) + tl.load(Bc_ptr + nb * V + ynb, mask=kmask, other=0.0)
        fnb = tl.load(mbase + nb, mask=kmask, other=0) != 0
        tl.store(fbase + nb, ((own > thr) & fnb).to(tl.int32), mask=kmask)
        lnew = tl.sum(tl.where(cols == new, Li, 0.0), 0)
        E = E + (lnew - tl.sum(tl.where(cols == old, Li, 0.0), 0)) * d
        tl.debug_barrier()
        tl.store(ybase + cell, new)
        tl.store(fbase + cell, (lnew > thr).to(tl.int32))
        tl.debug_barrier()
        cert = (found < 0) & (E <= tau)
        yv = tl.load(ybase + rows, mask=rmask & cert, other=0)
        tl.store(ycert_ptr + pid * N + rows, yv, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(E_ptr + pid, E)
    tl.store(found_ptr + pid, found)



class ShardedMoves:
    """Dispatch just the move kernel across GPUs in one host.

    All authoritative state belongs to the original search on devices[0].
    Each call begins and ends with that same full state. There is no split
    evaluation, separate RNG stream, independent compaction, or new budget.
    """
    MUTABLE = ("y", "G", "flag", "P", "found", "ycert")
    INPUTS = MUTABLE + ("freelist", "fmask", "nfree")

    def __init__(self, devices):
        self.devices = tuple(torch.device(d) for d in devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("Distinct CUDA devices are required")
        if any(d.type != "cuda" or d.index is None for d in self.devices):
            raise ValueError("Use explicit indexed CUDA devices")
        self.fixed = {}
        self.owner = None

    def __call__(self, search, nmoves):
        primary = self.devices[0]
        if search.y.device != primary:
            raise ValueError(f"Authoritative state {search.y.device} must be on {primary}")
        if self.owner is not search:
            self.fixed.clear()
            self.owner = search
        total = len(search.y)
        if not total:
            return
        assert total * 512 < 2 ** 31
        torch.cuda.synchronize(primary)
        # Keep all chains of a puzzle together. Range boundaries only affect
        # dispatch, never original refresh batching or compaction decisions.
        puzzles = total // search.C
        assert puzzles * search.C == total
        jobs = []
        for rank, device in enumerate(self.devices):
            lo = (puzzles * rank // len(self.devices)) * search.C
            hi = (puzzles * (rank + 1) // len(self.devices)) * search.C
            if lo == hi:
                continue
            with torch.cuda.device(device):
                if device not in self.fixed:
                    self.fixed[device] = tuple(t.to(device, non_blocking=True)
                                              for t in (search.nbr, search.R, search.Bc))
                data = {name: getattr(search, name)[lo:hi].to(device, non_blocking=True)
                        for name in self.INPUTS}
                seed = torch.tensor([search.seed * 1000003 + search.t // search.check],
                                    dtype=torch.int64, device=device)
                temp = torch.full((hi - lo,), search.current_temp(), device=device)
                jobs.append((device, lo, hi, data, seed, temp))
        # Stage every peer input before launching the primary shard. Otherwise
        # the peer-copy source event could unnecessarily wait for that kernel.
        for device, lo, hi, data, seed, temp in jobs:
            with torch.cuda.device(device):
                _moves_general_offset[(hi - lo,)](
                    *self.fixed[device], data["y"], data["G"], data["flag"], data["P"],
                    data["freelist"], data["fmask"], data["nfree"], temp, data["found"],
                    data["ycert"], seed, nmoves, search.t, search.thr, search.tau + search.tol,
                    MODE=2 if search.focused else 0, N=81, V=9, D=search.D,
                    NB=128, VB=16, DB=triton.next_power_of_2(search.D),
                    RNG_OFFSET=lo, num_warps=1)
        # Cross-device copies observe source work through CUDA copy events.
        # Returning after all destination copies are queued on the primary
        # default stream keeps the unchanged step_check correctly ordered.
        with torch.cuda.device(primary):
            for device, lo, hi, data, seed, temp in jobs:
                torch.cuda.synchronize(device)
                if device == primary:
                    continue
                for name in self.MUTABLE:
                    getattr(search, name)[lo:hi].copy_(data[name], non_blocking=True)
            torch.cuda.synchronize(primary)

    def install(self, search):
        import types
        search._moves = types.MethodType(lambda s, n: self(s, n), search)
        return search
