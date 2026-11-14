"""Sudoku certificate intervention derived from pinned Apache-2.0 CRAM move kernel.
Original: upstream/cram/cram_sudoku_general.py at62bbe830c7d6f87f2ba18f30a28039258b986454.
Retained license: references/CRAM-LICENSE.txt. Only stopping adds exact integer
rule conflicts; learned fields, heat bath, focus and random keys are unchanged.
"""
import triton
import triton.language as tl

@triton.jit
def _moves_gate(nbr_ptr, R_ptr, Bc_ptr, y_ptr, G_ptr, flag_ptr, E_ptr, violation_ptr, free_ptr, fmask_ptr, nfree_ptr,
                   temp_ptr, found_ptr, ycert_ptr, seed_ptr, nmoves, move0, thr, tau,
                   MODE: tl.constexpr, GATE: tl.constexpr, N: tl.constexpr, V: tl.constexpr, D: tl.constexpr,
                   NB: tl.constexpr, VB: tl.constexpr, DB: tl.constexpr):
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
    if GATE != 0:
        violations = tl.load(violation_ptr + pid).to(tl.int32)
    T = tl.load(temp_ptr + pid)
    nf = tl.load(nfree_ptr + pid)
    found = tl.load(found_ptr + pid)
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = (pid * 512).to(tl.int32)
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
        if GATE != 0:
            all_y = tl.load(ybase + rows, mask=rmask, other=-1)
            peer = rmask & (rows != cell) & ((rows // 9 == cell // 9) | (rows % 9 == cell % 9) | ((rows // 27 == cell // 27) & ((rows % 9) // 3 == (cell % 9) // 3)))
            violations += tl.sum((peer & (all_y == new)).to(tl.int32), 0) - tl.sum((peer & (all_y == old)).to(tl.int32), 0)
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
        if GATE == 0:
            cert = (found < 0) & (E <= tau)
        elif GATE == 1:
            cert = (found < 0) & (E <= tau) & (violations == 0)
        else:
            cert = (found < 0) & (violations == 0)
        yv = tl.load(ybase + rows, mask=rmask & cert, other=0)
        tl.store(ycert_ptr + pid * N + rows, yv, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(E_ptr + pid, E)
    tl.store(found_ptr + pid, found)
    if GATE != 0:
        tl.store(violation_ptr + pid, violations)

