"""Derived equal-digit backend; pinned upstream remains untouched.

Uses the released sparse move kernel with hscale=1 and its bias-aware arithmetic.
Retains GeneralSearch refresh ordering, floating fields, focus offsets and capture.
"""
from restart_capture import captured_restart_class as captured_search_class


def sparse_search_class(GS, SP):
    import torch
    import triton

    class DerivedSparse(captured_search_class(GS)):
        def __init__(self, S, Bc, sol, free, **kw):
            super().__init__(S, Bc, sol, free, **kw)
            assert bool((self.Bc == 0).all()), 'Only zero unary canonical models supported'
            J = self.Sg[:, :, 0, 0]
            eye = torch.eye(9, device=self.dev)
            assert torch.equal(self.Sg, J[:, :, None, None] * eye)
            assert torch.equal(J, J.T) and bool((J.diag() == 0).all())
            # Unique padding destinations avoid aliasing stores in the sparse kernel.
            nbr = []
            for i in range(81):
                present = torch.nonzero(J[i] != 0).flatten()
                missing = torch.nonzero(J[i] == 0).flatten()
                nbr.append(torch.cat((present, missing[:self.D-len(present)])))
            self.sparse_nbr = torch.stack(nbr).to(torch.int32).contiguous()
            self.w = J.gather(1, self.sparse_nbr.long()).contiguous()
            self.fl_move = self._fl_move()

        def _fl_move(self):
            refresh = 2 * 81 * self.D * 9 * 9 / 4096
            return 4 * self.D + (self.D + 1 if self.focused else 0) + 5 * 9 + 4 + refresh

        def _moves(self, nmoves):
            H = len(self.y)
            if H == 0:
                return
            assert H * 512 < 2 ** 31
            seed_t = torch.tensor([self.seed * 1000003 + self.t // self.check], dtype=torch.int64, device=self.dev)
            temp = torch.full((H,), self.current_temp(), device=self.dev)
            SP._moves_sparse[(H,)](
                self.sparse_nbr, self.w, self.B, self.x, self.y, self.G,
                self.flag, self.P, self.freelist, self.fmask, self.nfree, temp,
                self.found, self.ycert, seed_t, nmoves, self.t, 1.0, self.thr,
                self.tau + self.tol, MODE=2 if self.focused else 0, NOBIAS=False,
                N=81, V=9, D=self.D, NB=128, VB=16,
                DB=triton.next_power_of_2(self.D), num_warps=1)

    return DerivedSparse
