"""Board capture plus the pinned upstream restart branch; frozen original untouched."""
def captured_restart_class(GS):
    import torch
    class Captured(GS.GeneralSearch):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.best_boards = torch.zeros(self.NQ, 81, dtype=torch.int32, device=self.dev)
            self.cert_boards = torch.zeros_like(self.best_boards)

        def step_check(self):
            # Same pinned Search.step_check, with two board-only copies added.
            self._moves(self.check)
            self.t += self.check
            if self.t % 4096 == 0:
                self._periodic_refresh()
            nA, C = len(self.act), self.C
            right = (self.y.long() == self.tgt).all(1).view(nA, C)
            mn, am = self.P.view(nA, C).min(1)
            better = mn < self.bestP[self.act]
            self.best_boards[self.act[better]] = self.y.view(nA, C, 81)[better, am[better]]
            self.bestP[self.act] = torch.where(better, mn, self.bestP[self.act])
            self.bestok[self.act] = torch.where(better, right.gather(1, am[:, None]).squeeze(1), self.bestok[self.act])
            fu = self.found.view(nA, C).long()
            fu = torch.where(fu >= 0, fu, torch.full_like(fu, 1 << 60))
            first, fc = fu.min(1)
            anyc = first < (1 << 60)
            if bool(anyc.any()):
                rc = (self.ycert.long() == self.tgt).all(1).view(nA, C)
                a = self.act[anyc]
                self.cert_boards[a] = self.ycert.view(nA, C, 81)[anyc, fc[anyc]]
                self.ok[a] = rc[anyc].gather(1, fc[anyc][:, None]).squeeze(1)
                self.stop_t[a] = first[anyc]
                self.stopped[a] = True
            if self.restart and self.t >= self.next_restart:
                live = self.found < 0
                if bool(live.any()):
                    self._randomize(live)
                live_u = self.act[~self.stopped[self.act]]
                self.restarts[live_u] += 1
                self.interval = int(round(self.interval * self.growth / self.check)) * self.check
                self.next_restart = self.t + self.interval
            if bool(self.stopped[self.act].float().mean() > 0.25):
                self._compact()

        def boards(self):
            return torch.where(self.stopped[:, None], self.cert_boards, self.best_boards)
    return Captured
