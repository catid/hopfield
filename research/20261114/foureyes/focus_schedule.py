"""Alternate native focused/uniform blocks; use the normal exact refresh each check."""


def scheduled_search_class(base, arm):
    assert arm in ('focused', 'mixed16', 'uniform')

    class Scheduled(base):
        def step_check(self):
            assert self.check == 4096 and self.t % self.check == 0
            block = self.t // self.check
            self.focused = arm == 'focused' or (arm == 'mixed16' and block % 16 != 15)
            # Both kernels consume the same field representation. The inherited
            # 4096-move refresh reconstructs flags before the next block, even
            # when the uniform kernel did not update them incrementally.
            super().step_check()

    return Scheduled
