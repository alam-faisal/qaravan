"""RunContext: stopping criteria for iterative optimization loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunContext:
    """Stopping criteria for iterative optimization loops (environment sweep, VQE, etc.).

    should_stop receives per-sweep costs (one entry per complete pass).
    max_iter:      hard cap on number of full sweeps.
    stop_ratio:    stop if |Δcost| / (|cost[-2]| + ε) < stop_ratio; None to disable.
    stop_absolute: stop if cost[-1] < stop_absolute; None to disable.
    quiet:         suppress per-sweep log output.
    """

    max_iter: int = 1000
    stop_ratio: float | None = 1e-6
    stop_absolute: float | None = None
    quiet: bool = True

    def should_stop(self, sweep_costs: list[float], sweep: int) -> bool:
        """Return True if any stopping criterion is met.

        sweep_costs: one entry per completed sweep (not per gate update).
        sweep: number of sweeps completed so far.
        """
        if sweep >= self.max_iter:
            return True
        if self.stop_absolute is not None and sweep_costs[-1] < self.stop_absolute:
            return True
        if self.stop_ratio is not None and len(sweep_costs) >= 2:
            delta = abs(sweep_costs[-1] - sweep_costs[-2])
            if delta < self.stop_ratio * (abs(sweep_costs[-2]) + 1e-12):
                return True
        return False
