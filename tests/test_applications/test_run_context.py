"""Tests for RunContext in applications/run_context.py."""

import pytest

from qaravan.applications.run_context import RunContext


def test_run_context_stops_at_max_iter():
    """should_stop returns True at sweep == max_iter."""
    ctx = RunContext(max_iter=5, stop_ratio=0.0)
    costs = [1.0, 0.5, 0.3, 0.2, 0.15]
    assert ctx.should_stop(costs, 5)


def test_run_context_does_not_stop_before_max_iter():
    ctx = RunContext(max_iter=5, stop_ratio=0.0)
    costs = [1.0, 0.5, 0.3, 0.2]
    assert not ctx.should_stop(costs, 4)


def test_run_context_stops_at_plateau():
    """Relative change |Δcost| / |cost| < stop_ratio triggers stop."""
    ctx = RunContext(max_iter=1000, stop_ratio=1e-4)
    # Δcost = 1e-8, |cost[-2]| = 0.5 → ratio = 2e-8 < 1e-4
    costs = [1.0, 0.5, 0.5 - 1e-8]
    assert ctx.should_stop(costs, 3)


def test_run_context_does_not_stop_on_large_change():
    """Convergence not triggered when cost is still changing significantly."""
    ctx = RunContext(max_iter=1000, stop_ratio=1e-4)
    costs = [1.0, 0.5, 0.3]  # Δ=0.2, ratio=0.2/0.5=0.4 > 1e-4
    assert not ctx.should_stop(costs, 3)


def test_run_context_stops_at_absolute():
    """stop_absolute threshold: stops when cost < stop_absolute."""
    ctx = RunContext(max_iter=1000, stop_absolute=0.01)
    assert ctx.should_stop([0.5, 0.009], 2)
    assert not ctx.should_stop([0.5, 0.02], 2)


def test_run_context_no_stop_with_one_entry():
    """stop_ratio requires at least 2 entries; single entry doesn't trigger it."""
    ctx = RunContext(max_iter=100, stop_ratio=1e-6)
    assert not ctx.should_stop([1.0], 1)


def test_run_context_defaults():
    ctx = RunContext()
    assert ctx.max_iter == 1000
    assert ctx.stop_ratio == 1e-6
    assert ctx.stop_absolute is None
    assert ctx.quiet is True


def test_run_context_stop_ratio_none_disabled():
    """stop_ratio=None disables plateau stopping."""
    ctx = RunContext(max_iter=1000, stop_ratio=None)
    costs = [1.0, 1.0 - 1e-15]  # essentially zero change
    assert not ctx.should_stop(costs, 2)


def test_run_context_stop_absolute_none_disabled():
    """stop_absolute=None never triggers absolute stop."""
    ctx = RunContext(max_iter=1000, stop_absolute=None)
    assert not ctx.should_stop([1e-20], 1)
