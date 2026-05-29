"""Tests for DynamicRound and DynamicCircuit in core/dynamic_circuit.py."""

import numpy as np

from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit
from qaravan.core.dynamic_circuit import DynamicCircuit, DynamicRound
from qaravan.core.gates import H, X


# ---------------------------------------------------------------------------
# DynamicRound — decoder calling convention
# ---------------------------------------------------------------------------


def test_dynamic_round_no_measurement_passes_empty_string():
    """meas_sites=[] calls decoder with '' and applies the returned circuit.

    Catches: decoder invoked with wrong argument on unmeasured rounds, or skipped.
    """
    received: list[str] = []

    def tracking_decoder(outcome: str) -> Circuit:
        received.append(outcome)
        return Circuit([X(0)], num_sites=1)

    rnd = DynamicRound(meas_sites=[], decoder=tracking_decoder)
    dqc = DynamicCircuit(rounds=[rnd])
    sv = Statevector(bitstring="0")
    sv_out, outcomes = dqc.run(sv)

    assert received == [""], f"decoder received {received!r}, expected ['']"
    assert outcomes == []
    # X applied to |0⟩ → |1⟩
    np.testing.assert_allclose(sv_out.to_array(), [0.0, 1.0], atol=1e-12)


def test_dynamic_round_measurement_calls_decoder_with_outcome():
    """meas_sites non-empty: decoder receives the actual outcome string.

    Catches: decoder called with wrong string (e.g. always '').
    """
    received: list[str] = []

    def tracking_decoder(outcome: str) -> Circuit:
        received.append(outcome)
        return Circuit([], num_sites=1)

    # |0⟩ measured on site 0 → deterministic outcome "0"
    rnd = DynamicRound(meas_sites=[0], decoder=tracking_decoder)
    dqc = DynamicCircuit(rounds=[rnd])
    sv = Statevector(bitstring="0")
    _, outcomes = dqc.run(sv)

    assert received == ["0"]
    assert outcomes == ["0"]


# ---------------------------------------------------------------------------
# DynamicCircuit.run — round ordering and state evolution
# ---------------------------------------------------------------------------


def test_dynamic_circuit_run_applies_rounds_in_order():
    """Two no-measurement rounds: X then H gives |−⟩.

    Catches: rounds applied in wrong order.
    """
    round_0 = DynamicRound(
        meas_sites=[], decoder=lambda _: Circuit([X(0)], num_sites=1)
    )
    round_1 = DynamicRound(
        meas_sites=[], decoder=lambda _: Circuit([H(0)], num_sites=1)
    )
    dqc = DynamicCircuit(rounds=[round_0, round_1])

    sv_out, outcomes = dqc.run(Statevector(bitstring="0"))

    INV_SQRT2 = 1.0 / np.sqrt(2)
    np.testing.assert_allclose(sv_out.to_array(), [INV_SQRT2, -INV_SQRT2], atol=1e-12)
    assert outcomes == []


def test_dynamic_circuit_run_does_not_mutate_init_state():
    """init_state array is unchanged after run().

    Catches: shared-reference bug where run() modifies the input tensor in place.
    """
    sv = Statevector(bitstring="0")
    original = sv.to_array().copy()
    rnd = DynamicRound(meas_sites=[], decoder=lambda _: Circuit([X(0)], num_sites=1))
    DynamicCircuit(rounds=[rnd]).run(sv)
    np.testing.assert_array_equal(sv.to_array(), original)


# ---------------------------------------------------------------------------
# DynamicCircuit.run — output structure
# ---------------------------------------------------------------------------


def test_dynamic_circuit_run_returns_only_measurement_outcomes():
    """outcomes list has one entry per round with non-empty meas_sites.

    Catches: outcomes list includes entries for no-measurement rounds.
    """
    no_meas = DynamicRound(meas_sites=[], decoder=lambda _: Circuit([], num_sites=2))
    meas = DynamicRound(meas_sites=[0], decoder=lambda _: Circuit([], num_sites=2))
    dqc = DynamicCircuit(rounds=[no_meas, meas, no_meas])
    _, outcomes = dqc.run(Statevector(bitstring="00"))
    assert len(outcomes) == 1


def test_dynamic_circuit_outcome_string_lengths():
    """Each outcome string has length == len(meas_sites) for that round.

    Catches: off-by-one in meas_sites construction.
    """
    rnd = DynamicRound(meas_sites=[0, 1], decoder=lambda _: Circuit([], num_sites=2))
    dqc = DynamicCircuit(rounds=[rnd])
    _, outcomes = dqc.run(Statevector(bitstring="00"))
    assert len(outcomes) == 1
    assert len(outcomes[0]) == 2


def test_dynamic_circuit_empty_correction_is_no_op():
    """Decoder returning Circuit([], num_sites=N) leaves state unchanged.

    Catches: crash or state corruption when correction has no gates.
    """
    sv = Statevector(bitstring="0")
    original = sv.to_array().copy()
    rnd = DynamicRound(meas_sites=[0], decoder=lambda _: Circuit([], num_sites=1))
    sv_out, _ = DynamicCircuit(rounds=[rnd]).run(sv)
    np.testing.assert_allclose(sv_out.to_array(), original, atol=1e-12)


# ---------------------------------------------------------------------------
# Multi-round correctness
# ---------------------------------------------------------------------------


def test_dynamic_circuit_two_measurement_rounds():
    """Two measurement rounds each produce an outcome entry.

    Catches: outcomes list not updated on second measured round.
    """
    rnd = DynamicRound(meas_sites=[0], decoder=lambda _: Circuit([], num_sites=1))
    dqc = DynamicCircuit(rounds=[rnd, rnd])
    _, outcomes = dqc.run(Statevector(bitstring="0"))
    assert len(outcomes) == 2
    for o in outcomes:
        assert o in ("0", "1")
