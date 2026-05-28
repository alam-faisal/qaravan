"""Tests for dynamic_circuit_library: _kept_sites, build_ghz_decoder, ghz_fusion_dqc."""

import numpy as np
import pytest

from qaravan.applications.dynamic_circuit_library import (
    _kept_sites,
    build_ghz_decoder,
    ghz_fusion_dqc,
)
from qaravan.backends.statevector import Statevector
from qaravan.core.dynamic_circuit import DynamicCircuit

INV_SQRT2 = 1.0 / np.sqrt(2)


def _ghz_fidelity(sv: Statevector, fusion_sites: list[int], outcome: str) -> float:
    """F = |⟨GHZ_n|ψ_kept⟩|² using drop_sites to extract the pure kept-qubit state."""
    pure_kept = sv.drop_sites(fusion_sites, outcome)
    n = pure_kept.num_sites
    ghz = np.zeros(2**n)
    ghz[0] = ghz[-1] = INV_SQRT2
    return float(abs(np.dot(ghz, pure_kept.to_array())) ** 2)


# ---------------------------------------------------------------------------
# _kept_sites
# ---------------------------------------------------------------------------


def test_kept_sites_first_cluster():
    """First cluster (cluster_idx=0): sites 0..cluster_size-2."""
    assert _kept_sites(0, 3, 4) == [0, 1]
    assert _kept_sites(0, 4, 3) == [0, 1, 2]


def test_kept_sites_last_cluster():
    """Last cluster: sites (C-1)*k+1 .. C*k-1."""
    # 4 clusters of size 3: last cluster is sites 9,10,11 → kept = [10,11]
    assert _kept_sites(3, 3, 4) == [10, 11]


def test_kept_sites_interior_cluster():
    """Interior cluster: single site cluster_idx*k+1 for cluster_size=3."""
    # cluster_idx=1, cluster_size=3, num_clusters=4 → site 4
    assert _kept_sites(1, 3, 4) == [4]
    # cluster_idx=2, cluster_size=3, num_clusters=4 → site 7
    assert _kept_sites(2, 3, 4) == [7]


# ---------------------------------------------------------------------------
# build_ghz_decoder — output validation
# ---------------------------------------------------------------------------


def test_build_ghz_decoder_empty_outcome_gives_no_correction():
    """All-zero outcome '00' → no gates (no corrections needed)."""
    decoder = build_ghz_decoder(cluster_size=3, num_clusters=2, total_qubits=6)
    circ = decoder("00")
    assert circ.gates == []


def test_build_ghz_decoder_cancellations_agree_with_naive():
    """use_cancellations=True and False give same final state for n=4 (all 4 outcomes).

    Uses statevector application rather than full unitary matrix — avoids 4096×4096 matrices.
    Catches: cancellation folding giving wrong sign or wrong qubit.
    Misses: both paths wrong the same way.
    """
    cs, num_clusters, total_qubits = 3, 2, 6
    d_naive = build_ghz_decoder(cs, num_clusters, total_qubits, use_cancellations=False)
    d_cancel = build_ghz_decoder(cs, num_clusters, total_qubits, use_cancellations=True)
    sv_ref = Statevector(num_sites=total_qubits, random_seed=7)
    for outcome in ["00", "01", "10", "11"]:
        sv_naive = sv_ref.apply(d_naive(outcome)) if d_naive(outcome).gates else sv_ref
        sv_cancel = (
            sv_ref.apply(d_cancel(outcome)) if d_cancel(outcome).gates else sv_ref
        )
        np.testing.assert_allclose(
            sv_naive.to_array(),
            sv_cancel.to_array(),
            atol=1e-10,
            err_msg=f"outcome={outcome}: naive and cancel disagree",
        )


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — input validation
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_raises_for_small_cluster_size():
    """cluster_size <= 2 raises ValueError."""
    with pytest.raises(ValueError):
        ghz_fusion_dqc(4, cluster_size=2)


def test_ghz_fusion_dqc_raises_for_non_integer_num_clusters():
    """(n-2) not divisible by (cluster_size-2) raises ValueError."""
    with pytest.raises(ValueError):
        ghz_fusion_dqc(5, cluster_size=4)


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — structure
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_has_two_rounds():
    """ghz_fusion_dqc always returns a DynamicCircuit with exactly 2 rounds."""
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    assert isinstance(dqc, DynamicCircuit)
    assert len(dqc.rounds) == 2


def test_ghz_fusion_dqc_round0_no_measurement():
    """Round 0 has meas_sites=[]."""
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    assert dqc.rounds[0].meas_sites == []


def test_ghz_fusion_dqc_round1_meas_sites():
    """Round 1 meas_sites == boundary qubit indices.

    n=4, cluster_size=3: 2 clusters, 1 boundary at qubits [2,3].
    n=6, cluster_size=3: 4 clusters, 3 boundaries at [2,3,5,6,8,9].
    Catches: off-by-one or wrong boundary qubit formula.
    """
    dqc4 = ghz_fusion_dqc(4, cluster_size=3)
    assert dqc4.rounds[1].meas_sites == [2, 3]

    dqc6 = ghz_fusion_dqc(6, cluster_size=3)
    assert dqc6.rounds[1].meas_sites == [2, 3, 5, 6, 8, 9]


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — physics correctness
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_4qubit_fidelity():
    """n=4, cluster_size=3: 20 runs, fidelity with |GHZ_4⟩ on kept sites = 1.0.

    Fusion sites: {2,3}. Kept: {0,1,4,5}.
    Catches: wrong correction for any of the 4 outcomes.
    """
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    init = Statevector(bitstring="000000")
    for _ in range(20):
        sv, outcomes = dqc.run(init)
        assert np.isclose(_ghz_fidelity(sv, [2, 3], outcomes[0]), 1.0, atol=1e-10)


def test_ghz_fusion_dqc_4qubit_all_outcomes_covered():
    """100 runs of n=4 must yield all 4 outcome strings.

    Catches: correction logic that silently fails for specific outcomes.
    """
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    init = Statevector(bitstring="000000")
    seen: set[str] = set()
    for _ in range(100):
        _, outcomes = dqc.run(init)
        seen.add(outcomes[0])
    assert seen == {"00", "01", "10", "11"}


def test_ghz_fusion_dqc_6qubit_fidelity():
    """n=6, cluster_size=3: 20 runs, fidelity = 1.0, outcome length = 6.

    Fusion sites: {2,3,5,6,8,9}. 3 boundaries × 2 bits = 6-bit outcome.
    Catches: decoder error for multi-boundary case.
    """
    dqc = ghz_fusion_dqc(6, cluster_size=3)
    init = Statevector(bitstring="0" * 12)
    for _ in range(20):
        sv, outcomes = dqc.run(init)
        assert len(outcomes[0]) == 6
        assert np.isclose(
            _ghz_fidelity(sv, [2, 3, 5, 6, 8, 9], outcomes[0]), 1.0, atol=1e-10
        )


def test_ghz_fusion_dqc_use_cancellations_same_physics():
    """use_cancellations=True gives fidelity 1.0 — same final state as naive.

    Catches: cancellation folding introducing a sign or qubit error.
    """
    init = Statevector(bitstring="000000")
    for _ in range(20):
        dqc = ghz_fusion_dqc(4, cluster_size=3, use_cancellations=True)
        sv, outcomes = dqc.run(init)
        assert np.isclose(_ghz_fidelity(sv, [2, 3], outcomes[0]), 1.0, atol=1e-10)
