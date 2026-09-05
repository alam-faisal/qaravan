"""Eigensolvers in applications/algorithms.py, plus the spin-register derivation they use."""

import pytest
from qaravan.core.fermions import (
    FermionicEncoding,
    bath_interleaved_mode_order,
    interleaved_mode_order,
)


def _fh_encoding(num_fermionic_sites: int, ordering: str) -> FermionicEncoding:
    mode_order = (
        None if ordering == "stacked" else interleaved_mode_order(num_fermionic_sites)
    )
    return FermionicEncoding(
        "jordan_wigner", num_modes=2 * num_fermionic_sites, mode_order=mode_order
    )


# ---------------------------------------------------------------------------
# FermionicEncoding.spin_qubits
# ---------------------------------------------------------------------------


def test_spin_qubits_stacked_layout():
    # mode_order=None is the identity, so up modes 0,1,2 stay on qubits 0,1,2
    assert _fh_encoding(3, "stacked").spin_qubits() == ([0, 1, 2], [3, 4, 5])


def test_spin_qubits_interleaved_layout():
    # interleaved_mode_order puts up_j on qubit 2j and down_j on qubit 2j+1
    assert _fh_encoding(3, "interleaved").spin_qubits() == ([0, 2, 4], [1, 3, 5])


def test_spin_qubits_aim_bath_interleaved():
    # bath_interleaved_mode_order(2) = (1, 4, 2, 5, 0, 3); modes 0,1,2 are up, 3,4,5 down
    encoding = FermionicEncoding(
        "jordan_wigner", num_modes=6, mode_order=bath_interleaved_mode_order(2)
    )
    assert encoding.spin_qubits() == ([4, 0, 2], [5, 1, 3])


def test_spin_qubits_is_a_partition_of_all_qubits():
    for ordering in ["stacked", "interleaved"]:
        up, down = _fh_encoding(4, ordering).spin_qubits()
        assert sorted(up + down) == list(range(8)), ordering


def test_spin_qubits_raises_on_odd_mode_count():
    encoding = FermionicEncoding("jordan_wigner", num_modes=3, mode_order=(2, 0, 1))
    with pytest.raises(ValueError, match="even mode count"):
        encoding.spin_qubits()
