"""Tests for core/hamiltonians.py::FermionicHamiltonian -- the Hermitian-asserted
FermionicSum subclass that carries fermionic-native trotter_groups, encoded group-by-group
so no post-encoding re-chunking by size is needed."""

import numpy as np
import pytest
from scipy.linalg import expm

from qaravan.applications.algorithms import trotter_circuit
from qaravan.core.fermions import FermionicEncoding, c, hopping, number
from qaravan.core.hamiltonians import FermionicHamiltonian


def test_rejects_non_hermitian_terms():
    with pytest.raises(ValueError, match="not Hermitian"):
        FermionicHamiltonian([c(0)], num_modes=2)


def test_accepts_hermitian_terms_with_no_groups():
    fh = FermionicHamiltonian(hopping(0, 1, -1.0), num_modes=2)
    assert fh.trotter_groups is None


def test_encode_without_groups_gives_ungrouped_hamiltonian():
    fh = FermionicHamiltonian([number(0)], num_modes=1)
    encoding = FermionicEncoding("jordan_wigner", num_modes=1)
    ham = fh.encode(encoding)
    assert ham.trotter_groups is None
    assert ham.num_sites == 1


def test_encode_raises_on_mode_mismatch():
    fh = FermionicHamiltonian([number(0)], num_modes=1)
    encoding = FermionicEncoding("jordan_wigner", num_modes=2)
    with pytest.raises(ValueError, match="modes"):
        fh.encode(encoding)


def test_encode_with_custom_groups_preserves_group_structure():
    """A hand-built 2-mode hopping+onsite model with an explicit, user-chosen grouping --
    no FermiHubbard1D/AndersonImpurity subclassing involved."""
    hop = hopping(0, 1, -1.0)
    onsite = [number(0), number(1)]
    groups = [hop, onsite]
    terms = hop + onsite
    fh = FermionicHamiltonian(terms, num_modes=2, trotter_groups=groups)

    encoding = FermionicEncoding("jordan_wigner", num_modes=2)
    ham = fh.encode(encoding)

    assert ham.trotter_groups is not None
    assert len(ham.trotter_groups) == 2
    # hopping's two halves pair into 1 HamiltonianTerm; the 2 onsite terms stay separate
    assert len(ham.trotter_groups[0]) == 1
    assert len(ham.trotter_groups[1]) == 2


def test_encode_with_custom_groups_matches_exact_propagator():
    """The real end-to-end check: a Trotter circuit built entirely from user-defined
    groups -- a 3-mode chain, never touching FermiHubbard1D/AndersonImpurity -- converges
    to the exact propagator with the expected order-2 (~dt^3, single-step) scaling."""
    t, mu = 1.3, 0.4
    hop01 = hopping(0, 1, -t)
    hop12 = hopping(1, 2, -t)
    onsite = [mu * number(0), mu * number(1), mu * number(2)]
    groups = [hop01, hop12, onsite]
    terms = hop01 + hop12 + onsite

    fh = FermionicHamiltonian(terms, num_modes=3, trotter_groups=groups)
    encoding = FermionicEncoding("jordan_wigner", num_modes=3)
    ham = fh.encode(encoding)

    dt_values = [0.05, 0.025, 0.0125]
    errors = []
    for dt in dt_values:
        circ = trotter_circuit(ham, dt, order=2)
        exact = expm(-1j * dt * ham.as_matrix())
        errors.append(np.linalg.norm(circ.to_matrix() - exact, ord="fro"))
    slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
    assert 2.8 < slope < 3.2, f"order-2 slope={slope:.3f} (expected ~3)"
