"""Tests for AndersonImpurity (star and chain geometries)."""

import numpy as np
import pytest
from scipy.linalg import expm

from qaravan.applications.algorithms import ground_state, trotter_circuit
from qaravan.core.fermions import FermionicEncoding
from qaravan.core.hamiltonians import AndersonImpurity, bath_interleaved_mode_order


def _identity_encoding(aim: AndersonImpurity) -> FermionicEncoding:
    return FermionicEncoding("jordan_wigner", num_modes=aim.num_modes)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_star_num_modes():
    aim = AndersonImpurity.star(
        eps_d=0.3, U=1.0, eps_bath=[0.1, -0.2, 0.4], V=[0.5, 0.3, 0.2]
    )
    assert aim.num_bath == 3
    assert aim.num_modes == 8  # 2 * (3 + 1)


def test_chain_num_modes():
    aim = AndersonImpurity.chain(
        eps_d=0.3, U=1.0, eps_chain=[0.1, -0.2, 0.4], t_chain=[0.5, 0.3], V=0.6
    )
    assert aim.num_bath == 3
    assert aim.num_modes == 8


def test_star_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="eps_bath"):
        AndersonImpurity.star(eps_d=0.0, U=1.0, eps_bath=[0.1, 0.2], V=[0.5])


def test_chain_wrong_t_chain_length_raises():
    with pytest.raises(ValueError, match="t_chain"):
        AndersonImpurity.chain(
            eps_d=0.0, U=1.0, eps_chain=[0.1, 0.2, 0.3], t_chain=[0.5], V=0.6
        )


def test_chain_single_bath_needs_no_hopping():
    # num_bath=1 -> t_chain must be empty (no neighbor to hop to)
    aim = AndersonImpurity.chain(eps_d=0.0, U=1.0, eps_chain=[0.2], t_chain=[], V=0.6)
    assert aim.num_bath == 1


# ---------------------------------------------------------------------------
# Hermiticity
# ---------------------------------------------------------------------------


def test_star_hermitian():
    rng = np.random.default_rng(0)
    eps_bath = rng.uniform(-1, 1, size=3).tolist()
    V = rng.uniform(-1, 1, size=3).tolist()
    aim = AndersonImpurity.star(eps_d=0.4, U=2.0, eps_bath=eps_bath, V=V)
    H = aim.encode(_identity_encoding(aim)).as_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_chain_hermitian():
    rng = np.random.default_rng(1)
    eps_chain = rng.uniform(-1, 1, size=3).tolist()
    t_chain = rng.uniform(-1, 1, size=2).tolist()
    aim = AndersonImpurity.chain(
        eps_d=0.4, U=2.0, eps_chain=eps_chain, t_chain=t_chain, V=0.7
    )
    H = aim.encode(_identity_encoding(aim)).as_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# Atomic limit (num_bath=0): analytically known spectrum, no bath at all
# ---------------------------------------------------------------------------


def test_star_atomic_limit_spectrum():
    eps_d, U = 0.7, 2.3
    aim = AndersonImpurity.star(eps_d=eps_d, U=U, eps_bath=[], V=[])
    H = aim.encode(_identity_encoding(aim)).as_matrix()
    evals = np.sort(np.linalg.eigvalsh(H))
    expected = np.sort([0.0, eps_d, eps_d, 2 * eps_d + U])
    assert np.allclose(evals, expected, atol=1e-10)


def test_chain_atomic_limit_spectrum():
    eps_d, U = -0.5, 1.8
    aim = AndersonImpurity.chain(eps_d=eps_d, U=U, eps_chain=[], t_chain=[], V=0.0)
    H = aim.encode(_identity_encoding(aim)).as_matrix()
    evals = np.sort(np.linalg.eigvalsh(H))
    expected = np.sort([0.0, eps_d, eps_d, 2 * eps_d + U])
    assert np.allclose(evals, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Star ≡ chain for num_bath=1 -- the one valid star/chain agreement check
# ---------------------------------------------------------------------------


def test_star_equals_chain_for_single_bath():
    eps_d, U, eps_1, V = 0.2, 1.5, -0.3, 0.6
    star = AndersonImpurity.star(eps_d=eps_d, U=U, eps_bath=[eps_1], V=[V])
    chain = AndersonImpurity.chain(eps_d=eps_d, U=U, eps_chain=[eps_1], t_chain=[], V=V)
    H_star = star.encode(_identity_encoding(star)).as_matrix()
    H_chain = chain.encode(_identity_encoding(chain)).as_matrix()
    assert np.allclose(
        np.sort(np.linalg.eigvalsh(H_star)),
        np.sort(np.linalg.eigvalsh(H_chain)),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Particle-number conservation
# ---------------------------------------------------------------------------


def test_star_fixed_particle_number_ground_state_has_correct_n():
    aim = AndersonImpurity.star(eps_d=0.3, U=2.0, eps_bath=[0.1, -0.2], V=[0.5, 0.4])
    ham = aim.encode(_identity_encoding(aim))
    num_qubits = aim.num_modes
    n_op = np.diag([float(bin(i).count("1")) for i in range(2**num_qubits)])
    for n_elec in [1, 2, 3]:
        sv = ground_state(ham, fixed_particle_number=n_elec)
        arr = sv.to_array()
        expectation = (arr.conj() @ n_op @ arr).real
        assert abs(expectation - n_elec) < 1e-10


def test_chain_fixed_particle_number_ground_state_has_correct_n():
    aim = AndersonImpurity.chain(
        eps_d=0.3, U=2.0, eps_chain=[0.1, -0.2], t_chain=[0.3], V=0.5
    )
    ham = aim.encode(_identity_encoding(aim))
    num_qubits = aim.num_modes
    n_op = np.diag([float(bin(i).count("1")) for i in range(2**num_qubits)])
    for n_elec in [1, 2, 3]:
        sv = ground_state(ham, fixed_particle_number=n_elec)
        arr = sv.to_array()
        expectation = (arr.conj() @ n_op @ arr).real
        assert abs(expectation - n_elec) < 1e-10


# ---------------------------------------------------------------------------
# Trotter circuits
# ---------------------------------------------------------------------------


def test_star_trotter_vs_propagator():
    aim = AndersonImpurity.star(eps_d=0.3, U=1.5, eps_bath=[0.1, -0.2], V=[0.4, 0.3])
    ham = aim.encode(_identity_encoding(aim))
    dt = 0.01
    circ = trotter_circuit(ham, dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * ham.as_matrix())
    assert np.linalg.norm(trotter_mat - exact_mat, ord="fro") < 1e-5


def test_chain_trotter_vs_propagator():
    aim = AndersonImpurity.chain(
        eps_d=0.3, U=1.5, eps_chain=[0.1, -0.2], t_chain=[0.25], V=0.4
    )
    ham = aim.encode(_identity_encoding(aim))
    dt = 0.01
    circ = trotter_circuit(ham, dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * ham.as_matrix())
    assert np.linalg.norm(trotter_mat - exact_mat, ord="fro") < 1e-5


# ---------------------------------------------------------------------------
# bath_interleaved_mode_order
# ---------------------------------------------------------------------------


def test_bath_interleaved_mode_order_two_baths():
    # b1up, b1down, b2up, b2down, i_up, i_down
    assert bath_interleaved_mode_order(2) == (1, 4, 2, 5, 0, 3)


def test_bath_interleaved_mode_order_one_bath():
    # b1up, b1down, i_up, i_down
    assert bath_interleaved_mode_order(1) == (1, 3, 0, 2)


def test_bath_interleaved_mode_order_is_a_permutation():
    for num_bath in [0, 1, 2, 5]:
        order = bath_interleaved_mode_order(num_bath)
        assert sorted(order) == list(range(2 * (num_bath + 1)))


def test_bath_interleaved_mode_order_annotate_false_is_unchanged():
    assert bath_interleaved_mode_order(
        2, annotate=False
    ) == bath_interleaved_mode_order(2)


def test_bath_interleaved_mode_order_annotate_returns_order_and_string():
    order, annotation = bath_interleaved_mode_order(2, annotate=True)
    assert order == (1, 4, 2, 5, 0, 3)
    assert annotation == "q0=b1↑ q1=b1↓ q2=b2↑ q3=b2↓ q4=i↑ q5=i↓"


def test_bath_interleaved_mode_order_matches_stacked_spectrum():
    # Applying a different mode_order must not change the physics.
    aim = AndersonImpurity.star(eps_d=0.3, U=2.0, eps_bath=[0.1, -0.2], V=[0.5, 0.4])
    stacked = FermionicEncoding("jordan_wigner", num_modes=aim.num_modes)
    reordered = FermionicEncoding(
        "jordan_wigner",
        num_modes=aim.num_modes,
        mode_order=bath_interleaved_mode_order(2),
    )
    E_stacked = np.linalg.eigvalsh(aim.encode(stacked).as_matrix())[0]
    E_reordered = np.linalg.eigvalsh(aim.encode(reordered).as_matrix())[0]
    assert abs(E_stacked - E_reordered) < 1e-10
