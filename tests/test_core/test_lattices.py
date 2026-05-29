"""Tests for LinearLattice."""

from qaravan.core.lattices import LinearLattice


def test_linear_lattice_nn_pairs_open():
    lat = LinearLattice(4, periodic=False)
    assert lat.nn_pairs() == [(0, 1), (1, 2), (2, 3)]


def test_linear_lattice_nn_pairs_periodic():
    lat = LinearLattice(4, periodic=True)
    assert lat.nn_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_linear_lattice_n2_open():
    lat = LinearLattice(2, periodic=False)
    assert lat.nn_pairs() == [(0, 1)]


def test_linear_lattice_n3_periodic():
    lat = LinearLattice(3, periodic=True)
    assert lat.nn_pairs() == [(0, 1), (1, 2), (2, 0)]
