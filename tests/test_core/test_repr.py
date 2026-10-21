"""Repr conventions: no default reprs, no '+ -', elision, and the LaTeX layer."""

import numpy as np
import pytest
from qaravan.applications import trotter_circuit
from qaravan.backends.statevector import Statevector
from qaravan.core._repr import elide, format_coefficient, join_terms, latex_coefficient
from qaravan.core.base import Gate, Observable
from qaravan.core.fermions import (
    FermionicSum,
    MajoranaSum,
    c,
    cdag,
    hopping,
    majorana_bilinear,
)
from qaravan.core.gates import RX, H
from qaravan.core.hamiltonians import TFI, Heisenberg1D
from qaravan.core.observables import LocalOp, PauliString, PauliSum


def _concrete(base):
    """Every instantiable subclass, transitively."""
    found = []
    for cls in base.__subclasses__():
        found.append(cls)
        found.extend(_concrete(cls))
    return [cls for cls in found if not getattr(cls, "__abstractmethods__", None)]


# ---------------------------------------------------------------------------
# Every class has a repr -- the LocalOp defect
# ---------------------------------------------------------------------------


def test_local_op_has_a_repr():
    """LocalOp had none: Observable defined __str__ but not __repr__, unlike Gate."""
    assert "object at 0x" not in repr(LocalOp(np.eye(2), [0]))


@pytest.mark.parametrize("base", [Observable, Gate])
def test_no_subclass_falls_back_to_the_default_repr(base):
    """Guards the hole rather than the one class that fell into it."""
    for cls in _concrete(base):
        assert cls.__repr__ is not object.__repr__, cls.__name__


def test_str_and_repr_agree():
    """__repr__ is the only one defined; str() falls back to it."""
    for obj in [
        H(0),
        RX(0, 0.5),
        PauliString("XZ", -0.5),
        TFI(3),
        Statevector(bitstring="01"),
    ]:
        assert str(obj) == repr(obj), type(obj).__name__


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_pauli_string_repr_has_no_doubled_parens_or_phantom_imaginary():
    assert repr(PauliString("XZ", -0.5)) == "-0.5 XZ"
    assert repr(PauliString("XZ")) == "XZ"


def test_format_coefficient_keeps_small_magnitudes():
    """.4f would flatten 1e-7 to 0.0000; operator coefficients are unbounded."""
    assert format_coefficient(1e-7) == "1e-07"
    assert format_coefficient(1e-7, spec=".4f") == "0.0000"
    assert format_coefficient(2 + 0j) == "2"
    assert format_coefficient(2j) == "2j"
    assert format_coefficient(1 + 2j) == "(1+2j)"


@pytest.mark.parametrize(
    "op",
    [
        PauliSum([PauliString("ZI", -0.5), PauliString("IZ", 0.5)]),
        FermionicSum(hopping(0, 1, -1.0), 2),
        MajoranaSum([-1.0 * majorana_bilinear(0, 3)], 2),
        Statevector(array=np.array([0.6, -0.8], dtype=complex)),
    ],
)
def test_sums_never_emit_plus_minus(op):
    assert " + -" not in repr(op)
    assert " + -" not in op._repr_latex_()


def test_majorana_repr_decodes_mode_and_parity():
    text = repr(majorana_bilinear(0, 3))
    assert "γ_0" in text and "γ̄_1" in text and "γ_3" not in text


def test_long_sequences_elide():
    circ = trotter_circuit(Heisenberg1D(12), dt=0.01, order=2) * 20
    text = repr(circ)
    assert len(text) < 500, len(text)
    assert "more)" in text


def test_elide_and_join_helpers():
    assert elide(["a", "b", "c"], limit=2) == ["a", "b", "... (1 more)"]
    assert join_terms(["a", "-b", "c"]) == "a - b + c"
    assert join_terms([]) == "0"


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def test_latex_uses_backslash_dagger_not_the_literal_character():
    """A literal '†' has no superscript structure to render."""
    text = (cdag(0) @ c(1))._repr_latex_()
    assert text == r"$c^{\dagger}_{0} c_{1}$"
    assert "†" not in text


def test_latex_braces_every_label():
    """_10 would subscript only the '1'."""
    assert "_{10}" in PauliString("I" * 10 + "Z")._repr_latex_()


def test_latex_drops_identities_and_indexes_sites():
    assert PauliString("IZXI", -0.5)._repr_latex_() == r"$-0.5 \, Z_{1} X_{2}$"


def test_latex_leaves_unit_magnitudes_implicit():
    assert latex_coefficient(1j) == "i"
    assert (-1.0 * (cdag(0) @ c(1)))._repr_latex_() == r"$-c^{\dagger}_{0} c_{1}$"


def test_latex_majorana_symbols():
    assert (
        majorana_bilinear(0, 3)._repr_latex_() == r"$i \, \gamma_{0} \bar{\gamma}_{1}$"
    )


def test_hamiltonian_latex_names_the_model():
    assert TFI(3, h=0.5)._repr_latex_().startswith(r"$\text{TFI}_{3} = ")


def test_statevector_latex_renders_kets():
    assert r"\lvert 01 \rangle" in Statevector(bitstring="01")._repr_latex_()


# ---------------------------------------------------------------------------
# display helpers -- the substitute for print() in a notebook
# ---------------------------------------------------------------------------


def test_math_body_mixes_labels_and_operators():
    from qaravan.core._repr import math_body

    assert math_body("as written:") == r"\text{as written:}"
    assert math_body(cdag(0) @ c(1)) == r"c^{\dagger}_{0} c_{1}"
    assert math_body(True) == r"\text{True}"  # no _repr_latex_ -> plain fallback


def test_display_math_emits_one_block(monkeypatch):
    import IPython.display as ipd
    from qaravan.core._repr import display_math

    shown = []
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj.data))
    display_math("label:", cdag(0) @ c(1))
    assert len(shown) == 1
    assert shown[0] == r"\text{label:}\quad c^{\dagger}_{0} c_{1}"


def test_display_terms_emits_one_block_per_term(monkeypatch):
    import IPython.display as ipd
    from qaravan.core._repr import display_terms

    shown = []
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj.data))
    display_terms(FermionicSum(hopping(0, 1, -1.0), 2))
    assert len(shown) == 2


# ---------------------------------------------------------------------------
# Elision: math-safe marker, and caller control
# ---------------------------------------------------------------------------


def _long_sum():
    return PauliSum([PauliString("I" * i + "Z" + "I" * (14 - i)) for i in range(15)])


def test_latex_elision_marker_is_text_not_math():
    """Raw '... (4 more)' in math mode typesets as italic variables with the space lost."""
    latex = _long_sum()._repr_latex_()
    assert r"\text{... (5 more)}" in latex
    assert "+ ... (" not in latex


def test_term_limit_context_manager_controls_elision():
    from qaravan.core._repr import TERM_LIMIT, term_limit

    op = _long_sum()
    assert "more)" in repr(op)
    with term_limit(None):
        assert "more)" not in repr(op)
        assert "more" not in op._repr_latex_()
    with term_limit(3):
        assert "... (12 more)" in repr(op)
    from qaravan.core import _repr

    assert _repr.TERM_LIMIT == TERM_LIMIT  # restored on exit


def test_term_limit_restores_after_an_exception():
    from qaravan.core import _repr
    from qaravan.core._repr import term_limit

    before = _repr.TERM_LIMIT
    with pytest.raises(RuntimeError):
        with term_limit(2):
            raise RuntimeError("boom")
    assert _repr.TERM_LIMIT == before


def test_elide_limit_none_keeps_everything():
    assert elide(["a", "b", "c"], limit=None) == ["a", "b", "c"]
    assert elide(["a", "b", "c"], limit=2) == ["a", "b", "... (1 more)"]


def test_display_math_limit_is_per_call(monkeypatch):
    import IPython.display as ipd
    from qaravan.core._repr import display_math

    shown = []
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj.data))
    display_math("all:", _long_sum(), limit=None)
    assert "more" not in shown[0]
