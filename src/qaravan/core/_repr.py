"""Shared __repr__ helpers, plus the conventions they encode.

Imports nothing from qaravan, deliberately: base.py is the root ABC module and cannot import
observables.py, so these cannot live beside PauliString. Pass in plain values, get strings back.

Conventions:
  - Define __repr__, never __str__ -- Python falls back str() -> repr() on its own.
  - Algebraic types render as mathematics (PauliSum, FermionicSum, Statevector); structural
    types render as construction (Gate, Circuit, FermionicEncoding). A container of algebra
    prints a header, then the algebra.
  - Any repr over a sequence elides.
"""

from __future__ import annotations


def format_coefficient(
    coeff: complex, spec: str = ".4g", threshold: float = 1e-9
) -> str:
    """Real-only, imaginary-only, or (r+ij).

    spec=".4g" suits operator coefficients, whose magnitude is unbounded -- ".4f" would render
    1e-7 as 0.0000. Amplitudes are bounded by 1 and read better fixed-width, so they pass ".4f".
    """
    real, imag = coeff.real, coeff.imag
    if abs(imag) < threshold:
        return f"{real:{spec}}"
    if abs(real) < threshold:
        return f"{imag:{spec}}j"
    return f"({real:{spec}}{imag:+{spec}}j)"


def join_terms(parts: list[str]) -> str:
    """Join a sum, folding a leading '-' into ' - ' so it never reads 'a + -b'."""
    if not parts:
        return "0"
    joined = parts[0]
    for part in parts[1:]:
        joined += f" - {part[1:]}" if part.startswith("-") else f" + {part}"
    return joined


def elide(parts: list[str], limit: int = 10) -> list[str]:
    """First `limit` entries, then how many were dropped."""
    if len(parts) <= limit:
        return parts
    return [*parts[:limit], f"... ({len(parts) - limit} more)"]


def named(name: str, indices: list[int]) -> str:
    """The 'H([0])' form shared by Gate and Observable."""
    return f"{name}({indices})"


# ---------------------------------------------------------------------------
# LaTeX
#
# _repr_latex_ is what Jupyter calls to render an object as mathematics, so devbook cells
# typeset operators instead of printing ASCII. Two details that are easy to get wrong:
# write c^{\dagger}, since a literal "†" has no superscript structure to render; and brace
# every label (_{10}, not _10) or only its first character ends up subscripted.
# ---------------------------------------------------------------------------


def as_math(body: str) -> str:
    """Wrap a LaTeX body for Jupyter's _repr_latex_."""
    return f"${body}$"


def latex_coefficient(coeff: complex, spec: str = ".4g") -> str:
    """format_coefficient with i for the imaginary unit, and a unit magnitude left implicit
    (i, not 1i), as mathematics is written."""
    text = format_coefficient(coeff, spec).replace("j", "i")
    if text.startswith("1i"):
        return "i" + text[2:]
    if text.startswith("-1i"):
        return "-i" + text[3:]
    return text


def latex_pauli_string(string: str) -> str:
    """'IZXI' -> 'Z_{1} X_{2}'. Identities are dropped; all-identity renders as the symbol."""
    factors = [f"{char}_{{{site}}}" for site, char in enumerate(string) if char != "I"]
    return " ".join(factors) if factors else r"\mathbb{1}"


def latex_ladder_ops(ops: list[tuple[int, bool]]) -> str:
    """[(0, True), (1, False)] -> 'c^{\\dagger}_{0} c_{1}'."""
    factors = [
        f"c^{{\\dagger}}_{{{mode}}}" if is_dagger else f"c_{{{mode}}}"
        for mode, is_dagger in ops
    ]
    return " ".join(factors) if factors else r"\mathbb{1}"


def latex_majorana_ops(ops: list[int]) -> str:
    """Index 2m is \\gamma_m, 2m+1 is \\bar{\\gamma}_m."""
    factors = []
    for index in ops:
        symbol = r"\bar{\gamma}" if index % 2 else r"\gamma"
        factors.append(f"{symbol}_{{{index // 2}}}")
    return " ".join(factors) if factors else r"\mathbb{1}"


def latex_term(coeff: complex, operator: str, spec: str = ".4g") -> str:
    """Coefficient and operator. A coefficient of +-1 leaves only its sign, as written."""
    if coeff == 1.0:
        return operator
    if coeff == -1.0:
        return f"-{operator}"
    return f"{latex_coefficient(coeff, spec)} \\, {operator}"
