"""Formatting helpers for construction-style and mathematical reprs."""

from __future__ import annotations
from contextlib import contextmanager


def format_coefficient(
    coeff: complex, spec: str = ".4g", threshold: float = 1e-9
) -> str:
    """Format as real, imaginary, or parenthesized Cartesian form without erasing small values."""
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


TERM_LIMIT = 10
"""Terms a repr shows before eliding. None shows everything; see term_limit()."""

_DEFAULT = object()


def plain_more(count: int) -> str:
    return f"... ({count} more)"


def math_more(count: int) -> str:
    """Wrap an elision marker in LaTeX text mode."""
    return rf"\text{{... ({count} more)}}"


def elide(parts: list[str], limit=_DEFAULT, marker=plain_more) -> list[str]:
    """First `limit` entries, then how many were dropped. limit=None keeps all of them."""
    if limit is _DEFAULT:
        limit = TERM_LIMIT
    if limit is None or len(parts) <= limit:
        return parts
    return [*parts[:limit], marker(len(parts) - limit)]


@contextmanager
def term_limit(limit: int | None):
    """Temporarily change how many terms reprs show. term_limit(None) shows everything."""
    global TERM_LIMIT
    previous, TERM_LIMIT = TERM_LIMIT, limit
    try:
        yield
    finally:
        TERM_LIMIT = previous


def named(name: str, indices: list[int]) -> str:
    """The 'H([0])' form shared by Gate and Observable."""
    return f"{name}({indices})"


# === LaTeX ===

# In _repr_latex_ output, write c^{\dagger} (not "†") and brace labels (_{10}, not _10).


def as_math(body: str) -> str:
    """Wrap a LaTeX body for Jupyter's _repr_latex_."""
    return f"${body}$"


def latex_coefficient(coeff: complex, spec: str = ".4g") -> str:
    """Format a coefficient with i as the imaginary unit and implicit unit magnitude."""
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


def math_body(part) -> str:
    """Return an undelimited LaTeX body, using _repr_latex_ when available."""
    if isinstance(part, str):
        return rf"\text{{{part}}}"
    latex = getattr(part, "_repr_latex_", None)
    if latex is None:
        return rf"\text{{{part!r}}}"
    return latex().strip("$")


def display_math(*parts, sep: str = r"\quad ", limit=_DEFAULT) -> None:
    """Render strings and algebraic objects in one Jupyter math block."""
    from IPython.display import Math, display

    with term_limit(TERM_LIMIT if limit is _DEFAULT else limit):
        body = sep.join(math_body(part) for part in parts)
    display(Math(body))


def display_terms(operator, limit=_DEFAULT) -> None:
    """Display each retained operator term on its own typeset line."""
    from IPython.display import Math, display

    shown = elide([math_body(term) for term in operator.terms], limit, math_more)
    for line in shown:
        display(Math(line))
