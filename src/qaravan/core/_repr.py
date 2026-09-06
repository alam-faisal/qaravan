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
from contextlib import contextmanager


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


TERM_LIMIT = 10
"""Terms a repr shows before eliding. None shows everything; see term_limit()."""

_DEFAULT = object()


def plain_more(count: int) -> str:
    return f"... ({count} more)"


def math_more(count: int) -> str:
    """\\text{} wrapped -- raw '... (4 more)' in math mode typesets as italic variables."""
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


def math_body(part) -> str:
    """LaTeX body of one part, undelimited. A str becomes \\text{...}; anything with
    _repr_latex_ contributes its own body; anything else falls back to \\text{repr}."""
    if isinstance(part, str):
        return rf"\text{{{part}}}"
    latex = getattr(part, "_repr_latex_", None)
    if latex is None:
        return rf"\text{{{part!r}}}"
    return latex().strip("$")


def display_math(*parts, sep: str = r"\quad ", limit=_DEFAULT) -> None:
    """Render strings and algebraic objects as ONE typeset line in Jupyter.

    The substitute for print() in a notebook: print writes to a stdout stream, which carries no
    MIME bundle and so can never typeset. display(a, b) does typeset, but emits one output block
    per argument and shows strings quoted.

        display_math("as written:", forward, "vs", backward)
    """
    from IPython.display import Math, display

    with term_limit(TERM_LIMIT if limit is _DEFAULT else limit):
        body = sep.join(math_body(part) for part in parts)
    display(Math(body))


def display_terms(operator, limit=_DEFAULT) -> None:
    """One typeset line per term, rather than one long sum -- readable for a 20-term model."""
    from IPython.display import Math, display

    shown = elide([math_body(term) for term in operator.terms], limit, math_more)
    for line in shown:
        display(Math(line))
