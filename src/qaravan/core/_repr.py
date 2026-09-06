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
