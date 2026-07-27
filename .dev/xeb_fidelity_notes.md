# The XEB fidelity used in `dev_sampling.ipynb`

## 1. The problem it replaces

The naive plug-in TVD estimator (`0.5 * (sum |freq - p| over observed support + unseen mass)`)
needs about as many shots as the *support size* of the true distribution to be
trustworthy. With up to $2^{23}$ reachable bitstrings and only 100,000 shots
from Francesca, it was dominated by "probability mass on bitstrings we never
even sampled" rather than real disagreement — which is why it read ~0.48-0.82
even at theta=0, where both simulators are supposed to be exact. See
`matchgate_sampling_status.md` secs 3-4 for the full history.

## 2. Definition

Let $Q$ be our exact MPS distribution over bitstrings (probabilities are
computable one at a time via `mps.evaluate`), and let $P$ be whatever true
process Francesca is sampling from — we only ever see it through her
$N=100{,}000$ samples $x_1,\dots,x_N$ (as raw counts, per bitstring).

$$
F_{XEB} = \frac{C - U}{S_Q - U}
$$

where

$$
C = \frac{1}{N}\sum_{i=1}^N Q(x_i), \qquad
S_Q = \sum_y Q(y)^2 = \mathbb E_{y\sim Q}[Q(y)], \qquad
U = \frac{1}{D}.
$$

- **$C$** ("cross-entropy"): average of $Q$ evaluated on Francesca's *actual*
  samples, weighted by her raw counts. This reuses the exact same `Q(x_i)`
  values already computed for the old TVD estimator — no extra MPS
  evaluation cost.
- **$S_Q$** ("self-overlap" / collision probability of $Q$ in the
  computational basis): since $Q$ is *our own* distribution and fully known,
  we estimate this by drawing $M$ samples from our own MPS
  (`mps.fast_sample`) and averaging $Q$ over those — a pure self-consistency
  calculation, doesn't touch Francesca's data at all. Currently $M=300$
  (`XEB_SELF_SAMPLES`).
- **$U$**: the cross-entropy you'd get if Francesca's samples were actually
  uniform noise over the $D$ reachable basis states — exact, no estimation
  needed.
- **$D = 2^{n-1}$**, not $2^n$: matchgates conserve global fermion parity
  (confirmed directly — her matrices are exactly block-diagonal on
  $\{00,11\}$/$\{01,10\}$), and the extent state
  $(\cos\theta|0000\rangle+\sin\theta|1111\rangle)^{\otimes n/4}$ only ever
  populates weight-0 and weight-4-per-block terms (both even), so the whole
  circuit stays confined to the even-weight half of the Hilbert space.

## 3. Where the formula comes from

Assume (as a *model*, not a proven fact) that Francesca's true distribution
is a mixture of our reference $Q$ and pure noise:

$$
P \;=\; F\cdot Q \;+\; (1-F)\cdot \text{Uniform}_D
$$

Take the expectation of $Q(x)$ under this assumed $P$:

$$
C = \mathbb E_{x\sim P}[Q(x)] = F\underbrace{\mathbb E_{x\sim Q}[Q(x)]}_{S_Q} + (1-F)\underbrace{\mathbb E_{x\sim \text{Uniform}}[Q(x)]}_{U}
$$

Solve for $F$: $\;F = \dfrac{C-U}{S_Q-U}$. That's the whole derivation — $F$ is
whatever value makes the mixture model match the observed cross-entropy $C$.

This is the same algebraic move behind the standard linear-XEB fidelity used
to verify random circuit sampling experiments (Arute et al. 2019,
$F=2^n\langle p\rangle-1$). The published version additionally **assumes
Porter-Thomas statistics** for $Q$ (i.e. $Q$ behaves like a Haar-random
circuit's output), which forces $S_Q\approx 2/D$ and collapses the formula to
that single universal expression. **We don't make that assumption here** —
matchgate circuits are integrable/free-fermion, not chaotic, so Porter-Thomas
doesn't apply, and $S_Q$ is measured directly by Monte Carlo instead of
assumed. This is the more correct choice for this problem, at the cost of
needing that extra `self_overlap` sampling step.

## 4. Why this fixes the TVD problem

$C$ and $S_Q$ are both **plain Monte Carlo averages** of a bounded quantity.
Their statistical error shrinks as $1/\sqrt N$ (resp. $1/\sqrt M$)
**regardless of how large the support of $P$ or $Q$ is** — unlike TVD, which
needs the sample count to be comparable to the support size. That's the
entire point of switching metrics.

## 5. Is $F_{XEB}$ guaranteed to be between 0 and 1? **No.**

This is the important caveat, and it's the direct answer to "why am I seeing
values greater than 1":

**Mathematically**, $F_{XEB}\in[0,1]$ only holds if $P$ really is an exact
convex mixture of $Q$ and the uniform distribution, for some $F\in[0,1]$.
Nothing forces the actual relationship between our MPS and Francesca's true
process to take that specific form — $F_{XEB}$ is a *projection* onto a
one-parameter noise model, not a norm-bounded quantity like true state
fidelity. If $P$ happens to put more weight than $Q$ itself does on $Q$'s own
highest-probability outcomes (entirely possible for distributions that don't
fit the assumed mixture form), $C$ can exceed $S_Q$ and $F_{XEB}>1$ pops out
algebraically.

**Statistically**, even when the true value of $F$ really is exactly 1 (which
is precisely the expectation at theta=0, where both simulators should be
exact), our *estimate* is built from noisy Monte Carlo means ($N=100{,}000$
for $C$, only $M=300$ for $S_Q$). A value sitting exactly at a boundary,
measured with noise, lands above the boundary about half the time by pure
chance — same phenomenon as measuring a probability of exactly 1 with a noisy
instrument and sometimes reading 1.02. **Getting $F_{XEB}$ slightly above 1 at
theta=0 is not a bug — if anything it's consistent with (not contradicting)
the prediction that the pipeline is correct there.**

## 6. The `1e-12` you're seeing is a plotting artifact, not a result

The bottom panel plots $1-F_{XEB}$ on a **log scale**, and log of a negative
number errors out, so the plotting code clips with `max(1 - F_XEB, 1e-12)`.
When $F_{XEB}>1$ (negative infidelity), this floor makes the point look like
"agreement to 1 part in $10^{12}$" — which is not remotely justified by the
actual precision of a $M=300$-sample Monte Carlo estimate (realistically a
few percent relative error on $S_Q$, propagating into $F_{XEB}$). **Don't
read those clipped points as a precision claim** — they just mean "$F_{XEB}$
came out at or above 1, i.e. consistent with perfect agreement within this
estimator's noise."

Fixes worth making later (not yet applied):
- Plot $|1-F_{XEB}|$ instead of clamping, so points above/below 1 are at
  least visually distinguishable (e.g. different marker or color) rather
  than silently floored.
- Add real error bars on $F_{XEB}$ propagated from the $S_Q$ Monte Carlo
  variance (not just the 5-seed spread across circuit instances), so a point
  near 1 visibly shows a noise band straddling both sides.
- Since the sweep is now fast (~20 min at $M=300$), consider bumping
  `XEB_SELF_SAMPLES` specifically for the theta=0 and small-theta points,
  where nailing down "does this really validate the pipeline" matters most.
