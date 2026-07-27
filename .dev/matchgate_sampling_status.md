# Matchgate sampling cross-check with Francesca — status

## 1. What this side-project is

Francesca has a classical algorithm for computing sampling probabilities / TVD for
**matchgate (free-fermion/Gaussian) circuits doped with a non-Gaussian "extent"
resource**, parametrized by an angle `theta` (consistent with qaravan's own
`algebraQ` "doped matchgate simulator" line of work: a matchgate circuit is
efficiently classically simulable on its own, and `theta` controls how much
non-Gaussian resource — "extent" — is injected via the input state). She needs an
independent numerical ground truth to validate that algorithm before it goes in a
paper/draft.

Your role has been to provide that ground truth using qaravan's exact MPS
simulator:

- She sends a circuit as a list of `(4×4 matchgate unitary U, qubit_i, qubit_j)`
  tuples (big-endian bit order), by pickle.
- You rebuild it in qaravan as `Gate("U", [i,j], U)` → `Circuit` → `MPSSim`,
  producing an exact `MPS` state (see `.dev/dev_sampling.ipynb`).
- You sample bitstrings from the MPS (`fast_measure`/`fast_sample`) and/or
  compute exact outcome probabilities (`mps.evaluate(basis)`), and compare
  against her simulator's output.

Progress so far, in order:
1. **n=4, single layer, 10k shots** — your MPS frequencies vs. her matchgate-sim
   frequencies agreed to within shot noise. First convention check passed
   (endian order matches, no correction needed).
2. **n=100, single circuit** — sanity check that qaravan's MPS pipeline holds up
   at scale (ran, bond dimension moderate, `evaluate()` used to cross-check her
   supplied sample frequencies against exact MPS probabilities for the same
   bitstrings — this cell in the notebook is unfinished, no summary statistic
   was written yet).
3. **Separately**, a big *random* (non-matchgate) 100-qubit brickwall circuit was
   run purely to stress-test the MPS sampler's performance/bond-dimension growth
   (`chi` maxed at 1744, ~15 min runtime) — this is not part of the Francesca
   exchange, just a scaling check on your own sampler.

Francesca has now run the full study herself for `n = 4, 8, 12, 16, 20, 24`, at
both **constant theta** and **theta scaled with n**, across multiple circuits
(different seeds), computing **TVD** between sampled and exact/predicted
distributions, and wants mean/variance of TVD per (n, theta-setting) — mirroring
plots you both already made for a related quantity `p_y` (the exact matchgate
outcome probability, presumably computed by her Gaussian/Pfaffian method). She
was able to compute this herself only up to **n=16** — n=20 and n=24 are where
her method presumably becomes too costly, which is where your MPS route is
supposed to take over.

## 2. What you need to do next

1. **Reply to the scheduling ask** — she asked for a catch-up Wed or Thu *next
   week* to review a near-final draft. Confirm a slot (her last message doesn't
   need this specifically, but it's still open from the prior message).
2. **Get the zip of circuits** she mentioned (one matchgate circuit per
   `(n, seed)`, for both constant-theta and theta-scaled-with-n cases, at
   n = 20 and 24).
3. **For each circuit**, repeat the `dev_sampling.ipynb` pipeline:
   - build the `Circuit`/`MPSSim` from her `(U, i, j)` tuples,
   - generate a large sample (`mps.fast_sample`, likely 10⁴–10⁵ shots — check
     what shot count she used for n≤16 so the noise floor is comparable),
   - get exact bitstring probabilities via `mps.evaluate()` where feasible, or
     work directly from the sampled frequencies if exact enumeration is too
     costly at n=20/24 (2²⁰–2²⁴ basis states — likely need TVD estimated from
     samples on both sides rather than full enumeration).
4. **Compute TVD** between your MPS-based distribution and hers for each
   `(n, seed, theta-setting)`, then aggregate **mean and variance across
   seeds** per `(n, theta-setting)` — same format as the existing `p_y` plots,
   so hers and yours can go on the same figure.
5. **Watch truncation**: at n=20–24 with generic (non-Gaussian-input) evolution
   the MPS bond dimension can grow fast (you saw χ=1744 at n=100 for a
   *random* circuit) — for extent-doped matchgate circuits the growth may or
   may not be that bad, but if you set `msvr` (truncation threshold) confirm
   it's tight enough that the truncation error doesn't itself contaminate the
   TVD you're using as ground truth. Worth reporting the achieved bond
   dimension alongside the TVD numbers.
6. **Send Francesca**: the TVD mean/variance table or plot for n=20, 24
   (both theta settings), ideally before the Wed/Thu meeting.

## Open uncertainty (flagging per your usual preference)

I'm inferring "extent" and the doped-matchgate framing from qaravan's own
`algebraQ` doped-matchgate simulator and your stated research interest in
classical-simulation boundaries — I don't have Francesca's paper/formula for
`p_y`, so I can't tell you precisely what she computes it from (Pfaffian of a
covariance matrix restricted to the extent decomposition, most likely, but
that's a guess). Worth confirming with her directly what `p_y` and her TVD
baseline actually are before you finalize a comparison method.

*(Update: `p_y` is just confirmed to mean P(measure bitstring y). She has multiple
estimators for it — truncated vs. unbiased — which matters below.)*

---

## 3. Concrete next steps, after unzipping and inspecting the data

The three zips were unzipped into `.dev/unzipped/{circuits,constant_theta,theta_scaling}/`
(scratch copies from this session — regenerate from the zips or delete freely).

### 3.1 What's actually in them

- **`circuits.zip` → `circuits/matchgates_list_n{20,24}_seed{12,36,89,1000,17895}.pkl`**
  10 files total (2 sizes × 5 seeds). Same format as the earlier n=4/n=100 circuits:
  a list of `(4×4 matchgate unitary, i, j)` tuples. n=20 has 95 gates, n=24 has 115 —
  both are exactly a **depth-5 nearest-neighbor brickwall** (`(n-1)*5` gates: even
  pairs then odd pairs, repeated 5 times), matching qaravan's own
  `brickwall_skeleton`.
  **Important: there is one circuit file per `(n, seed)`, not per `theta`.** The
  same circuit is reused for every theta value below — `theta` must therefore be
  parametrizing the **input state only**, applied before this circuit runs, exactly
  as Francesca said for the n=100 case ("the extent state with theta = pi/4").
  This is *not currently built anywhere in qaravan* — `MPSSim.initialize_state`
  only knows `None` (all-zero), an `MPS`, or a computational-basis bitstring
  ([mps_sim.py:12](src/qaravan/tensorQ/mps_sim.py#L12)) — so this is real code you
  need to write, not just a notebook.

- **`constant_theta.zip` → `samples_freq_dict_n{20,24}_theta{value}_seed{seed}.pkl`**
  100 files (2 sizes × 10 theta values × 5 seeds). `theta` runs `0` to `pi/4` in
  9 equal steps of `5°` (`0.000000, 0.087266, …, 0.785398`). Each file is a
  **raw integer count dict** (`{bitstring: count}`, big-endian, e.g.
  `'00000001110001110000'`), summing to exactly **100,000** shots per file.
  n=20 has ~48k unique outcomes out of 100k shots at theta=0 (so on that count
  alone, quite spread out already even at theta=0 — see 3.3).

- **`theta_scaling.zip` → `samples_freq_dict_n{20,24}_theta_1overn{,2}_value{value}_seed{seed}.pkl`**
  20 files (2 sizes × {1/n, 1/n²} × 5 seeds). Same count-dict format, 100,000
  shots each. Values confirm the scaling exactly: n=20 → 1/n=0.05, 1/n²=0.0025;
  n=24 → 1/n=0.041667, 1/n²=0.001736.

### 3.2 Feasibility check (already done, good news)

Ran the two n=20/n=24 seed=12 circuits through `MPSSim` with **no truncation**
(`msvr=None`, exact) and default all-zero input:

| n  | gates | max bond dim χ | runtime |
|----|-------|-----------------|---------|
| 20 | 95    | 182             | 0.5 s   |
| 24 | 115   | 148             | 0.5 s   |

This is nothing like the earlier 100-qubit *fully random* brickwall test
(χ=1744, 15 min) — depth is only 5 here, so the light cone caps entanglement
growth hard, not the system size. **Exact (untruncated) MPS simulation at
n=20/24 is cheap**, so there's no truncation-error concern to manage. Evaluating
`mps.evaluate(y)` for all ~48k of Francesca's observed bitstrings for one file
took ~7s. With 120 files total (100 + 20), a full sweep is a **~15–25 minute
script**, not a big job.

### 3.3 A finding that changes the plan: don't do naive sample-vs-sample TVD

Since MPS gives you the *exact* probability of any given computational-basis
string cheaply, the right comparison is **not** "sample from MPS and compare
two empirical frequency dicts" — it's: for every bitstring `y` Francesca
observed, compute her empirical frequency `count[y]/100000` against your exact
`p_y = mps.evaluate(y)`. No MPS sampling noise at all on your side.

Pilot test: `n=20, seed=12, theta=0`, assuming literal `|0…0>` input (the only
theta=0-safe assumption) and her big-endian bit order directly (not reversed):

- Her top 5 most-sampled bitstrings vs. your exact `p_y` agree to within
  ~20–30% relative, e.g. `00000100011000110100`: hers `0.00084`, yours
  `0.000736`; `00000100011000101100`: hers `0.00074`, yours `0.000743`. Reversing
  bit order kills this agreement completely (probabilities drop to ~1e-6) — so
  **big-endian, non-reversed, all-zero-input is the right convention for
  theta=0**, consistent with the earlier n=4/n=100 checks.
- But the naive plug-in TVD over her full observed support came out to **0.39**,
  with only 69% of your predicted probability mass landing on the ~48k
  bitstrings she happened to sample.

That's not evidence of a mismatch — it's what you'd expect when a distribution
has enormous effective support (up to 2²⁰ outcomes) and you only have 100k
shots: most of the true probability mass sits on outcomes that were never
drawn, and a naive `0.5·Σ|freq - p|` estimator conflates that "unseen mass"
with real disagreement. **This is exactly the truncated-vs-unbiased-estimator
distinction Francesca mentioned for `p_y`.** Before running the full sweep,
ask her which TVD estimator she actually uses (Good-Turing-style unseen-mass
correction? Something using her own p_y truncation order?) so your numbers
are comparable to her n≤16 results, not a different statistic that happens to
share the name "TVD."

### 3.4 Pipeline (pseudocode, real paths)

```python
import pickle, re
from qaravan.core import Circuit, Gate
from qaravan.tensorQ import MPSSim

CIRCUITS_DIR = ".dev/unzipped/circuits/circuits"
CONST_THETA_DIR = ".dev/unzipped/constant_theta/constant_theta"
THETA_SCALING_DIR = ".dev/unzipped/theta_scaling/theta_scaling"

def load_circuit(n, seed):
    with open(f"{CIRCUITS_DIR}/matchgates_list_n{n}_seed{seed}.pkl", "rb") as f:
        mat_list = pickle.load(f)
    gate_list = [Gate("U", [i, j], U) for U, i, j in mat_list]
    return Circuit(gate_list=gate_list, n=n)

def prepare_extent_state(n, theta):
    """BLOCKED: confirm exact construction with Francesca.
    theta=0 must reduce to all-zero (validated above). Working guess for
    theta>0: pairwise Givens rotation on qubits (2k, 2k+1),
    |00> -> cos(theta)|00> + sin(theta)|11>, applied to all n/2 pairs before
    the main circuit — this is itself a matchgate, so it composes cleanly
    with MPSSim. UNVERIFIED — do not trust results built on this until she
    confirms or you validate against a couple of her theta>0 files.
    """
    prep_gates = [...]  # one matchgate per qubit pair, parametrized by theta
    return Circuit(gate_list=prep_gates, n=n)

def mps_probs_on_support(n, seed, theta, support):
    circ = prepare_extent_state(n, theta) + load_circuit(n, seed)
    sim = MPSSim(circ)  # msvr=None: exact, chi stays small (see 3.2)
    sim.run()
    mps = sim.state
    return {y: mps.evaluate([int(b) for b in y]) for y in support}

def load_counts(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # {bitstring: raw count}, sums to 100_000

def tvd(counts, p_mps):
    """PLACEHOLDER estimator — swap for whatever Francesca uses (3.3)."""
    total = sum(counts.values())
    seen_mass = sum(p_mps.values())
    return 0.5 * (sum(abs(counts[y] / total - p_mps[y]) for y in counts)
                  + (1 - seen_mass))

THETA_SCALING_RE = re.compile(
    r"samples_freq_dict_n(\d+)_theta_(1overn2?)_value([\d.]+)_seed(\d+)\.pkl"
)

# --- constant-theta sweep ---
results = []
for n in (20, 24):
    for theta_str in ("0p000000", "0p087266", ..., "0p785398"):
        theta = float(theta_str.replace("p", "."))
        for seed in (12, 36, 89, 1000, 17895):
            path = f"{CONST_THETA_DIR}/samples_freq_dict_n{n}_theta{theta_str}_seed{seed}.pkl"
            counts = load_counts(path)
            p_mps = mps_probs_on_support(n, seed, theta, counts.keys())
            results.append((n, theta, seed, tvd(counts, p_mps), max(mps_get_max_dim_used)))

# aggregate mean/var of tvd across seeds, per (n, theta) -> mirrors her p_y plots
```

(`theta_scaling` files follow the same loop, parsed with `THETA_SCALING_RE` to
recover `n`, the `1/n` vs `1/n²` tag, and the numeric theta value for the x-axis.)

### 3.5 Concrete to-do list, in order

1. **Ask Francesca** (before writing more code): (a) the exact extent-state
   construction as a function of theta, (b) her TVD estimator (naive plug-in
   vs. unseen-mass-corrected vs. something tied to her truncated/unbiased
   `p_y`), (c) confirm Wed/Thu next week for the catch-up.
2. Implement `prepare_extent_state` for real once (a) is confirmed; sanity
   check it reduces to all-zero at theta=0 and reproduces the theta=0 pilot
   agreement above.
3. Implement the shared `tvd` estimator matching (b).
4. Run the full sweep (constant_theta: 100 files, theta_scaling: 20 files) —
   budget ~20–30 min based on the per-file timings in 3.2.
5. Aggregate mean/variance of TVD across the 5 seeds for each `(n, theta)` and
   `(n, scaling-law)` point; report achieved max bond dimension alongside each
   (should stay small/untruncated per 3.2 — flag any point where it's not).
6. Plot in the same style as the existing p_y mean/variance-vs-n(or theta)
   plots so hers (n≤16) and yours (n=20,24) sit on one figure; send before the
   meeting.

## 4. First full sweep — done, provisional (see `dev_sampling.ipynb`)

Ran the above end to end with the Givens-rotation guess and the naive plug-in
estimator (steps 2–4 above, using the *guessed* ansatz — step 1 with Francesca
still hasn't happened). All 120 files processed exactly, all bond dimensions
stayed small (no truncation needed, confirming 3.2 held up at full scale).
Results:

| n  | TVD @ theta=0    | TVD @ theta=pi/4 |
|----|------------------|-------------------|
| 20 | 0.477 ± 0.050    | 0.994 ± 0.001     |
| 24 | 0.816 ± 0.029    | 1.000 ± 0.000     |

**New finding that changes priority #1 above:** theta=0 needs no ansatz at all
(trivial all-zero input either way), so it's a clean read on the *estimator*
alone — and it already shows n=24's baseline TVD (0.816) is far worse than
n=20's (0.477). Cause: at theta=0, 89% of n=24's 100,000 shots are already
unique bitstrings vs. 59% for n=20 (bigger Hilbert space, not a more spread
distribution), so the plug-in estimator's "unseen mass" term inflates with n
almost mechanically. **This means these plug-in TVD numbers are not
comparable across n, and probably not comparable to Francesca's own n≤16
figures either.** Getting her actual estimator (§3.3/3.5 step 1b) is now the
higher-priority of the two open questions — without it, no number in the
table above should be read as a real accuracy statement, even the theta=0
column.

The theta>0 trend is smooth in both n (TVD grows monotonically with theta,
support size shrinks monotonically) — consistent with either "the ansatz is
qualitatively right" or "it's wrong but happens to vary smoothly with theta."
Cannot be resolved without Francesca's formula (§3.5 step 1a).

## 5. Extent-state formula confirmed; TVD replaced with an XEB fidelity

Faisal confirmed the real extent-state construction:

$$\left(\cos\theta\,|0000\rangle + \sin\theta\,|1111\rangle\right)^{\otimes n/4}$$

— a tensor product of 4-qubit GHZ-like "cat" states, *not* the pairwise
Givens-rotation guess from §3–4. This state is genuinely non-Gaussian for
generic theta (a Gaussian state from vacuum can't skip the 2-particle sector
the way this skips 0→4), consistent with it being the paper's magic/extent
resource. Implemented in the notebook with one `RY` + three `CNOT`s per
block; verified exactly against analytic amplitudes for n=4, n=8.

Separately, diagnosed *why* even the theta=0 TVD (no ansatz ambiguity at all)
came out so large: the naive plug-in TVD's sample complexity scales with the
*support size* of the true distribution (up to 2²³ here), not the 100,000
shot count, so it's dominated by "mass never sampled" rather than real
disagreement — not fixable by getting the ansatz right.

Replaced it with a linear cross-entropy (XEB) fidelity, adapted from the
RCS-verification literature but *not* assuming Porter-Thomas statistics
(matchgate circuits are integrable, not chaotic, so that assumption doesn't
hold here — both anchor points are computed directly instead of assumed):

$$
F_{XEB} = \frac{C-U}{S_Q-U}, \qquad C=\tfrac1N\textstyle\sum_i Q(x_i), \qquad
S_Q=\textstyle\sum_y Q(y)^2, \qquad U=\tfrac1D
$$

$Q$ = our exact MPS distribution; $x_i$ = Francesca's samples; $S_Q$ estimated
via Monte Carlo sampling from our own MPS (`mps.fast_sample`); $D=2^{n-1}$,
the even-parity subspace (matchgates conserve global fermion parity, and the
extent state only ever populates weight-0/4-per-block terms, both even —
confirmed from the matrices themselves being exactly block-diagonal on
{00,11}/{01,10}). Unlike TVD, this is a plain Monte Carlo average, so its
error shrinks as $1/\sqrt N$ regardless of support size — the falsifiable
prediction is $F_{XEB}\to1$ at theta=0, where both simulators should be
exact; if not, that's a real bug, not an estimator artifact.

`dev_sampling.ipynb` now computes both TVD (kept for comparison) and
$F_{XEB}$ side by side for the constant-theta sweep only — the
theta-scaled-with-n runs were dropped from the notebook per Faisal's request
(not needed right now; revisit §3.1/3.5 if that changes). Code is written,
not yet executed by Claude — Faisal is running it himself.
