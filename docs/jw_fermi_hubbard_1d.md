# Jordan–Wigner Encoding of the 1D Fermi–Hubbard Model

*Both mode orderings, derived from first principles, with the Trotterization in each case.*

---

## 1. The model

On an open chain of $L$ sites with spin-$\tfrac12$ fermions, the Fermi–Hubbard Hamiltonian is

$$
H = -t \sum_{j=0}^{L-2} \sum_{\sigma\in\{\uparrow,\downarrow\}}
\left( c^\dagger_{j,\sigma} c_{j+1,\sigma} + \mathrm{h.c.} \right)
\;+\; U \sum_{j=0}^{L-1} n_{j,\uparrow}\, n_{j,\downarrow}.
\tag{1}
$$

Symbols: $t>0$ is the hopping amplitude; $U$ the on-site interaction; $c^\dagger_{j,\sigma},c_{j,\sigma}$ create/annihilate a fermion of spin $\sigma$ at site $j$; $n_{j,\sigma}=c^\dagger_{j,\sigma}c_{j,\sigma}$ is the number operator. The fermionic algebra is

$$
\{c_{a},c^\dagger_{b}\}=\delta_{ab}, \qquad \{c_a,c_b\}=0,
\tag{2}
$$

where $a,b$ are composite indices $(j,\sigma)$. There are $2L$ fermionic modes, hence $N=2L$ qubits after encoding.

The work splits into two independent choices that we keep conceptually separate:

1. **A linear ordering** of the $2L$ modes (this is where *interleaved* vs *stacked* lives).
2. **The Jordan–Wigner map** applied along that ordering.

The energy spectrum is invariant under (1); only the *qubit-space* structure (operator weights, entanglement, circuit layout) depends on the ordering. We return to this in §6.

### 1.1 Mode labels vs. qubit placement — two separate steps in code

"A linear ordering of the modes" is itself two independent decisions, and `core/fermions.py`
makes both of them literal, separate steps rather than one combined choice.

**Step A — mode labeling.** Fixed once, not a per-encoding choice. Before any qubit exists,
every physical fermion gets an integer label. For `FermiHubbard1D` this is fixed inside
`FermiHubbard1D._terms`:

$$
\mathrm{mode}(j,\sigma) = \sigma L + j, \qquad \sigma \in \{0=\uparrow,\ 1=\downarrow\}.
$$

So modes $0,\dots,L-1$ are *always* $\uparrow_0,\dots,\uparrow_{L-1}$ and modes $L,\dots,2L-1$
are *always* $\downarrow_0,\dots,\downarrow_{L-1}$ — this labeling never changes, regardless of
"stacked" vs "interleaved" below. It is arbitrary (any bijection between $(j,\sigma)$ pairs and
$\{0,\dots,2L-1\}$ would do), but it has to be fixed once so that `hopping(a, b, coeff)` /
`number(mode)` calls refer to definite physical fermions.

**Step B — mode order.** The actual per-encoding choice, `FermionicEncoding.mode_order`.
`mode_order[qubit] = mode` assigns each already-labeled mode to a qubit. §3 and §4 below are
exactly the two choices of this permutation:

- **Stacked** = the identity permutation. Because Step A already numbers $\uparrow$ before
  $\downarrow$, doing nothing (`mode_order=None`) already gives the register
  $[\uparrow_0,\dots,\uparrow_{L-1},\downarrow_0,\dots,\downarrow_{L-1}]$ of §4 for free.
- **Interleaved** = `mode_order[2j] = j` (mode $\uparrow_j$), `mode_order[2j+1] = L+j` (mode
  $\downarrow_j$) — implemented as `interleaved_mode_order(L)`.

The reason stacked is the "free" (identity) case is entirely a consequence of Step A's arbitrary
choice ($\sigma L + j$, $\uparrow$ before $\downarrow$) — labeling modes as $2j+\sigma$ instead
would make interleaved the identity and stacked the permutation, with no physical difference
whatsoever. Neither the spectrum (eq. 20) nor any observable can depend on either step; both are
unitary relabelings of the same Hilbert space.

In code: `FermiHubbard1D` performs Step A internally and is otherwise bare — no qubits at all
exist until `.encode(...)` is called. Step B is supplied explicitly and separately, as a
`FermionicEncoding(name="jordan_wigner", num_modes=2*L, mode_order=...)`, threaded through both
the Hamiltonian and any observable evaluated against its ground state, so the two can't
independently drift onto different encodings.

Step A is only a real decision when a model has more than one fermionic mode per site — spin
here, but equally orbitals, bath levels, etc. For a spinless, single-orbital chain, `mode = site`
trivially and there is nothing to fix: `num_modes = num_sites`, and only Step B (which qubit
gets which site) remains a choice at all.

---

## 2. The Jordan–Wigner map and its one nontrivial identity

Label the chosen modes $k=0,\dots,2L-1$. JW maps

$$
c_k = \mathbf{Z}_{<k}\,\frac{X_k+iY_k}{2}, \qquad
c^\dagger_k = \mathbf{Z}_{<k}\,\frac{X_k-iY_k}{2}, \qquad
\mathbf{Z}_{<k}\equiv \prod_{m=0}^{k-1} Z_m ,
\tag{3}
$$

with $\mathbf{Z}_{<0}\equiv I$. The string $\mathbf{Z}_{<k}$ enforces anticommutation across modes. The number operator is string-free:

$$
n_k = c^\dagger_k c_k = \frac{X_k-iY_k}{2}\cdot\frac{X_k+iY_k}{2} = \frac{I-Z_k}{2}.
\tag{4}
$$

(Convention: $Z_k=+1$ is unoccupied, $Z_k=-1$ occupied; $c^\dagger_k=(X_k-iY_k)/2=|1\rangle\langle0|_k$ raises occupation.)

**The string-cancellation identity.** Consider a bilinear $c^\dagger_a c_b$ with $a<b$. The strings overlap on qubits $0,\dots,a-1$ (and square to $I$), leaving the segment $Z_a Z_{a+1}\cdots Z_{b-1}$. The $Z_a$ is absorbed by the on-site factor of $c^\dagger_a$ via

$$
\frac{X-iY}{2}\,Z = |1\rangle\langle0|\,Z = |1\rangle\langle0| = \frac{X-iY}{2},
\tag{5}
$$

so a $Z$-string of length $(b-a-1)$ survives **on the intervening qubits** $a+1,\dots,b-1$:

$$
c^\dagger_a c_b + \mathrm{h.c.}
= \tfrac12\Big( X_a\,\mathbf{Z}_{(a,b)}\,X_b + Y_a\,\mathbf{Z}_{(a,b)}\,Y_b \Big),
\qquad \mathbf{Z}_{(a,b)}\equiv\!\!\prod_{m=a+1}^{b-1}\!\!Z_m .
\tag{6}
$$

Two consequences drive everything below:

- **Adjacent modes** ($b=a+1$): the string is empty, giving a clean weight-2 term $\tfrac12(X_aX_{a+1}+Y_aY_{a+1})$.
- **Modes separated by one spectator** ($b=a+2$): a single residual $Z$ remains, giving weight-3 $\tfrac12(X_a Z_{a+1} X_{a+2}+Y_aZ_{a+1}Y_{a+2})$.

The interaction never produces a string, since $n_k=(I-Z_k)/2$ is already diagonal.

---

## 3. Interleaved ordering

Interleave the spin species so each real-space site occupies two consecutive qubits:

$$
k=\begin{cases} 2j & (j,\uparrow)\\ 2j+1 & (j,\downarrow)\end{cases},
\qquad j=0,\dots,L-1.
\tag{7}
$$

Register: $[\uparrow_0,\downarrow_0,\uparrow_1,\downarrow_1,\dots]$.

### 3.1 Hopping

A same-spin hop $j\!\to\!j\!+\!1$ connects modes two apart in the ordering, with the opposite spin sitting in between. By (6) with one spectator:

$$
c^\dagger_{j,\uparrow}c_{j+1,\uparrow}+\mathrm{h.c.}
= \tfrac12\big(X_{2j}Z_{2j+1}X_{2j+2}+Y_{2j}Z_{2j+1}Y_{2j+2}\big),
\tag{8}
$$

$$
c^\dagger_{j,\downarrow}c_{j+1,\downarrow}+\mathrm{h.c.}
= \tfrac12\big(X_{2j+1}Z_{2j+2}X_{2j+3}+Y_{2j+1}Z_{2j+2}Y_{2j+3}\big).
\tag{9}
$$

The middle $Z$ is the residual JW string threading the interleaved spectator qubit.

### 3.2 Interaction

On-site, hence local and string-free:

$$
n_{j,\uparrow}n_{j,\downarrow}
= \frac{I-Z_{2j}}{2}\cdot\frac{I-Z_{2j+1}}{2}
= \tfrac14\big(I - Z_{2j} - Z_{2j+1} + Z_{2j}Z_{2j+1}\big).
\tag{10}
$$

### 3.3 Full qubit Hamiltonian (interleaved)

$$
\boxed{
\begin{aligned}
H = -\frac{t}{2}\sum_{j=0}^{L-2}\Big[
&X_{2j}Z_{2j+1}X_{2j+2}+Y_{2j}Z_{2j+1}Y_{2j+2}\\
+\,&X_{2j+1}Z_{2j+2}X_{2j+3}+Y_{2j+1}Z_{2j+2}Y_{2j+3}\Big]
\;+\;\frac{U}{4}\sum_{j=0}^{L-1}\big(I - Z_{2j} - Z_{2j+1} + Z_{2j}Z_{2j+1}\big)
\end{aligned}}
\tag{11}
$$

Every term acts on $\le 3$ *consecutive* qubits — locality is preserved because each spectator string is length-1. (Grouping all $\uparrow$ then all $\downarrow$ instead — §4 — keeps hopping cleaner but moves the cost elsewhere.)

### 3.4 Trotterization (interleaved)

Split into mutually commuting groups. For each spin, "even" and "odd" bonds act on disjoint qubit triples and so commute internally:

$$
H_{\uparrow,\mathrm{e}}=-\frac{t}{2}\!\!\sum_{j\,\mathrm{even}}\!\!\big(X_{2j}Z_{2j+1}X_{2j+2}+Y_{2j}Z_{2j+1}Y_{2j+2}\big),
\tag{12}
$$

and analogously $H_{\uparrow,\mathrm{o}},H_{\downarrow,\mathrm{e}},H_{\downarrow,\mathrm{o}}$; the interaction $H_U$ is a sum of mutually commuting $Z$ and $ZZ$ terms (one layer). Within a group all terms commute and apply in parallel; across the even/odd split they share boundary qubits and do **not** commute — hence the split.

Second-order (symmetric) Suzuki–Trotter step of duration $\tau$:

$$
e^{-iH\tau}\approx
e^{-iH_U\tfrac{\tau}{2}}
e^{-iH_{\downarrow,\mathrm o}\tfrac{\tau}{2}}
e^{-iH_{\downarrow,\mathrm e}\tfrac{\tau}{2}}
e^{-iH_{\uparrow,\mathrm o}\tfrac{\tau}{2}}
e^{-iH_{\uparrow,\mathrm e}\tau}
e^{-iH_{\uparrow,\mathrm o}\tfrac{\tau}{2}}
e^{-iH_{\downarrow,\mathrm e}\tfrac{\tau}{2}}
e^{-iH_{\downarrow,\mathrm o}\tfrac{\tau}{2}}
e^{-iH_U\tfrac{\tau}{2}}
\tag{13}
$$

with per-step error $\mathcal O(\tau^3)$ and global error $\mathcal O(\tau^2)$ at fixed total time. Adjacent half-steps of the *same* group from consecutive Trotter steps merge, so the steady-state cost is **4 hopping layers + 1 interaction layer**. Each hopping layer is a parallel set of 3-qubit $e^{-i\theta(XZX+YZY)}$ rotations; on nearest-neighbor hardware the central $Z$ adds a CZ-like gadget relative to the bare 2-qubit case.

---

## 4. Stacked ordering

Place all $\uparrow$ modes first, then all $\downarrow$:

$$
k=\begin{cases} j & (j,\uparrow)\\ L+j & (j,\downarrow)\end{cases},
\qquad j=0,\dots,L-1.
\tag{14}
$$

Register: $[\uparrow_0,\dots,\uparrow_{L-1},\downarrow_0,\dots,\downarrow_{L-1}]$.

### 4.1 Hopping

Same-spin hops now connect modes that are **adjacent** in the ordering, so by (6) the string is empty:

$$
c^\dagger_{j,\uparrow}c_{j+1,\uparrow}+\mathrm{h.c.}=\tfrac12\big(X_jX_{j+1}+Y_jY_{j+1}\big),
\tag{15}
$$

$$
c^\dagger_{j,\downarrow}c_{j+1,\downarrow}+\mathrm{h.c.}=\tfrac12\big(X_{L+j}X_{L+j+1}+Y_{L+j}Y_{L+j+1}\big).
\tag{16}
$$

Clean weight-2 $XX+YY$ terms — no residual $Z$.

### 4.2 Interaction

Modes $j$ ($\uparrow$) and $L+j$ ($\downarrow$) are $L$ apart in the register, but since both factors are *number* operators the strings cancel internally; the term stays weight-2:

$$
n_{j,\uparrow}n_{j,\downarrow}
=\frac{I-Z_j}{2}\cdot\frac{I-Z_{L+j}}{2}
=\tfrac14\big(I-Z_j-Z_{L+j}+Z_jZ_{L+j}\big).
\tag{17}
$$

The price is **geometric**, not algebraic: $Z_jZ_{L+j}$ couples two qubits separated by $L-1$ others.

### 4.3 Full qubit Hamiltonian (stacked)

$$
\boxed{
H = -\frac{t}{2}\sum_{j=0}^{L-2}\big(X_jX_{j+1}+Y_jY_{j+1}+X_{L+j}X_{L+j+1}+Y_{L+j}Y_{L+j+1}\big)
\;+\;\frac{U}{4}\sum_{j=0}^{L-1}\big(I-Z_j-Z_{L+j}+Z_jZ_{L+j}\big)}
\tag{18}
$$

This is **two decoupled $XX$ chains** (one per spin) coupled only through the long-range Ising rungs $Z_jZ_{L+j}$ — i.e. a **2-leg ladder** with $XX$ legs and $ZZ$ rungs, in which the rungs happen to be far apart in the register.

### 4.4 Trotterization (stacked)

Same five-group structure as (12)ff., now with the cleaner generators:

$$
H_{\uparrow,\mathrm e}=-\frac{t}{2}\!\!\sum_{j\,\mathrm{even}}\!\!\big(X_jX_{j+1}+Y_jY_{j+1}\big),
\quad\text{etc.}, \qquad
H_U=\frac{U}{4}\sum_j\big(I-Z_j-Z_{L+j}+Z_jZ_{L+j}\big).
\tag{19}
$$

The second-order step is identical in form to (13). Steady-state cost is again **4 hopping layers + 1 interaction layer**.

What differs is the hardware mapping:

- Each hopping term $e^{-i\theta(X_jX_{j+1}+Y_jY_{j+1})/2}$ is exactly an $XY(\theta)$ / $\sqrt{i\mathrm{SWAP}}$ — a single native 2-qubit gate (superconducting and trapped-ion).
- The interaction $e^{-i\theta Z_jZ_{L+j}}$ is native on all-to-all hardware (Quantinuum `ZZPhase`), but on nearest-neighbor connectivity (IBM heavy-hex) it needs a SWAP chain of depth $\mathcal O(L)$ to bring the partners together.

So stacked is **optimal on all-to-all / long-range hardware** and **awkward on nearest-neighbor hardware** — exactly the reverse of where interleaved's length-1 strings are convenient.

---

## 5. Side-by-side

| Feature | Interleaved (11) | Stacked (18) |
|---|---|---|
| Hopping term | weight-3, $XZX,\,YZY$ | weight-2, $XX,\,YY$ |
| Interaction term | weight-2, **nearest-neighbor** $ZZ$ | weight-2, **range-$L$** $ZZ$ |
| All terms geometrically local? | yes (3 consecutive qubits) | hopping yes; $U$ no |
| Trotter layers / step | 4 hopping + 1 $U$ | 4 hopping + 1 $U$ |
| Native fit — IBM heavy-hex | good | needs SWAP routing for $U$ |
| Native fit — Quantinuum / neutral atom | extra $Z$ gadget per hop | excellent (every term native) |

---

## 6. What is invariant, and what is not

**Energy is invariant — exactly.** Both orderings (and Bravyi–Kitaev, parity, ternary-tree) are unitary isomorphisms of the same $2^{2L}$ Hilbert space; the encoded Hamiltonian is $U_{\mathrm{enc}}H_{\mathrm{ferm}}U_{\mathrm{enc}}^\dagger$, so the entire spectrum is identical. This is the single sharpest correctness test available:

$$
E_0^{(\mathrm{interleaved})}=E_0^{(\mathrm{stacked})}\stackrel{!}{=}E_0^{\text{Lieb–Wu}}.
\tag{20}
$$

Encoding-agreement catches ordering/transcription bugs; the independent Lieb–Wu (Bethe-ansatz) value catches a wrong Hamiltonian (e.g. a hopping sign error common to all encoders).

**Entanglement is *not* invariant — and that is physical, not a bug.** For a contiguous cut aligned with the JW ordering, the qubit (tensor-product) entanglement entropy equals the fermionic mode-cut entropy (the parity-superselection correction is sub-leading, bounded by the log of the particle-number variance). But the *physical bipartition* a half-cut represents differs:

- **Interleaved**, cut at qubit $2k$: separates real-space sites $\{0,\dots,k-1\}$ (both spins) from the rest — a **real-space** cut.
- **Stacked**, cut at qubit $L$: separates **all $\uparrow$** from **all $\downarrow$** — a **spin-species** cut.

A clean, falsifiable consequence at $U=0$, where the two species decouple into a product of Slater determinants:

$$
U=0:\quad
S_{\mathrm{stacked}}(\text{cut }L)=0,
\qquad
S_{\mathrm{interleaved}}(\text{half})\sim \tfrac{2}{3}\log L .
\tag{21}
$$

The vanishing inter-species entropy at $U=0$ is a unit test for the stacked encoder; as $U$ grows it directly measures interaction-induced spin correlation.

> **Open (conjecture, not established):** the ordering minimizing the *max-over-cuts* entanglement — and hence the MPS bond dimension / preparation cost — is expected to depend on $U/t$, with stacked favored at weak coupling (it does not drag two nearly-decoupled species across one cut) and a crossover at larger $U$. The $U=0$ endpoints in (21) are rigorous; the crossover location is empirical and is exactly what a small exact-diagonalization sweep would pin down.

---

## 7. Verification checklist

For a numerics-first implementation, in order:

1. **Energy agreement (20).** Build $H_{\mathrm{ferm}}$ once; encode under both orderings; ED both; require identical $E_0$ to machine precision. *Catches:* ordering bugs, string bugs, h.c. bugs. *Misses:* a sign error shared by all encoders.
2. **Lieb–Wu anchor.** Compare $E_0$ to the Bethe-ansatz ground energy at the same $(L,U/t,\text{filling},\text{BC})$. *Catches:* a wrong fermionic Hamiltonian. *Misses:* nothing about the encoding per se.
3. **Free-fermion limit.** At $U=0$ check both encoders reproduce the tight-binding spectrum, and that $S_{\mathrm{stacked}}(\text{cut }L)=0$ (21). *Catches:* a spurious inter-species coupling in the stacked $ZZ$.
4. **Symmetry sectors.** Total number $\hat N=\sum_k (I-Z_k)/2$ and $S_z\propto \hat N_\uparrow-\hat N_\downarrow$ commute with $H$; verify the ground state lives in the expected sector. *Catches:* particle-number-violating terms.
5. **Trotter convergence.** Confirm $\|e^{-iH\tau}-U_{\text{ST2}}(\tau)\|=\mathcal O(\tau^3)$ per step for both orderings. *Catches:* a mis-assigned (non-commuting) term inside a layer.
