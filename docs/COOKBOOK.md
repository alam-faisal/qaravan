## Qaravan Cookbook

Task-indexed, verified-working recipes against the *currently implemented* code. Unlike
`API_VISION.md` (the target design for the finished v0.2 API, including backends not yet
built), every snippet here should run as written against the present state of `src/`.

### Apply a circuit directly to a state

Every `State` subclass carries a `default_simulator`
([`core/base.py`](../src/qaravan/core/base.py)), so you don't need to name the simulator
class up front:

```python
sv = Statevector(bitstring="01")
final_sv = sv.apply(circ)                        # instead of StatevectorSimulator(circ, sv).run()
```

`**kwargs` forward straight to the simulator constructor, e.g. `state.apply(circ,
noise_model=thermal_nm)`.

`Walkers.apply()` raises `NotImplementedError`: Monte Carlo needs a pure `State` (e.g.
`Statevector`) as its initial state, not a `Walkers` collection — build it explicitly with
`MonteCarloSimulator(circ, initial_walkers, pauli_nm).run()` instead.

### Compute the expectation value of a Hamiltonian

`Hamiltonian` isn't itself an `Observable` — convert it with `as_observable()`, then call
`expectation()` on the state:

```python
obs = tfi.as_observable()          # -> PauliSum
exp = sv.expectation(obs)
```

`Statevector.expectation()` dispatches on the observable's type; for a `PauliSum` it sums
`_expectation_pauli_string` term-by-term and never builds the full `2^n × 2^n` matrix, so
this is the efficient path, not just a convenient one.

`as_observable()` reconstructs the `PauliSum` (merging duplicate Pauli strings) on every
call — cache it (`obs = ham.as_observable()`) rather than calling it inside a loop over many
expectation evaluations.

### Encode a fermionic operator into qubits

`FermionicHamiltonian` is a `FermionicSum` that asserts Hermiticity at construction
(`is_hermitian()`) and carries an optional fermionic-native `trotter_groups`.

Only the `Sum` classes encode. `FermionicTerm`/`MajoranaTerm` don't: Pauli strings need a
register length, and only the sums carry `num_fermionic_modes`.

```python
encoding = FermionicEncoding("jordan_wigner", num_modes=4)    # mode_order=None -> identity

FermionicSum(hopping(0, 1, -1.0), 4).encode(encoding)         # -> PauliSum
MajoranaSum([majorana_bilinear(0, 3)], 4).encode(encoding)    # -> PauliSum
FermiHubbard1D(2).encode(encoding)                            # -> Hamiltonian
```

`FermionicHamiltonian.encode` overrides `FermionicSum.encode` and returns a `Hamiltonian`, not
a `PauliSum` — it preserves `trotter_groups`, so `trotter_circuit()` works on the result.
`as_observable()` gets the `PauliSum` back.
