# Proposal: noisy pure-state paths and a delegated Monte Carlo backend

## Decision

Retain `Walkers` and `MonteCarloSimulator` together in `backends/monte_carlo.py`, but change
their relationship and remove the statevector duplication:

```text
Statevector.apply(..., noise_model=...)
    → StatevectorSimulator.run()
    → one sampled noisy pure-state path

Walkers.apply(..., noise_model=...)
    → MonteCarloSimulator.run()
    → one independently sampled path per member
    → Walkers
```

`MonteCarloSimulator` becomes a generic ensemble adapter. It accepts `Walkers`, infers the
member simulator from the first walker, and delegates compilation, noise insertion, gate
translation, and gate application. It no longer accepts a scalar `Statevector`, owns an
ensemble size, or provides `run_ensemble()`.

Dynamic circuits use the noisy capability of `StatevectorSimulator`; they do not use
`MonteCarloSimulator` for scalar paths.

## Noise placement

Placement is part of `PauliNoise`, not a simulator argument or a requirement on every
`NoiseModel`:

```python
class PauliNoise(NoiseModel):
    def __init__(
        self,
        strings: list[str],
        probs: list[float],
        *,
        placement: Literal["layer", "circuit"] = "layer",
    ): ...


class DepolarizingNoise(PauliNoise):
    def __init__(
        self,
        num_sites: int,
        error_rate: float,
        *,
        placement: Literal["layer", "circuit"] = "layer",
    ): ...
```

The error rate is then unambiguously per layer or per circuit application. A future thermal or
gate-dependent model need not adopt vocabulary that may not describe its schedule. A future
density-matrix backend using `PauliNoise` can read the same placement.

Remove every `noise_placement=` simulator argument. Existing DQC mitigation calls become:

```python
noise = DepolarizingNoise(num_sites, error_rate, placement="circuit")
```

## `Simulator` execution stages

Split `run()` at the reusable deterministic/stochastic boundary:

```python
class Simulator(ABC):
    def prepare_circuit(self) -> Circuit:
        circuit = self.circuit.copy()
        if self.decompose:
            circuit.decompose()
        circuit.construct_layers()
        return circuit

    def evolve(self, initial_state: State, prepared_circuit: Circuit) -> State:
        state = copy.deepcopy(initial_state)
        for layer in prepared_circuit.layers:
            for gate in layer:
                translated_gate = self.translate_gate(gate)
                self._apply_translated_gate(state, translated_gate)
        return state

    def run(self) -> State:
        prepared_circuit = self.prepare_circuit()
        realized_circuit = self._insert_noise(prepared_circuit, self.noise_model)
        return self.evolve(self.initial_state, realized_circuit)
```

`translate_gate()` and `_apply_translated_gate()` remain abstract and continue to define a
simulator. `prepare_circuit()` contains only circuit copying, optional decomposition, and layer
construction. `evolve()` copies the state and applies an already prepared realization.

`_insert_noise()` must not mutate its prepared input. This permits one prepared circuit to
generate independent trajectory realizations without accumulated noise layers.

This split now has a concrete non-Monte-Carlo benefit as well: any backend can apply one
prepared circuit to several compatible states without repeating decomposition or layer
construction.

## One noisy statevector path

Move RNG ownership and sampled-Pauli insertion from the old `MonteCarloSimulator` into
`StatevectorSimulator`:

```python
class StatevectorSimulator(Simulator):
    def __init__(
        self,
        circuit: Circuit,
        initial_state: Statevector,
        noise_model: PauliNoise | None = None,
        rng: np.random.Generator | int | None = None,
        decompose: bool = False,
    ): ...

    def _insert_noise(
        self,
        prepared_circuit: Circuit,
        noise_model: PauliNoise | None,
    ) -> Circuit: ...
```

- With no noise model, behavior remains deterministic and unchanged.
- `placement="layer"` samples after every prepared layer.
- `placement="circuit"` samples once after the final layer.
- The method returns a fresh circuit/layer realization and leaves `prepared_circuit` untouched.
- Unsupported noise representations fail through their missing sampled-error operation; no
  broad compatibility registry is added.

`Statevector` does not override `apply()`. Its inherited method now naturally supports:

```python
final_state = initial_state.apply(circuit)
noisy_path = initial_state.apply(circuit, noise_model=noise, rng=rng)
```

The second call returns one random pure-state trajectory, not the channel-averaged mixed state.

## `Walkers` and `MonteCarloSimulator`

### `Walkers`

```python
class Walkers(State):
    def __init__(
        self,
        walkers: State | list[State],
        *,
        num_walkers: int | None = None,
    ): ...

    @property
    def default_simulator(self) -> type[Simulator]:
        return MonteCarloSimulator

    def __iter__(self) -> Iterator[State]: ...
    def __getitem__(self, index: int) -> State: ...
```

- A singular state is replicated `num_walkers` times; the default is one.
- A list defines its own multiplicity and rejects `num_walkers`.
- Reject only empty lists and non-positive replication counts. Do not police hypothetical
  mixtures of representations or dimensions.
- Construct basis states explicitly, for example
  `Walkers(Statevector(bitstring="00"), num_walkers=500)`.
- Keep the current equal-weight `expectation`, `sample`, `rdm`, `measure_and_collapse`, and
  `measure_each` semantics.
- Remove the custom `apply()` that currently raises. Inherited `State.apply()` constructs
  `MonteCarloSimulator` just as every other state constructs its default simulator.

### `MonteCarloSimulator`

```python
class MonteCarloSimulator(Simulator):
    def __init__(
        self,
        circuit: Circuit,
        initial_state: Walkers,
        noise_model: NoiseModel | None = None,
        rng: np.random.Generator | int | None = None,
        decompose: bool = False,
        **member_simulator_kwargs,
    ):
        super().__init__(circuit, initial_state, noise_model, decompose)

        representative = initial_state[0]
        member_simulator_type = representative.default_simulator
        self.member_simulator = member_simulator_type(
            circuit,
            representative,
            noise_model=noise_model,
            rng=rng,
            decompose=decompose,
            **member_simulator_kwargs,
        )

    def run(self) -> Walkers:
        prepared_circuit = self.member_simulator.prepare_circuit()
        final_walkers = []
        for walker in self.initial_state:
            realized_circuit = self._insert_noise(
                prepared_circuit,
                self.noise_model,
            )
            final_walkers.append(self.evolve(walker, realized_circuit))
        return Walkers(final_walkers)

    def _insert_noise(
        self,
        prepared_circuit: Circuit,
        noise_model: NoiseModel | None,
    ) -> Circuit:
        return self.member_simulator._insert_noise(prepared_circuit, noise_model)

    def translate_gate(self, gate: Gate):
        return self.member_simulator.translate_gate(gate)

    def _apply_translated_gate(self, state: State, translated_gate) -> None:
        self.member_simulator._apply_translated_gate(state, translated_gate)
```

`_validate_state()` accepts `Walkers`. The member simulator validates the representative member
and owns all backend-specific options. The ensemble cost is

$$
T_{\mathrm{prepare}} + N\left(T_{\mathrm{noise\ realization}} + T_{\mathrm{evolve}}\right),
$$

not $N T_{\mathrm{prepare}}$. A `Walkers[MPS]` will use `MPSSimulator` once that backend exists;
no Monte Carlo code changes are required.

Remove `num_trajectories` and `run_ensemble()`. `run()` evolves the supplied `Walkers` once and
returns `Walkers`, matching the meaning of every other simulator's `run()`.

## Dynamic circuits

Remove the simulator override from `DynamicCircuit.execute()`. The state already chooses its
compatible simulator through `State.apply()`; allowing a second, caller-selected simulator
creates two competing dispatch mechanisms and existed mainly to force scalar statevectors
through the old `MonteCarloSimulator`.

```python
def execute(
    self,
    initial_state: State,
    *,
    noise_model: NoiseModel | None = None,
    measurement_rng: np.random.Generator | int | None = None,
    noise_rng: np.random.Generator | int | None = None,
    observer: Callable[[int, State], None] | None = None,
    **simulator_kwargs,
) -> tuple[State, list[str]]: ...
```

Resolve both generators once before the round loop. Each decoded circuit uses the state's
default simulator:

```python
measurement_rng = np.random.default_rng(measurement_rng)
noise_rng = np.random.default_rng(noise_rng)

if circuit.gates:
    if noise_model is None:
        state = state.apply(circuit, **simulator_kwargs)
    else:
        state = state.apply(
            circuit,
            noise_model=noise_model,
            rng=noise_rng,
            **simulator_kwargs,
        )
```

Every noisy decoded round invokes one `StatevectorSimulator.run()`. The same noise generator is
forwarded across rounds; resolving an integer seed inside the loop would restart the stream.
Measurement and noise use distinct generators. Observer timing and decoder selection remain
unchanged. The noiseless branch does not impose noise-specific constructor arguments on other
backends.

`DynamicCircuit.execute(Walkers(...))` means one mixed state conditioned on one shared
measurement record: `Walkers.measure_and_collapse()` produces the posterior ensemble, then its
default `MonteCarloSimulator` applies the common decoded circuit.

The current `dqc_ensemble()` remains the independent-record operation for now. It repeats the
entire scalar DQC because different outcomes select different decoder circuits. Its
`num_trajectories` counts independent DQC records, not simulator configuration.

```python
def dqc_ensemble(
    dqc: DynamicCircuit,
    initial_state: State,
    num_trajectories: int,
    rng: np.random.Generator | None = None,
    observer: Callable[[int, State], None] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    **simulator_kwargs,
) -> tuple[Walkers, list[list[str]]]:
    measurement_rng, noise_rng = np.random.default_rng(rng).spawn(2)
    finals, records = [], []
    for _ in range(num_trajectories):
        final, outcomes = dqc.execute(
            initial_state,
            observer=observer,
            measurement_rng=measurement_rng,
            noise_rng=noise_rng,
            noise_model=noise_model,
            **simulator_kwargs,
        )
        finals.append(final)
        records.append(outcomes)
    return Walkers(finals), records
```

The advancing generators preserve the current draw order across rounds and trajectories.

```text
DynamicCircuit.execute(Walkers)   one shared record, posterior mixed state
dqc_ensemble(...)                 one independent record per complete trajectory
```

### Open naming and location decision

`dqc_ensemble()` currently lives in `applications/dynamic_circuit_library.py:243` and is
re-exported from `qaravan.applications`. That module otherwise constructs protocols, circuits,
and decoders; an execution primitive does not belong there.

Before implementation, choose between:

1. Move and rename it to `DynamicCircuit.execute_ensemble(...)`. This is the most discoverable
   companion to `execute()`, but makes `core.dynamic_circuit` construct the backend-specific
   `Walkers` result.
2. Move it to `applications/dynamic_circuit_execution.py` and rename it
   `execute_dynamic_circuit_ensemble(...)`. This preserves the `core -> backend` boundary, at
   the cost of a longer free-function API.

I recommend the second option. `Walkers` belongs to a backend, so core should not import it;
the public re-export from `qaravan.applications` keeps the operation central and easy to import.
Remove `dqc_ensemble` outright rather than keeping a compatibility alias, and update every test,
notebook, and document in the same migration.

## DQC and notebook preservation

Dynamic circuits are the only nontrivial current noisy-trajectory application. Before changing
behavior, record seeded references for:

- Scalar noisy `DynamicCircuit.execute`: outcomes and final statevector.
- `dqc_ensemble`: every record and final state for a small ensemble.
- GHZ fusion: kept-site density matrix and fidelity at the notebook error rates.
- Collisional dynamics: occupation trace across cycles.
- Non-unital mitigation: noisy density curves, calibration quantities, and mitigated result.

After migration, compare records exactly and arrays with stated tolerances. Preserve the order
and number of calls to `sample_error()`: moving the method between simulator classes must not
change the seeded stream. If a notebook already fails, record that baseline and isolate its last
valid physics calculation rather than conflating an unrelated repair with this refactor.

Migrate, execute, and compare:

- `examples/devbooks/monte_carlo_depolarizing.ipynb`
- `examples/devbooks/ghz_fusion.ipynb`
- `examples/research/collisional_models.ipynb`
- `examples/research/nonunital_mitigation.ipynb`
- `scratch/build_notebook.py`, which generates the non-unital notebook

Search all notebook markdown and outputs, docs, tests, and scratch utilities for stale
`MonteCarloSimulator` scalar-path usage and `noise_placement` arguments. Legacy
`qaravan.tensorQ.MonteCarloSim` notebooks remain outside this rewrite.

## Staged implementation

1. **Freeze behavior.** Add seeded DQC regression fixtures and record the notebook quantities
   above with the current implementation.
2. **Split the base pipeline.** Write `prepare_circuit()`/`evolve()` tests first, refactor
   `Simulator.run()` without changing results, and make `_insert_noise()` non-mutating.
   Immediately run the full repository suite and commit this checkpoint independently. No
   backend or Monte Carlo API changes begin until it passes.
3. **Move scalar noise to statevector simulation.** Add `placement` to `PauliNoise`, implement
   statevector RNG/noise insertion, and compare seeded scalar paths against the old
   `MonteCarloSimulator`. Run statevector, circuit, noise, and DQC tests; commit independently.
4. **Refactor the ensemble backend.** Generalize `Walkers`, make its default simulator
   `MonteCarloSimulator`, replace the simulator's statevector kernel with member-backend
   delegation, and remove `num_trajectories`/`run_ensemble()`. Test direct noiseless/noisy
   `Walkers.apply()` and preparation-once behavior; commit independently.
5. **Migrate DQC.** Remove the `simulator` override, expose `noise_model`, `measurement_rng`, and
   `noise_rng`, relocate/rename `dqc_ensemble()` after resolving the open API decision, preserve
   the two RNG streams, and add shared-record `Walkers` coverage. Run all DQC, Monte Carlo,
   measurement, and application tests, then the full repository suite. Compare the frozen
   references before committing.
6. **Audit examples and docs.** Execute the five notebook/generator items, compare their physics
   outputs, and document tolerances or pre-existing failures. Update `API_VISION.md`,
   `COOKBOOK.md`, and `GOTCHAS.md` only after results are stable.
7. **Finish under repository protocol.** Run `check-docstrings`, write and density-edit the
   report and notebook material, and wait for review before final formatting, full-suite rerun,
   and push.

## Tests

- Base pipeline: source circuit/state immutability, unchanged `run()` results, prepared-circuit
  reuse, and non-accumulating noise realizations.
- Noise models: placement construction and per-layer/per-circuit sampled-error counts.
- Statevector: deterministic equivalence, seeded path equivalence with the old implementation,
  analytical depolarizing expectations, and advancing RNG behavior.
- Walkers: replication/list construction, inherited `apply()`, preserved cardinality/order,
  one preparation per ensemble, independent noise paths, backend delegation, and existing
  expectation/sampling/RDM/measurement laws.
- Use a small fake second member state/simulator pair to prove delegation without implementing
  speculative MPS behavior.
- DQC: exact seeded records, final states, RNG isolation, observer timing, zero-noise
  equivalence, GHZ fidelity, collisional traces, and non-unital mitigation quantities.
- Distinguish shared-record `DynamicCircuit.execute(Walkers)` from independent-record
  `dqc_ensemble()`.

## Documentation and report

- Keep `Walkers` and `MonteCarloSimulator` exported from `qaravan.backends`; document the new
  `Walkers → Walkers` simulator contract.
- Rewrite the Monte Carlo devbook around noisy `Statevector.apply()` and `Walkers.apply()`.
- Document that one noisy pure-state application is one trajectory, not a mixed-state channel.
- Extend `GOTCHAS.md` with shared-record mixed-state DQC versus independent DQC trajectories.
- Correct `API_VISION.md` and `COOKBOOK.md` examples.
- Write `reports/2026-09-27-walkers-native-monte-carlo.md` with signatures, tests, before/after
  DQC comparisons, and notebooks to inspect.

No implementation begins until this third proposal is explicitly approved.
