# Walkers-native Monte Carlo and explicit noisy DQC execution

**Status: complete and ready for review.** Monte Carlo is now the ensemble
orchestrator for `Walkers`, while individual noisy trajectories remain ordinary
backend simulations. Existing seeded dynamic-circuit results are unchanged.

## Public API

```python
Walkers(
    walkers: State | list[State],
    *,
    num_walkers: int | None = None,
)

MonteCarloSimulator(
    circuit: Circuit,
    initial_state: Walkers,
    noise_model: NoiseModel | None = None,
    rng: np.random.Generator | int | None = None,
    decompose: bool = False,
    **member_simulator_kwargs,
)

DynamicCircuit.execute(
    initial_state: State,
    *,
    noise_model: NoiseModel | None = None,
    measurement_rng: np.random.Generator | int | None = None,
    noise_rng: np.random.Generator | int | None = None,
    observer: Callable[[int, State], None] | None = None,
    **simulator_kwargs,
) -> tuple[State, list[str]]

execute_dynamic_circuit_ensemble(
    dynamic_circuit: DynamicCircuit,
    initial_state: State,
    num_trajectories: int,
    rng: np.random.Generator | int | None = None,
    observer: Callable[[int, State], None] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    **simulator_kwargs,
) -> tuple[Walkers, list[list[str]]]
```

`PauliNoise` and `DepolarizingNoise` now take
`placement: Literal["layer", "circuit"] = "layer"`. Placement is a property of
the noise model rather than a simulator call.

## Architecture

- `State.apply` remains the common entry point. `Walkers.apply(circuit, ...)`
  selects `MonteCarloSimulator`, just as `Statevector.apply` selects
  `StatevectorSimulator`.
- `Walkers(state, num_walkers=N)` represents an equal-weight ensemble with `N`
  copies of one state. `Walkers([state_0, ...])` retains an explicit ensemble.
- Monte Carlo infers the member backend from the first walker, prepares the
  circuit once, then independently inserts sampled noise and evolves every
  walker. It does not re-run decomposition, layering, or backend setup per
  trajectory.
- `Simulator.run()` is factored into `prepare_circuit()` and `evolve()` so this
  prepare-once loop is reusable without weakening the simulator abstraction.
  Ordinary simulators still define themselves through `translate_gate` and
  `_apply_translated_gate`.
- A noisy `Statevector.apply` is one sampled pure-state trajectory. Monte Carlo
  repeats that operation over an ensemble; it does not implement statevector
  gate action itself. This leaves the same delegation route available to a
  future MPS-backed `Walkers` ensemble.
- `DynamicCircuit.execute` receives a noise model, not a simulator. Each round
  applies the current state's backend, with separate measurement and noise RNG
  streams. The free `execute_dynamic_circuit_ensemble` function repeats whole
  trajectories and returns every classical record.
- The former `run_ensemble`, `dqc_ensemble`, DQC `simulator=...`, and simulator
  `noise_placement=...` APIs were removed rather than retained as aliases.

## Measurement semantics

`Walkers.measure_and_collapse(sites)` now implements measurement of the mixed
state represented by an equal-weight ensemble. It samples one shared outcome
from the ensemble Born distribution, posterior-resamples walker ancestors with
replacement, and projects every selected ancestor onto that outcome. The result
is generally still mixed on unmeasured sites.

`Walkers.measure_each(sites)` is intentionally different: it measures each
walker independently and returns one outcome per walker. This is useful for
independent trajectories, but it is not a single measurement of the represented
mixed state.

## Verification

- A full-suite checkpoint after the source/API migration passed: **695 tests**.
- The final focused regression set passed: **78 tests** covering Walkers,
  Monte Carlo delegation, statevector sampled noise, DQC execution, and DQC
  applications.
- Ruff is clean on all touched Python source and test files.
- Frozen seeded DQC regressions preserve both records and amplitudes. The scalar
  fixture still returns records `1, 1, 1, 0, 1` and final amplitudes
  `[-i, -i] / sqrt(2)`.
- A recording backend verifies that Monte Carlo calls member preparation once,
  delegates translation/application to the member simulator, and realizes
  noise separately for each walker. This catches accidental recompilation and
  hard-coding of statevector evolution; it does not yet exercise a real MPS
  implementation because that backend is not present.
- Sampling and collapse tests distinguish mixture measurement from independent
  walker measurement and verify posterior resampling statistics.

All affected notebooks execute without errors. Their numerical checks are:

| Notebook | Result |
|---|---|
| `monte_carlo_depolarizing.ipynb` | Noiseless one-walker differences are zero; maximum single-qubit analytic-curve deviation is 0.0160; Bell `ZZ` decay is monotone. |
| `ghz_fusion.ipynb` | Noiseless fidelities remain 1; noisy fidelities at `p = 0, 0.05, 0.2` are 1.0000, 0.7700, 0.3250. |
| `collisional_models.ipynb` | Lindblad deviation is 1.14 sigma; timestep errors decrease 0.0568, 0.0149, 0.0055; final noisy occupations increase 0.0005, 0.0742, 0.2148. |
| `nonunital_mitigation.ipynb` | Inferred decay differs from truth by 1.3%; mitigation lowers mean absolute error from 0.0158 to 0.0040; reference inversion errors are below `6e-17`. |

## Review checklist

- [ ] Read `docs/API_VISION.md` and `docs/COOKBOOK.md` for the intended split
      between a single noisy trajectory and a `Walkers` ensemble.
- [ ] Read `docs/GOTCHAS.md` for the distinction among
      `measure_and_collapse`, `measure_each`, and independent DQC records.
- [ ] Run `examples/devbooks/monte_carlo_depolarizing.ipynb` and confirm the
      ensemble API and displayed decay curves are the interface you want.
- [ ] Run `examples/devbooks/ghz_fusion.ipynb` and
      `examples/research/collisional_models.ipynb` to verify DQC records and
      noisy physics.
- [ ] Inspect `tests/test_backends/test_monte_carlo.py`, especially the fake
      member backend and mixed-state collapse tests.

Commits: `56f3c94`, `7c06e83`, `206e582`, `c716748`, `7888a11`.
The earlier measurement correction is in `da9a8cd`.
