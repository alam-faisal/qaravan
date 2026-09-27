# Qaravan Gotchas

## Partial measurement does not usually purify a mixed state

Split the register into measured sites $M$ and remaining sites $R$. For a mixed state
$\rho_{MR}$, computational-basis outcome $m$ occurs with

$$
p(m)=\operatorname{Tr}\!\left[(P_m\otimes I)\rho_{MR}\right].
$$

Conditioning on that outcome gives

$$
\rho_{MR\mid m}
=\frac{(P_m\otimes I)\rho_{MR}(P_m\otimes I)}{p(m)}
=|m\rangle\langle m|_M\otimes\rho_{R\mid m}.
$$

The measured sites are in the pure basis state $|m\rangle$, but $\rho_{R\mid m}$ and hence
the full conditional state are generally mixed. A rank-one projective measurement of a
globally pure state is the exception: its conditional state on $R$ is pure.

Discarding the outcome instead gives the generally mixed non-selective state

$$
\rho'=\sum_m(P_m\otimes I)\rho_{MR}(P_m\otimes I).
$$

### `Walkers.measure_and_collapse`: one measurement of the mixed state

For an equal-weight ensemble

$$
\rho_N=\frac1N\sum_i\rho_i,
$$

`walkers.measure_and_collapse(sites, rng)` samples one shared outcome $m$ from

$$
p(m)=\frac1N\sum_i p_i(m),
\qquad p_i(m)=\operatorname{Tr}(P_m\rho_i).
$$

Conditioning changes the ensemble weights:

$$
q_i(m)=\frac{p_i(m)}{\sum_jp_j(m)},
\qquad
\rho_{N\mid m}=\sum_iq_i(m)\rho_{i\mid m}.
$$

Because `Walkers` stores only equal-weight statevectors, the method draws $N$ ancestor
indices with replacement from $q_i(m)$ and projects them onto $m$. The result is an unbiased
Monte Carlo representation of $\rho_{N\mid m}$, not an exact finite decomposition;
duplicates encode the posterior weights.

```python
collapsed, outcome = walkers.measure_and_collapse(sites, rng)
# outcome: one str shared by the conditional ensemble
# collapsed.num_walkers == walkers.num_walkers
```

Resampling adds variance and can impoverish the ensemble after an unlikely outcome. Increase
the walker count when accurate conditional states matter.

### `Walkers.measure_each`: independent experimental trajectories

`walkers.measure_each(sites, rng)` instead measures every walker independently:

$$
m_i\sim p_i(m),
\qquad
(\rho_i)_{i=1}^N\longmapsto(\rho_{i\mid m_i},m_i)_{i=1}^N.
$$

```python
collapsed, outcomes = walkers.measure_each(sites, rng)
# outcomes[i] belongs to collapsed._walkers[i]
```

Use `measure_and_collapse` when `Walkers` represents one mixed state conditioned on one
observed outcome. Use `measure_each` when walkers represent separate experimental runs, as in
batched dynamic-circuit trajectories. Averaging the latter while discarding `outcomes`
estimates the non-selective measurement channel.
