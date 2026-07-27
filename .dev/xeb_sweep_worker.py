"""Multiprocessing-safe worker for the constant-theta TVD/XEB sweep.

Must be a real importable module (not defined inline in a notebook) --
ProcessPoolExecutor uses the 'spawn' start method on macOS, which can't
pickle interactively-defined functions. ('fork' would dodge that, but
Apple's Accelerate BLAS framework is not fork-safe, so it's the riskier
choice on this machine.)
"""
import os
from pathlib import Path

# cap BLAS threads per worker BEFORE numpy is imported: unrestricted threading
# on chi~150-260 matmuls wastes ~13.8x the CPU-seconds for only ~1.5-2x
# wall-clock gain (measured directly). Parallelize across the (n,theta,seed)
# grid across worker processes instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from qaravan.core import Circuit, Gate, RY, CNOT
from qaravan.tensorQ import MPSSim

BASE_DIR = Path(__file__).resolve().parent  # .dev/ -- anchored here regardless of CWD
CIRCUITS_DIR = BASE_DIR / "unzipped" / "circuits" / "circuits"
CONST_THETA_DIR = BASE_DIR / "unzipped" / "constant_theta" / "constant_theta"
XEB_SELF_SAMPLES = 4_800  # 16x the earlier 300 -- affordable now that this runs in parallel


def load_matchgate_circuit(n, seed):
    path = CIRCUITS_DIR / f"matchgates_list_n{n}_seed{seed}.pkl"
    with open(path, "rb") as f:
        mat_list = pickle.load(f)
    return Circuit(gate_list=[Gate("U", [i, j], U) for U, i, j in mat_list], n=n)


def extent_prep_circuit(n, theta):
    """ (cos(theta)|0000> + sin(theta)|1111>)^{ox n/4}, one block of 4 qubits at a time. """
    gate_list = []
    for b in range(n // 4):
        base = 4 * b
        gate_list += [RY([base], theta), CNOT([base + 1, base]),
                      CNOT([base + 2, base + 1]), CNOT([base + 3, base + 2])]
    return Circuit(gate_list=gate_list, n=n)


def load_counts(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # {bitstring: raw count}, sums to 100_000


def tvd_no_unseen_mass(counts, p_mps):
    """ Plug-in TVD without the unseen-mass correction -- see chat/status doc. """
    total = sum(counts.values())
    return 0.5 * sum(abs(counts[y] / total - p_mps[y]) for y in counts)


def tvd_with_unseen_mass(counts, p_mps):
    """ Plug-in TVD including the unseen-mass term -- treats every bitstring
    Francesca never sampled as 100% disagreement. Exact given p_mps (no
    sampling), unlike the XEB self-overlap step. """
    total = sum(counts.values())
    seen_mass = sum(p_mps.values())
    return 0.5 * (sum(abs(counts[y] / total - p_mps[y]) for y in counts) + (1 - seen_mass))


def process_one_tvd_only(task):
    """ Both TVD variants, no self_overlap -- much cheaper than process_one. """
    n, theta_str, seed = task
    theta = float(theta_str.replace("p", "."))
    path = CONST_THETA_DIR / f"samples_freq_dict_n{n}_theta{theta_str}_seed{seed}.pkl"
    counts = load_counts(path)

    circ = extent_prep_circuit(n, theta) + load_matchgate_circuit(n, seed)
    sim = MPSSim(circ)
    sim.run(progress_bar=False)
    mps = sim.state
    p_mps = {y: mps.evaluate([int(b) for b in y]) for y in counts}
    seen_mass = sum(p_mps.values())

    return dict(n=n, theta=theta, seed=seed,
                tvd_no_unseen=tvd_no_unseen_mass(counts, p_mps),
                tvd_with_unseen=tvd_with_unseen_mass(counts, p_mps),
                seen_mass=seen_mass, support_size=len(counts), chi=mps.get_max_dim())


def self_overlap(mps, num_samples=XEB_SELF_SAMPLES, batch_size=300):
    """ Monte Carlo estimate of S_Q = sum_y Q(y)^2, sampling from our own MPS. """
    total = 0.0
    remaining = num_samples
    while remaining > 0:
        k = min(batch_size, remaining)
        ys = mps.fast_sample(num_samples=k)
        total += sum(mps.evaluate([int(b) for b in y]) for y in ys)
        remaining -= k
    return total / num_samples


def xeb_fidelity(counts, p_mps, s_q, dim):
    """ Linear cross-entropy fidelity -- see chat derivation / xeb_fidelity_notes.md. """
    total = sum(counts.values())
    cross = sum(cnt * p_mps[y] for y, cnt in counts.items()) / total
    uniform = 1.0 / dim
    return (cross - uniform) / (s_q - uniform)


def process_one(task):
    n, theta_str, seed = task
    theta = float(theta_str.replace("p", "."))
    path = CONST_THETA_DIR / f"samples_freq_dict_n{n}_theta{theta_str}_seed{seed}.pkl"
    counts = load_counts(path)

    circ = extent_prep_circuit(n, theta) + load_matchgate_circuit(n, seed)
    sim = MPSSim(circ)  # msvr=None: exact, chi stays small at this depth
    sim.run(progress_bar=False)
    mps = sim.state
    p_mps = {y: mps.evaluate([int(b) for b in y]) for y in counts}

    total = sum(counts.values())
    tvd = tvd_no_unseen_mass(counts, p_mps)
    s_q = self_overlap(mps)
    f_xeb = xeb_fidelity(counts, p_mps, s_q, 2 ** (n - 1))

    return dict(n=n, theta=theta, seed=seed, tvd=tvd, f_xeb=f_xeb, s_q=s_q,
                support_size=len(counts), total_shots=total, chi=mps.get_max_dim())
