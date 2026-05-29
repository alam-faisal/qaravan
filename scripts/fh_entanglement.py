"""Entanglement depth and preparation depth vs system size for FermiHubbard1D.

Sweeps num_fermions from 2 to max_nf (both orderings unless restricted),
computes the ground state at fixed n_electrons, records entanglement depth
and (optionally) preparation depth via environment sweep, then saves data
and a plot to the same directory as this script.

Estimated wall-clock at t=U=1, half-filling, both orderings:
  max_nf=5  ~10 min  (ent + prep depth up to nf=5)
  max_nf=6  ~30 min  (ent depth all; prep depth up to nf=5)
  max_nf=7  ~20 min  (ent depth only, no prep sweep)
  max_nf=8  ~90 min  (ent depth only — exceeds 30 min budget)

Usage:
  uv run python scripts/fh_entanglement.py [options]

Examples:
  # Default: ent + prep depth, nf=2..6, half-filling, t=U=1
  uv run python scripts/fh_entanglement.py

  # Entanglement depth only, up to nf=7
  uv run python scripts/fh_entanglement.py --max-nf 7 --no-prep-depth

  # Custom parameters, single ordering
  uv run python scripts/fh_entanglement.py --t 2.0 --U 4.0 --ordering stacked --max-nf 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FH entanglement depth and preparation depth vs system size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--min-nf", type=int, default=2, help="Minimum num_fermions")
    p.add_argument("--max-nf", type=int, default=6, help="Maximum num_fermions")
    p.add_argument("--t", type=float, default=1.0, help="Hopping amplitude")
    p.add_argument("--U", type=float, default=1.0, help="On-site repulsion")
    p.add_argument("--mu", type=float, default=0.0, help="Chemical potential")
    p.add_argument(
        "--n-electrons",
        type=int,
        default=None,
        help="Fixed electron number (default: half-filling = num_fermions)",
    )
    p.add_argument(
        "--ordering",
        choices=["both", "stacked", "interleaved"],
        default="both",
        help="JW ordering(s) to evaluate",
    )
    p.add_argument(
        "--no-prep-depth",
        action="store_true",
        help="Skip preparation depth sweep (faster; entanglement depth only)",
    )
    p.add_argument(
        "--fidelity-target",
        type=float,
        default=0.99,
        help="Fidelity threshold for preparation depth",
    )
    p.add_argument(
        "--max-repeats",
        type=int,
        default=8,
        help="Max brickwall repeats per sweep (true_depth = 2 * max_repeats)",
    )
    p.add_argument(
        "--sweep-iters",
        type=int,
        default=300,
        help="Max environment sweep iterations per repeat",
    )
    p.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames",
    )
    return p.parse_args()


def orderings_from_arg(ordering_arg: str) -> list[str]:
    if ordering_arg == "both":
        return ["stacked", "interleaved"]
    return [ordering_arg]


def run(args: argparse.Namespace) -> None:
    # Late imports — keep startup fast when called with --help
    import matplotlib.pyplot as plt

    from qaravan.applications.circuit_library import brickwall_skeleton
    from qaravan.applications.compilation import environment_state_prep
    from qaravan.applications.quantum_info import entanglement_depth, entanglement_profile
    from qaravan.applications.run_context import RunContext
    from qaravan.backends.statevector import Statevector
    from qaravan.core.hamiltonians import FermiHubbard1D
    from qaravan.plot import set_style

    set_style()

    orderings = orderings_from_arg(args.ordering)
    sizes = list(range(args.min_nf, args.max_nf + 1))
    ctx = RunContext(max_iter=args.sweep_iters)

    # Records keyed by ordering → list (one entry per nf)
    records: dict[str, list[dict]] = {o: [] for o in orderings}

    t_run_start = time.time()

    for nf in sizes:
        n_electrons = args.n_electrons if args.n_electrons is not None else nf
        n_qubits = 2 * nf

        for ordering in orderings:
            print(
                f"\n--- nf={nf}, n_qubits={n_qubits}, n_electrons={n_electrons}, "
                f"ordering={ordering} ---"
            )

            # Ground state
            t0 = time.time()
            h = FermiHubbard1D(nf, t=args.t, U=args.U, mu=args.mu, ordering=ordering)
            gs = h.ground_state(n_electrons=n_electrons)
            print(f"  Ground state: {time.time()-t0:.2f}s")

            # Entanglement profile + depth
            t0 = time.time()
            profile = entanglement_profile(gs)
            ent_d = entanglement_depth(gs)
            ent_time = time.time() - t0
            print(
                f"  Entanglement profile: {ent_time:.2f}s — "
                f"max_S={profile.max():.4f}, ent_depth={ent_d}"
            )

            rec: dict = {
                "nf": nf,
                "n_qubits": n_qubits,
                "n_electrons": n_electrons,
                "ordering": ordering,
                "profile": profile.tolist(),
                "max_S": float(profile.max()),
                "ent_depth": ent_d,
                "ent_profile_time_s": ent_time,
                "prep_depth": None,
                "prep_fidelity": None,
                "prep_time_s": None,
            }

            # Preparation depth sweep
            if not args.no_prep_depth:
                init = Statevector(bitstring="0" * n_qubits)
                t0 = time.time()
                found_depth = None
                found_fidelity = None
                for repeat in range(1, args.max_repeats + 1):
                    skel = brickwall_skeleton(n_qubits) * repeat
                    _, costs = environment_state_prep(gs, init, skeleton=skel, context=ctx)
                    fidelity = 1.0 - costs[-1]
                    print(
                        f"  prep r={repeat} (true_d={2*repeat}): "
                        f"fidelity={fidelity:.4f} [{time.time()-t0:.1f}s elapsed]"
                    )
                    if fidelity >= args.fidelity_target:
                        found_depth = 2 * repeat
                        found_fidelity = fidelity
                        break
                prep_time = time.time() - t0
                rec["prep_depth"] = found_depth
                rec["prep_fidelity"] = found_fidelity
                rec["prep_time_s"] = prep_time
                if found_depth:
                    print(f"  → prep_depth={found_depth}, fidelity={found_fidelity:.4f}")
                else:
                    print(f"  → not reached in true_depth <= {2*args.max_repeats}")

            records[ordering].append(rec)

    elapsed_total = time.time() - t_run_start
    print(f"\nTotal wall-clock: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Save data
    tag = f"_{args.output_tag}" if args.output_tag else ""
    param_str = f"t{args.t}_U{args.U}"
    stem = f"fh_entanglement_{param_str}{tag}"

    data_path = SCRIPT_DIR / f"{stem}.json"
    all_records = [rec for o in orderings for rec in records[o]]
    with open(data_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "records": all_records,
                "total_time_s": elapsed_total,
            },
            f,
            indent=2,
        )
    print(f"Data saved → {data_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_ent, ax_prep = axes
    markers = {"stacked": "o", "interleaved": "s"}
    colors = {"stacked": "C0", "interleaved": "C1"}

    for ordering in orderings:
        recs = records[ordering]
        nf_vals = [r["nf"] for r in recs]
        ent_depths = [r["ent_depth"] for r in recs]
        ax_ent.plot(
            nf_vals,
            ent_depths,
            markers[ordering] + "-",
            color=colors[ordering],
            label=ordering,
        )

    ax_ent.set_xlabel("$L$ (num_fermions)")
    ax_ent.set_ylabel("Entanglement depth (gate layers)")
    ax_ent.set_title(
        f"Entanglement depth lower bound\n$t={args.t},\\ U={args.U}$, "
        f"$N_e={args.n_electrons or 'L'}$"
    )
    ax_ent.legend()
    ax_ent.yaxis.get_major_locator().set_params(integer=True)

    if not args.no_prep_depth:
        for ordering in orderings:
            recs = records[ordering]
            nf_vals = [r["nf"] for r in recs if r["prep_depth"] is not None]
            prep_depths = [r["prep_depth"] for r in recs if r["prep_depth"] is not None]
            if nf_vals:
                ax_prep.plot(
                    nf_vals,
                    prep_depths,
                    markers[ordering] + "-",
                    color=colors[ordering],
                    label=f"{ordering}",
                )
                # Mark points where target was not reached
                nf_missing = [r["nf"] for r in recs if r["prep_depth"] is None]
                if nf_missing:
                    ax_prep.plot(
                        nf_missing,
                        [2 * args.max_repeats] * len(nf_missing),
                        markers[ordering],
                        color=colors[ordering],
                        mfc="none",
                        label=f"{ordering} (not converged)",
                    )

        ax_prep.set_xlabel("$L$ (num_fermions)")
        ax_prep.set_ylabel("Preparation depth (gate layers)")
        ax_prep.set_title(
            f"Preparation depth (generic brickwall, F$\\geq${args.fidelity_target})\n"
            f"$t={args.t},\\ U={args.U}$"
        )
        ax_prep.legend()
        ax_prep.yaxis.get_major_locator().set_params(integer=True)
    else:
        ax_prep.set_visible(False)
        fig.set_size_inches(6, 4.5)

    fig.tight_layout()
    plot_path = SCRIPT_DIR / f"{stem}.pdf"
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    run(parse_args())
