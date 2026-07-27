"""Runs the constant-theta TVD/XEB sweep across a process pool.

Usage:
    uv run python run_xeb_sweep.py [--limit N] [--workers W]

--limit N restricts to the first N tasks, for a quick correctness/timing
test before committing to the full 100-task sweep.

Saving hygiene: results are written to const_sweep_results.pkl after every
few completions (atomic write via a temp file + os.replace, so a crash or
Ctrl-C never leaves a half-written file), and again in a `finally` block on
exit -- whatever has completed is never lost, even if the run is interrupted.
"""
import argparse
import os
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from xeb_sweep_worker import process_one

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "const_sweep_results.pkl"

SEEDS = [12, 36, 89, 1000, 17895]
CONST_THETAS = ["0p000000", "0p087266", "0p174533", "0p261799", "0p349066",
                "0p436332", "0p523599", "0p610865", "0p698132", "0p785398"]


def save(results):
    tmp_path = RESULTS_PATH.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_path, RESULTS_PATH)  # atomic on POSIX -- never a half-written file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=10)  # ~71% of 14 physical cores
    args = parser.parse_args()

    tasks = [(n, theta_str, seed) for n in (20, 24) for theta_str in CONST_THETAS for seed in SEEDS]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"{len(tasks)} tasks, {args.workers} workers, saving to {RESULTS_PATH}", flush=True)

    results = []
    t_start = time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"FAILED {task}: {e!r}", flush=True)

                if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                    save(results)
                    print(f"{i + 1}/{len(tasks)} done ({time.time() - t_start:.0f}s elapsed), "
                          f"saved {len(results)} results", flush=True)
    finally:
        save(results)
        print(f"STOPPED/FINISHED: {len(results)}/{len(tasks)} results saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
