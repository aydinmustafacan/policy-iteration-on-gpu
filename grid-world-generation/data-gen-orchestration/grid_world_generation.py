#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- Defaults: edit freely ----------
# GRID: only G1–G3 (slip, no walls, fixed goal, γ=0.9)
GRID_SIZES      = ["512x512"]
GRID_TRANS      = ["slip"]            # only slip for perf
GRID_BLOCKED    = [0.0]               # no walls
GRID_OBSTACLES  = [0.0]               # no penalty cells
GRID_GAMMAS     = [0.9]               # only 0.9
GRID_GOALS      = ["fixed"]
GRID_STORAGE    = "sparse"
GRID_GEN_ONLY   = True
GRID_SEED       = 123
GRID_BOUNDARIES = ["reflect"]         # boundary handling

# RANDOM-SPARSE: only R1–R2 (5k states, very vs moderate)
RND_STATES      = [5000]
RND_SPARSITIES  = ["very", "moderate"]
RND_RTYPES      = ["uniform"]
RND_ACTIONS_BY_SP = {"very":"fixed:4", "moderate":"fixed:4"}
RND_STORAGE     = "sparse"
RND_GEN_ONLY    = True
RND_SEED        = 456

# TOY: none for perf; we’ll skip
TOY_SIZES       = []

# EDGE: only absorbing1 (E1)
EDGE_CASES      = ["absorbing1"]



DEFAULT_GAMMA = 0.9
DEFAULT_THETA = 1e-8
DEFAULT_MAXITER = 1000


# ---------- Job helpers ----------
def safe_tag(s: str) -> str:
    return s.replace(":", "").replace("/", "_")


def cmd_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_job(cmd: List[str], out_dir: Path, timeout: int = 0) -> Tuple[int, float, str]:
    """Run a single job, redirect stdout/err to run.log, return (rc, elapsed, cmdstr)."""
    ensure_dir(out_dir)
    log_path = out_dir / "run.log"
    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write("# CMD: " + cmd_str(cmd) + "\n\n")
        try:
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout or None)
        except subprocess.TimeoutExpired:
            rc = -9
            logf.write("\nTIMEOUT\n")
    elapsed = time.time() - t0
    return rc, elapsed, cmd_str(cmd)


def write_manifest_header(path: Path):
    if not path.exists():
        with path.open("w") as f:
            f.write("tag,dataset,params,dir,status,return_code,elapsed_s\n")


def append_manifest(path: Path, row: Dict[str, str]):
    with path.open("a") as f:
        f.write(
            "{tag},{dataset},{params},{dir},{status},{return_code},{elapsed_s}\n".format(
                **row
            )
        )


# ---------- Builders for each family ----------
def build_grid_jobs(bin_path: Path, outroot: Path, gamma: float, theta: float, maxiter: int,
                    storage: str, seed: int, gen_only: bool) -> List[Tuple[List[str], Path, Dict]]:
    jobs = []
    for sz, trans, bnd, blk, obs, g, goals in itertools.product(
            GRID_SIZES, GRID_TRANS, GRID_BOUNDARIES, GRID_BLOCKED, GRID_OBSTACLES, GRID_GAMMAS, GRID_GOALS
    ):
        H, W = map(int, sz.split("x"))
        tag = f"gw_{sz}_{trans}_{bnd}_blk{blk}_obs{obs}_g{g}_{safe_tag(goals)}"
        out_dir = outroot / tag
        ensure_dir(out_dir)

        cmd = [
            str(bin_path),
            "--dataset", "grid",
            "--height", str(H), "--width", str(W),
            "--trans", trans,
            "--boundary", bnd,
            "--blocked", str(blk),
            "--obstacles", str(obs),
            "--goals", goals,
            "--gamma", str(g),
            "--theta", str(theta),
            "--max-iter", str(maxiter),
            "--storage", storage,
            "--seed", str(seed),
            "--out-mdp", str(out_dir / "mdp.json"),
            "--out-policy", str(out_dir / "policy.csv"),
            "--out-value", str(out_dir / "value.csv"),
        ]
        if gen_only:
            cmd.append("--gen-only")
        jobs.append((cmd, out_dir, {"dataset": "grid", "tag": tag}))
    return jobs


def build_random_jobs(bin_path: Path, outroot: Path, gamma: float, theta: float, maxiter: int,
                      storage: str, seed: int, gen_only: bool) -> List[Tuple[List[str], Path, Dict]]:
    jobs = []
    for S, sp, rt in itertools.product(RND_STATES, RND_SPARSITIES, RND_RTYPES):
        actions = RND_ACTIONS_BY_SP[sp]
        tag = f"rnd_S{S}_{sp}_{rt}"
        out_dir = outroot / tag
        ensure_dir(out_dir)
        cmd = [
            str(bin_path),
            "--dataset", "random",
            "--states", str(S),
            "--actions", actions,
            "--sparsity", sp,
            "--rtype", rt,
            "--pattern-k", "10",
            "--gamma", str(gamma),
            "--theta", str(theta),
            "--max-iter", str(maxiter),
            "--storage", storage,
            "--seed", str(seed),
            "--out-mdp", str(out_dir / "mdp.json"),
            "--out-policy", str(out_dir / "policy.csv"),
            "--out-value", str(out_dir / "value.csv"),
        ]
        if gen_only:
            cmd.append("--gen-only")
        jobs.append((cmd, out_dir, {"dataset": "random", "tag": tag}))
    return jobs


def build_toy_jobs(bin_path: Path, outroot: Path, gamma: float, theta: float, maxiter: int) -> List[
    Tuple[List[str], Path, Dict]]:
    jobs = []
    for n in TOY_SIZES:
        tag = f"toy_{n}x{n}"
        out_dir = outroot / tag
        ensure_dir(out_dir)
        cmd = [
            str(bin_path),
            "--dataset", "toy",
            "--size", str(n),
            "--gamma", str(gamma),
            "--theta", str(theta),
            "--max-iter", str(maxiter),
            "--out-mdp", str(out_dir / "mdp.json"),
            "--out-policy", str(out_dir / "policy.csv"),
            "--out-value", str(out_dir / "value.csv"),
        ]
        jobs.append((cmd, out_dir, {"dataset": "toy", "tag": tag}))
    return jobs


def build_edge_jobs(bin_path: Path, outroot: Path, gamma: float, theta: float, maxiter: int) -> List[
    Tuple[List[str], Path, Dict]]:
    jobs = []
    for e in EDGE_CASES:
        tag = f"edge_{safe_tag(e)}"
        out_dir = outroot / tag
        ensure_dir(out_dir)
        cmd = [
            str(bin_path),
            "--dataset", "edge",
            "--edge", e,
            "--gamma", str(gamma),
            "--theta", str(theta),
            "--max-iter", str(maxiter),
            "--out-mdp", str(out_dir / "mdp.json"),
            "--out-policy", str(out_dir / "policy.csv"),
            "--out-value", str(out_dir / "value.csv"),
        ]
        jobs.append((cmd, out_dir, {"dataset": "edge", "tag": tag}))
    return jobs


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Orchestrate MDP dataset generation")
    ap.add_argument("--bin", default="../build/grid_world_generation", help="Path to compiled generator binary")
    ap.add_argument("--outdir", default="datasets", help="Root directory for outputs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel jobs")
    ap.add_argument("--timeout", type=int, default=0, help="Per-job timeout (sec). 0=none")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--theta", type=float, default=DEFAULT_THETA)
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAXITER)
    ap.add_argument("--grid-seed", type=int, default=GRID_SEED)
    ap.add_argument("--random-seed", type=int, default=RND_SEED)
    ap.add_argument("--grid-storage", default=GRID_STORAGE, choices=["dense", "sparse"])
    ap.add_argument("--random-storage", default=RND_STORAGE, choices=["dense", "sparse"])
    ap.add_argument("--grid-gen-only", action="store_true", default=GRID_GEN_ONLY)
    ap.add_argument("--random-gen-only", action="store_true", default=RND_GEN_ONLY)
    ap.add_argument("--skip", nargs="*", default=[], choices=["grid", "random", "toy", "edge"], help="Datasets to skip")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and exit")
    args = ap.parse_args()

    bin_path = Path(args.bin).resolve()
    if not bin_path.exists():
        print(f"Binary not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    outroot = Path(args.outdir).resolve()
    ensure_dir(outroot)
    manifest = outroot / "manifest.csv"
    write_manifest_header(manifest)

    jobs = []
    if "grid" not in args.skip:
        jobs += build_grid_jobs(bin_path, outroot, args.gamma, args.theta, args.max_iter,
                                args.grid_storage, args.grid_seed, args.grid_gen_only)
    if "random" not in args.skip:
        jobs += build_random_jobs(bin_path, outroot, args.gamma, args.theta, args.max_iter,
                                  args.random_storage, args.random_seed, args.random_gen_only)
    if "toy" not in args.skip:
        jobs += build_toy_jobs(bin_path, outroot, args.gamma, args.theta, args.max_iter)
    if "edge" not in args.skip:
        jobs += build_edge_jobs(bin_path, outroot, args.gamma, args.theta, args.max_iter)

    if args.dry_run:
        for cmd, out_dir, meta in jobs:
            print(cmd_str(cmd))
        print(f"\n{len(jobs)} commands (dry-run).")
        return

    print(f"Running {len(jobs)} jobs with {args.jobs} workers…")
    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_job, cmd, out_dir, args.timeout): (cmd, out_dir, meta)
                for (cmd, out_dir, meta) in jobs}
        for fut in as_completed(futs):
            cmd, out_dir, meta = futs[fut]
            try:
                rc, elapsed, cstr = fut.result()
            except Exception as e:
                rc, elapsed, cstr = 99, 0.0, cmd_str(cmd)
                print(f"[ERROR] {out_dir.name}: {e}", file=sys.stderr)

            status = "OK" if rc == 0 else "FAIL"
            if rc == 0: ok += 1
            append_manifest(manifest, {
                "tag": meta["tag"],
                "dataset": meta["dataset"],
                "params": json.dumps({"cmd": cstr}),
                "dir": str(out_dir),
                "status": status,
                "return_code": str(rc),
                "elapsed_s": f"{elapsed:.3f}",
            })
            print(f"[{status}] {meta['tag']}  ({elapsed:.1f}s)")

    print(f"\nDone. {ok}/{len(jobs)} succeeded.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
