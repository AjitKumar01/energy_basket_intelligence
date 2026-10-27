# Bench: timing and law checks for the normaliser and sampler

Tools for changing `v4/exact.py`, `v4/sampling.py` or the C++ kernels in `v4/native/`
safely: time the costly operations on a fitted model, and check that a change keeps the law
(exact log Z, exact draws, Gibbs chains, held-out likelihood).

## Setup

```bash
source bench/env.sh       # PY, kernel build folder, 4 threads, the two models (all overridable)
```

The two fitted models the scripts load (`--world`):

| `--world` | model | variables | nmax | categories | products per trip |
|---|---|---|---|---|---|
| `syn` | synthetic discount world, seed 3 (`run/small_world.sh`) | `SYN_RUN_DIR`, `SYN_DATA`, `SYN_CK` | 31 | 48 | 522 |
| `real` | Dunnhumby commodity model (`run/commodity_comp.sh`) | `REAL_RUN_DIR`, `REAL_DATA`, `REAL_CK` | 120 | 187 | 5,455 |

`*_RUN_DIR` is the folder the checkpoint's recorded paths are relative to (the scripts change
into it). Set them if the runs are elsewhere than this checkout's `runs/`. The held-out trips
are the fit's own validation split, in a fixed order, so two code versions see the same trips.

Write outputs under `runs/bench/`. Use `$PY -u ... > file 2>&1` (never pipe a run through
`grep` into a file: grep buffers and the log stays empty), at most 4 threads per job, and time
only on an otherwise idle machine.

## Scripts

| command | what it gives |
|---|---|
| `$PY -u bench/timing.py --world W [--quick]` | exact log Z and one Gibbs sweep per 256 trips; likelihood of one basket and per basket in 256, at the validation setting |
| `$PY -u bench/law_checks.py --world W --check chain [--seed S]` | Gibbs chains from the observed baskets: mean and largest basket size, trips at the item cap, after 0–400 sweeps |
| `$PY -u bench/law_checks.py --world W --check draws` | exact draws given z against the exact incidence, at z near the observed basket and at z from 200-sweep chains |
| `$PY -u bench/law_checks.py --world W --check logz --save F.npz` | exact log Z at fixed (trip, z) pairs |
| `$PY -u bench/law_checks.py --world W --check ll --trips 256 --save F.npz` | held-out log-likelihood at 1,024 draws, per basket |
| `$PY -u bench/modes.py --world W [--trips 64]` | chains from the observed basket, from the model without interactions and from two far starts: basket size, share at the cap, and the z-marginal weight of where each ends (a heavy region the data's chain does not reach) |
| `$PY bench/compare.py OLD.npz NEW.npz` | per-trip NEW − OLD: min, max, mean, standard error |

## Checking a change

Run the same command on the old and the new code (two git worktrees; give each its own
`V4_BUILD_DIR`), then compare:

```bash
$PY -m pytest -q tests/                       # exactness against enumeration of every basket
$PY -u bench/law_checks.py --world real --check logz --save runs/bench/new_logz.npz
$PY bench/compare.py runs/bench/old_logz.npz runs/bench/new_logz.npz
```

Agreement of log Z at fixed z is not enough. An uncapped variant once matched log Z to 7e-12
at fixed z, but its chains ran away (one real trip went to about 2,660 products in 100
sweeps). Always also compare `--check chain` on several seeds and `--check ll` with
`compare.py`. And check against an independent calculation where values are extreme: the
2-D size table once lost 123 nats of log Z on a real trip at a chain-end z, identically in the
old and new code, so old-against-new agreement did not show it (`docs/PIPELINE.md` §7.0).
