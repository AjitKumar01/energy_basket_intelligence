# Bench: timing and law checks for the normaliser and sampler

Everything needed to measure and check a change to `v4/exact.py`, `v4/sampling.py`,
`v4/native/exact_dp.cpp` or `v4/native/sampler.cpp` on the two fitted models. All commands
have been run as written on this machine.

## Setup (every new shell)

```bash
cd /Users/ajit/Projects/nf_dunnhumby/ebi_v4_pipeline
source bench/env.sh            # sets PY, the model paths and OMP_NUM_THREADS=4
```

- Use `$PY` (Python 3.13.4 at `/Users/ajit/.pyenv/versions/3.13.4/bin/python3`). It has torch,
  pandas, scipy and pytest. Do not use the Homebrew Python 3.11: it has no pytest, and its
  timings differ (one synthetic basket: 0.198 s there, 0.265 s here). Compare old and new code
  with the same interpreter.
- The C++ kernels build themselves on first use (clang++) into `build/`, and rebuild when
  the `.cpp` file is newer than the library. After editing a `.cpp` file, the first run
  prints nothing for a few seconds while it compiles.
- Write outputs under `runs/bench/` (git-ignored). Never pipe a run through `grep` into a
  file: grep buffers and the log stays empty until the end. Use `$PY -u ... > file 2>&1`
  and follow it with `tail -f file`.
- At most 4 threads per job. Run timings with nothing else running (check with `top`), and
  never kill a process you did not start.

## Tests (exactness against enumeration of every basket)

```bash
$PY -m pytest -q tests/test_law.py          # 35 passed in about 2 s
```

`tests/tiny.py` builds a world where every basket can be listed; `tests/test_law.py` checks
log Z, incidence, the exact draw, Gibbs and importance sampling against that list. Add a
test there for any new computation.

## The two models

| `--world` | model | nmax | categories | products per trip | load time |
|---|---|---|---|---|---|
| `syn` | synthetic world, seed 3 (`ebi_v4_scale/runs/disc_c48_s3_promo/v4_comp.pt`) | 31 | 48 | 522 | ~2 s |
| `real` | Dunnhumby commodity model (`runs/commodity/v4_comp_promo.pt`) | 120 | 187 | 5,455 | ~15 s |

`bench/common.py` loads either (it changes into the directory the checkpoint's paths are
relative to) and takes the fit's own held-out trips in a fixed order, so two code versions
see the same trips.

## Scripts

| command | what it measures | time on syn / real |
|---|---|---|
| `$PY -u bench/timing.py --world syn --quick` | smoke test: 32 trips | 4 s / 48 s |
| `$PY -u bench/timing.py --world syn` | exact log Z and one Gibbs sweep per 256 trips; likelihood per basket, 1 basket and 256, at 128 draws | 30 s / ~8 min |
| `$PY -u bench/law_checks.py --world syn --check chain [--seed 9]` | Gibbs chains from the observed baskets: mean and largest basket size, trips at the item cap, after 0-400 sweeps | 10 s / ~2 min |
| `$PY -u bench/law_checks.py --world syn --check draws` | exact draws given z against the exact incidence, at z near the observed basket and at z from 200-sweep chains | 10 s / ~3 min |
| `$PY -u bench/law_checks.py --world syn --check logz --save F.npz` | exact log Z at fixed (trip, z) pairs, saved for a trip-by-trip comparison | 5 s / 1 min |
| `$PY -u bench/law_checks.py --world syn --check ll --trips 256 --save F.npz` | log-likelihood of held-out baskets at 1,024 draws, saved per basket | ~1 min / ~1 h |
| `$PY bench/compare.py OLD.npz NEW.npz` | per-trip NEW - OLD: min, max, mean and standard error | instant |
| `$PY -u bench/cap_profile.py --world real` | cost of log Z and a sweep at item caps K < nmax, and the log Z each trip loses at one typical z | ~1 min |

Options: `--trips N` (number of held-out trips), `--seed S`, `--threads N`; see `--help`.

## Comparing old and new code

Keep the old code reachable (for example an environment switch read at call time) or
use two git worktrees, and run the same command with each:

```bash
$PY -u bench/law_checks.py --world syn --check logz --save runs/bench/old_logz.npz   # old code
$PY -u bench/law_checks.py --world syn --check logz --save runs/bench/new_logz.npz   # new code
$PY bench/compare.py runs/bench/old_logz.npz runs/bench/new_logz.npz
```

Fixed-z agreement is not enough. The uncapped variant (branch `v4-category-dp`,
commits 31ad866 and b503a4a) matched log Z to 7e-12 at fixed z but its chains ran away (one
real trip went to about 2,660 items in 100 sweeps). Always also run `--check chain` with
several seeds and compare the size paths, and `--check ll` with `compare.py`.

## Baseline (current code, Python 3.13.4, 4 threads, machine idle)

Synthetic (`runs/bench/baseline/syn_timing.log`, `syn_chain.log`):

| measure | value |
|---|---|
| exact log Z, 256 trips | 0.0072 s |
| one Gibbs sweep, 256 trips | 0.0072 s |
| likelihood, 1 basket (128 draws) | 0.265 s |
| likelihood, per basket in 256 | 0.0443 s |
| chain seed 9, mean size after 0/10/50/100/200/400 sweeps | 9.39 / 8.89 / 8.17 / 8.89 / 9.53 / 9.75 (largest 31 = cap at 200) |

Real (`bench/timing.py --world real --quick`, and earlier runs with the same settings):

| measure | value |
|---|---|
| exact log Z, 256 trips | 0.28 s |
| one Gibbs sweep, 256 trips | 0.39-0.49 s |
| likelihood, 1 basket (128 draws) | 3.95 s |
| likelihood, per basket in 256 | 1.46 s |
