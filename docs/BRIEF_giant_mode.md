# Research brief: the fitted model hides a giant-basket mode

Date: 2026-10-27. Repository: this checkout, branch `v4-final` (start a new branch from it).

## 1. The problem in one paragraph

The Version-4 basket model (`docs/THEORY.md`, `docs/textbook.pdf`) is a normalised law over
whole baskets, p(S | x) = exp E(S) / Z, with a rank-16 pair term removed by a Gaussian latent
z. On the Dunnhumby commodity model, for many trips the law puts most of its probability on
baskets at the size cap (120 products), a region far from every real basket. Training does
not see this region, so nothing pushes it down; the held-out log-likelihood estimate usually
does not see it either, so validation reports numbers that are far too good. The model is
therefore wrong as a generator, and its likelihood numbers cannot be trusted. We need a
training procedure and/or model constraint, general rather than tuned to Dunnhumby, under
which the fitted law has no such mode, and a likelihood estimate that would reveal one.

## 2. Evidence (all reproducible, commands in section 5)

**a. Chains started away from the data run to the cap and stay there.** `bench/modes.py`,
64 held-out trips, 200 Gibbs sweeps per start (logs in `runs/bench/modes/`):

| | Synthetic (seed 3, cap 31) | Old real fit `v4_comp_promo.pt` | Refit, epoch 1 `v4_comp_promo_fixed2_best.pt` |
|---|---|---|---|
| chains from z = -1.5 sigma_1 v_1: share at the cap | 0% | 14% | 77% |
| same chains: mean basket size | 9.1 | 34.1 | 113.0 |
| chains from the observed basket: share at the cap | 0% | 3% | 8% |
| trips whose heaviest chain-end region beats the observed-start chain by > 20 nats | 1 | 0 | 17 |
| sigma_1^2 of the pair vectors | 76 | 440 | 361 |

(Real data: 99% of baskets have <= 43 products, median 4, mean 7.65.)

**b. The giant region carries most of the probability.** The z-marginal is
q(z) ~ N(z; 0, I) Z_z, so log w = log Z_z - |z|^2/2 at a chain's end measures how much mass
that region holds (one draw, noisy by a few nats). On 12 trips of the refit, the region at
120 products had log w between +50 and +88 on 9 trips, against about -2 to -7 at the end of
the chain from the observed basket. For the old fit the giant region reached +6 to +13 on 5
of 12 trips. Differences of 50-90 nats are far beyond the noise.

**c. The validation number misses it, except on a few trips.** Held-out log-likelihood by
importance sampling (`bench/law_checks.py --check ll`, 256 trips, 1,024 draws, correct
code): old fit -36.644; refit epoch 1 -40.162. Per trip, the median change is only -0.09,
but 19 trips (7%) are 10-84 nats worse: those are the trips where the importance sampler's
annealed chains found the giant region. On the other trips it was not found, so their
estimates are too high by an unknown amount. The fit's own validation (1,024 trips, 128
draws) reported -42.02 at epoch 1.

**d. What the fitted terms look like (old fit).** The size term rho_0(m) (on the number of
categories present) rises to m = 9, then falls to -26 at 39 categories and -273 at 120, so
many categories are rewarded; 130 of 187 categories have rho_c <= 0 (no cost for several
products of one category); the pair vectors are large (sigma_1^2 = 440).

**e. Not the cause any more:** two numerical errors in the exact normaliser were found and
fixed on 2026-10-26 (`docs/PIPELINE.md` section 7.0; commits f065f37 and eb13d1f). With them
fixed, the training path, the fast path and an independent log-space calculation agree to
1e-10 on all tested (trip, z) pairs. They made the problem visible; they do not explain it.

## 3. Why it can happen (hypotheses, not established)

1. **Training is effectively short-run contrastive divergence.** SML (`scripts/fit_interactions.py`,
   `sml_step`) keeps one persistent chain per training trip, started at the observed basket,
   and advances it 4 sweeps when the trip's batch is visited, i.e. 4 sweeps per epoch. Over
   6 epochs a chain makes about 24 sweeps from the data. The negative phase therefore samples
   near the data and never the giant region, so its weight is never penalised.
2. **The model makes large baskets cheap.** The pair term is 1/2 |s(S)|^2 and grows with the
   square of the basket's summed vector; only the hard cap (120) bounds it. rho_0(m) is free
   and can become a reward; rho_c may be <= 0. Nothing in the model favours small baskets
   once z points along a strong pair direction.
3. **Validation cannot select against it.** The importance proposal is built from four
   chains; the two started at +-1.5 sigma_1 v_1 and annealed sometimes reach the giant region,
   mostly not. Selection on validation then rewards models whose bad region is hard to find.

## 4. What is wanted

- A diagnosis that confirms or rejects the hypotheses above with measurements.
- A fix that is general (not a Dunnhumby special case): for example negative-phase sampling
  that actually explores (more sweeps, chains started from the model rather than the data,
  tempering or parallel tempering), and/or model constraints that keep large baskets from
  being rewarded (a size term that cannot decrease, bounds on pair strength, ...). Choose by
  evidence; say what each option costs.
- A likelihood estimate that would show such a mode (or a separate check that must pass
  before a validation number is trusted), so this cannot hide again.
- Success criteria, measured on held-out trips: `bench/modes.py` shows no region beyond the
  data's sizes holding the mass (share at the cap near 0 from every start, gaps of the
  synthetic model's size), held-out log-likelihood estimated with a proposal that covers the
  model's modes, and basket sizes by period close to the data (`checks/size_by_period.py`).
- Work on synthetic data first (see section 6), then a short real-data run, then propose the
  full real fit (about 15 hours; do not start it, report the command).

## 5. How to run

```bash
cd /Users/ajit/Projects/nf_dunnhumby/ebi_v4_final       # branch v4-final; create your branch from it
export PY=/Users/ajit/Projects/nf_dunnhumby/.bench-python/venv/bin/python3
export V4_BUILD_DIR=$PWD/build/native-research
export SYN_RUN_DIR=/Users/ajit/Projects/nf_dunnhumby/ebi_v4_scale
source bench/env.sh
$PY -m pytest -q tests/                                   # 49 passed
$PY -u bench/modes.py --world syn > runs/bench/modes/syn_mine.log 2>&1
REAL_CK=runs/commodity/v4_comp_promo_fixed2_best.pt $PY -u bench/modes.py --world real > runs/bench/modes/real_refit_mine.log 2>&1
REAL_CK=runs/commodity/v4_comp_promo.pt $PY -u bench/law_checks.py --world real --check ll --trips 256 --save runs/bench/mine_ll.npz > runs/bench/mine_ll.log 2>&1
$PY bench/compare.py runs/bench/verify/real_ll_fixed.npz runs/bench/mine_ll.npz
```

- `bench/README.md` describes every bench script. `runs/` is a link to the run folder; the
  fitted checkpoints are in `runs/commodity/`, the synthetic one in
  `$SYN_RUN_DIR/runs/disc_c48_s3_promo/`.
- The real fit is `run/commodity_comp.sh` (settings in its header); the synthetic worlds are
  built and fitted by `run/small_world.sh`.
- C++ kernels rebuild themselves after a `.cpp` edit (first run silent for 5-10 s).

## 6. Synthetic data first

The synthetic seed-3 world (48 categories, 522 products, cap 31) does not show the problem
(0% at the cap from every start). A synthetic setting that does reproduce it would be the most
valuable first result, for example a world with a larger size cap and catalogue, stronger
pair structure, or the same training schedule over more epochs. Then show the fix removes it
there before touching real data.

## 7. Rules

- At most 4 threads per job; long jobs detached with logs written line by line
  (`$PY -u ... > log 2>&1`; never pipe through grep into a file).
- Never kill a process you did not start; never touch `run_api.py`.
- Do not modify other repositories or the data bundles; write outputs under `runs/`.
- Select only on validation; do not open the test weeks.
- Every exact computation you add needs an enumeration test in `tests/test_law.py`.
- Report exact commands, log paths and measured numbers; say what you did not check.
