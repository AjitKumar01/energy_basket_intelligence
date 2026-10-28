# Research brief: why the mixture fit gets worse on held-out trips, and the fix

Repository: `/Users/ajit/Projects/nf_dunnhumby/ebi_v4_final`, branch **`v4-mixture`**.
Read first: `docs/why_mixture.pdf` (the model and why it has this form),
`docs/mixture_theory.pdf` (the full theory: prior, law given z, likelihood estimate, training),
`docs/overfitting_theory.pdf` (the analysis so far; its Section 4 fix **did not work**, see §2.4).

This brief is self-contained: everything you need to know about earlier work is in it or in
those three documents.

## 1. The question

The mixture-form basket model (`v4/mixture.py`, fitted by `scripts/fit_mixture.py`) improves on
held-out trips for a while, then gets worse while it keeps improving on training trips. This
happens on synthetic data and on real data. We need:

1. **The cause**, shown by measurement, not by argument alone.
2. **A fix derived from theory**, with the test it must pass stated before it is run.
3. **The fix tested** on the synthetic world against that test.

Theory comes before experiments. Every fit you run must test a stated hypothesis.

## 2. What is already measured

All scores: `research/score_checkpoints.py` on 1,024 trips, K = 256 importance draws, the same
random numbers for every checkpoint, so differences are paired trip by trip. Units: nats per trip.

### 2.1 Synthetic world, plain fit (10 epochs, 1,210 steps, `runs/syn_mixture/mixture_e10*`)

| checkpoint | held-out | training | training − held-out | generated size (data 8.69) |
|---|---|---|---|---|
| step 242 (`mixture.pt`, the 2-epoch run) | −27.29 | −26.76 | 0.54 | 8.57 |
| step 420 (`mixture_e10_best.pt`) | −26.84 | −25.84 | 1.00 | 8.68 |
| step 1,210 (`mixture_e10.pt`) | −27.02 | −24.54 | 2.48 | 7.92 |

From step 420 to 1,210: held-out −0.18 ± 0.04, training +1.31 ± 0.04.
- **By basket size,** the held-out change is +0.17 (1–4 products), +0.08 (5–9), −0.50 (10–19) and
  −2.25 (20 or more). Held-out loss is concentrated in large baskets.
- **By household,** the training gain is +2.46 for households with 0–4 training trips and +1.06
  for those with 12 or more. The held-out change does not depend on trips per household.

### 2.2 Block swap (final checkpoint with one block put back to step 420; `research/swap_blocks.py`)

| block put back | held-out | training |
|---|---|---|
| household (θ_h taste, γ_h price slopes) | +0.43 ± 0.02 | −0.47 ± 0.02 |
| product context (α, β, μ, ζ, display/mailer, recency, u_c, …) | +0.25 ± 0.02 | −0.49 ± 0.02 |
| interactions (φ, ψ, λ) | −0.87 ± 0.05 | −0.70 ± 0.05 |
| size terms (ρ_0, ρ_c, f) | −0.49 ± 0.04 | −0.64 ± 0.04 |

Final checkpoint with only the household block put back: −26.59 held-out, better than the best
checkpoint (−26.84).

### 2.3 Data facts (synthetic world `disc_c48_s3`)

- 7,695 training trips and 1,024 held-out trips.
- 928 households with training trips: median 7 trips each, mean 8.3.
- 522 products: median 50 training purchases, 10% bought 6 times or fewer.
- Each household has 40 free numbers: θ_h (32, the last one a size shift) and γ_h (8).
- The base model (`scripts/fit_additive.py`) regularised these only by early stopping: best at
  iteration 800 of 30,000, weight decay 1e-5.

### 2.4 The tried fix that failed: `--hier-prior` (`v4/hier_prior.py`)

What it does:
- Gives each household and product-context block a Gaussian prior N(m, diag s²), with one mean and
  one variance per column.
- Re-estimates m and s² once per epoch by EM.
- In the EM step, the posterior-variance term uses a curvature from squared batch-sum likelihood
  gradients (an empirical Fisher of the K = 8 importance-bound gradient).

Run `runs/syn_mixture/hier_e10*`, stopped at step 930; `hier_e10_best.pt` is step 420. On 64
held-out trips:

| step | 120 | 240 | 420 | 600 | 900 |
|---|---|---|---|---|---|
| with the prior | −29.61 | −28.93 | **−28.63** | −28.70 | −28.81 |
| plain fit | −29.42 | −28.67 | **−28.34** | −28.40 | — |

- It is about 0.3 worse throughout, and it still peaks at step 420 and declines after.
- The EM `data_weight` (share of each row's value taken from its own data) is about 0.49 for θ
  and α, 0.25–0.33 for μ and ζ, and **0.00 for γ and β**. The price slopes collapsed to their
  block mean, presumably because they enter through a softplus with slope about 0.04 at their values.
- Possible reasons, not verified:
  - the curvature estimate is wrong in scale, so the shrinkage is too strong;
  - the late decline is not mainly household and product memorisation.

### 2.5 Real data (Dunnhumby, commodity partition)

Checkpoints in `runs/commodity/`:
- `v4_mixture_step240.pt`, `v4_mixture_step360.pt` = `v4_mixture_w_best.pt`, and `v4_mixture_w.pt`
  (step 1,244, final).
- All were fitted by the older `fit_mixture.py`. It started from the energy-form checkpoint
  `v4_comp_promo.pt` and trained only λ, φ, ψ, ρ_0, ρ_c, f. Household and context terms were frozen.

| checkpoint | held-out (1,024 trips) | ESS | generated size (data 7.26) |
|---|---|---|---|
| step 240 | −38.10 | 0.35 | 8.02 |
| step 360 | −38.16 | 0.31 | 8.20 |
| step 1,244 | −38.76 | 0.24 | 7.69 |

Final − step 240: −0.66 ± 0.08. **The household and context terms were frozen here, so §2.2 does
not explain this drop.**

## 3. Research questions

1. **Curvature.** For a few households (and a few products), compare the curvature used by
   `v4/hier_prior.py` with the exact curvature: the second derivative of the household's
   log-likelihood summed over all its trips, with respect to θ_h. Compute it with large K, or by
   finite differences of a high-K estimate. Is the empirical Fisher of the K = 8 bound gradient off,
   and by how much?
2. **The late decline with the prior on.** Run the block swap of §2.2 on a hier-prior run (final
   against best) and on the plain run's final against step 242. Which blocks still lose on
   held-out trips?
3. **Large baskets and size drift.** Held-out loss sits in baskets of 10 or more products, and
   generated size drifts down on training trips too (8.17 against 8.74). Is this overfitting, or
   a bias of training on the K = 8 importance bound (it penalises trips with low ESS, which are the
   large ones)? Test by measuring the bound gap log p − L_8 per trip against basket size, and by
   whether training with larger K changes the drift.
4. **Real data.** Which block carries the −0.66 in §2.5? Run the block swap of `v4_mixture_w.pt`
   against `v4_mixture_step240.pt`. Only interaction and size blocks were trained there.
5. **The fix.** From 1–4, derive the fix with its theory. Write the test it must pass before you
   run it. A starting point for the test is the list in Section 6 of `docs/overfitting_theory.pdf`.

## 4. How to run

### 4.1 Environment (already set up on this machine)

- **Python and packages:** Python 3.13.4 (`/Users/ajit/.pyenv/versions/3.13.4/bin/python3`, which is
  `python3` on PATH), torch 2.9.1 (CPU), numpy 2.3.5, pandas 2.3.3, pyarrow 21.0.0.
  `requirements.txt` has the exact pins; on a new machine run `pip install -r requirements.txt`.
- **Compiler:** Apple clang 17 (`clang++`). The two native kernels in `v4/native/` compile on
  first use. Point `V4_BUILD_DIR` at a folder of your own, so you never share a build with a
  running job.
- **Check everything:**
  ```bash
  cd /Users/ajit/Projects/nf_dunnhumby/ebi_v4_final
  git switch v4-mixture
  export V4_BUILD_DIR=$PWD/build/native-research OMP_NUM_THREADS=4
  python3 -m pytest -q tests        # 57 tests, about 30 s the first time (it compiles the kernels)
  ```

### 4.2 Data (read only; do not write into these folders)

| world | bundle (`ENERGY_MODEL_DATA_ROOT`) | initialization / base model |
|---|---|---|
| synthetic `disc_c48_s3` | `/Users/ajit/Projects/nf_dunnhumby/ebi_v4_scale/runs/disc_c48_s3/world/model_input_category` | `runs/syn_giant/init_nmax80.pt`, `runs/syn_giant/additive_nmax80.pt` |
| Dunnhumby, commodity | `/Users/ajit/Projects/nf_dunnhumby/ebi_v4_final/runs/bundle_commodity` | `runs/commodity/initialization.pt`, `runs/commodity/additive/v3_pipeline_additive_best.pt` |

The true law of the synthetic world: `/Users/ajit/Projects/nf_dunnhumby/ebi_v4_scale/runs/disc_c48_s3/world/truth.json`.
Its true within-category competition is 0.05.

### 4.3 Commands (run from the repository root, with 4.1's exports)

Write everything you produce under `runs/research/` (git-ignored). Run long jobs detached, and log line by line (`python3 -u`).

```bash
SYN=/Users/ajit/Projects/nf_dunnhumby/ebi_v4_scale/runs/disc_c48_s3/world/model_input_category
mkdir -p runs/research

# The plain synthetic fit (reproduces runs/syn_mixture/mixture_e10*): about 17 min
ENERGY_MODEL_DATA_ROOT=$SYN nohup python3 -u scripts/fit_mixture.py \
  --initialization runs/syn_giant/init_nmax80.pt --additive runs/syn_giant/additive_nmax80.pt \
  --out runs/research/plain_e10 --recency --centre-history mean --within-competition --promo-common \
  --max-per-category 40 --batch 64 --epochs 10 --val-every 60 --threads 4 \
  > runs/research/plain_e10.out 2>&1 &
tail -f runs/research/plain_e10.log           # validation rows and every-30-step rows
tail -f runs/research/plain_e10.out           # everything, including errors

# Add --hier-prior (and --em-every N) for the fix of section 2.4.  The output files are
# OUT.pt (final), OUT_best.pt (best on 64 held-out trips) and OUT.log.

# Score checkpoints on 1,024 held-out or training trips (paired): about 25 s per checkpoint on the
# synthetic world, about 3.5 min on Dunnhumby
ENERGY_MODEL_DATA_ROOT=$SYN python3 -u research/score_checkpoints.py --split heldout \
  --out runs/research/heldout.pt runs/syn_mixture/mixture_e10_best.pt runs/syn_mixture/mixture_e10.pt
ENERGY_MODEL_DATA_ROOT=$SYN python3 -u research/score_checkpoints.py --split training \
  --out runs/research/training.pt runs/syn_mixture/mixture_e10_best.pt runs/syn_mixture/mixture_e10.pt

# Block swap: copies of A with each block taken from B, then score them with A and B
python3 research/swap_blocks.py --base runs/syn_mixture/mixture_e10.pt \
  --donor runs/syn_mixture/mixture_e10_best.pt --out-dir runs/research/swap_syn
ENERGY_MODEL_DATA_ROOT=$SYN python3 -u research/score_checkpoints.py --split heldout \
  --out runs/research/swap_syn_heldout.pt runs/syn_mixture/mixture_e10_best.pt \
  runs/syn_mixture/mixture_e10.pt runs/research/swap_syn/*.pt

# Real data: the same scripts with
#   ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/ebi_v4_final/runs/bundle_commodity
# A full real-data fit takes about 19 s per step (1,244 steps is about 7 hours); avoid it unless a
# hypothesis needs it.
```

Verified on 2026-10-29: `score_checkpoints.py` reproduces the step-420 held-out score, −26.8393,
exactly.

### 4.4 Where things are in the code

- `v4/mixture.py`
  - `Posterior`: the posterior of z for one trip; one peak, found by Armijo ascent.
  - `log_lik`: the importance estimate of log p(S); with `grad=True` it is the training bound.
  - `generate`: exact draws.
  - `whiten_`: the PX-EM rescaling of φ, ψ.
- `scripts/fit_mixture.py`: the training loop. Moment matching of λ, ρ_0, ρ_c and whitening run
  every `--every` steps, the instrument estimate every epoch, and the `--hier-prior` EM every
  `--em-every` steps.
- `v4/training.py`
  - `add_blocks`: builds the full model from the base model.
  - `penalties`: the fixed priors.
  - `project`: gauges and bounds.
- `v4/hier_prior.py`: the failed fix of §2.4.
- `v4/model.py`: utilities (`utilities`, `slot_utilities_given_z`), the energy, and `basket_field` s(S).
- `tests/test_law.py`: exact checks against enumeration of a tiny world; `tests/tiny.py` builds it.

## 5. Rules

- **Processes:** never stop a process you did not start. In particular, never touch `run_api.py`
  (PID 54614, with children 54617 and 54618) or any other Python process. Stop only by your own PIDs.
- **Resources:** at most 4 threads per job. No `/tmp`: write under `runs/research/`.
- **Other folders:** do not modify `ebi_v4_scale`, `energy_basket_intelligence`, `m4a`,
  `basket_chain` or `ebi_v4x`; read their data only.
- **Data use:** choose between models on validation (held-out training-split trips) only. Never open
  the test weeks.
- **Reporting:** report measured numbers with standard errors, from paired comparisons where
  possible. No hedged averages. Say plainly what was not verified.
- **Theory first:** for every fit you run, state beforehand which hypothesis it tests and what result
  would refute it.
- **Writing for the user:**
  - Use simple words, and keep a strict logical order: what is modelled, then the definitions, then
    the argument, then the measurement.
  - Define every symbol before you use it. Say where a quantity comes from, e.g. "Z(z) is the
    normalising constant of the basket law given z".
  - Explanations and reports go in TeX with a compiled PDF, not in Markdown or chat.
- **Long jobs:** whenever you start one, give the exact `tail -f` command for its logs (absolute paths).
  Report status before waiting on anything long; do not block silently.
- **Logs:** line-buffered (`python3 -u`, `flush=True`). Never pipe a job's output through `grep` into
  a file, and never edit a shell script while bash is running it.

## 6. What to send back

1. A short report, TeX with a compiled PDF in `docs/`. It contains the answer to each question in §3
   with its measurements, the cause, the fix with its derivation, the pre-stated test, and the test
   result.
2. Any new script under `research/`, with how to run it.
3. Code changes to `v4/` or `scripts/` only for the fix itself, with an exact test in `tests/` where
   one is possible. Do not commit; leave the changes in the working tree and list them.
