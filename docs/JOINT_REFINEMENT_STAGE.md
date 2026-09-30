# Joint refinement of the staged Version-4 model

**Code:** `scripts/version4/joint_refinement.py`, `scripts/version4/fit_joint_refinement.py`;
enabled in the pipeline with `run_pipeline.py --joint-refinement`.

**Status:** the full pipeline, with refinement, passes every certification gate on the
476-product synthetic stress world. The refined model beats the staged model on held-out test
likelihood by **+0.113 nats per basket** (95% interval [0.092, 0.134]). Recommendation, generation
and price counterfactuals are unchanged. On seven exact-likelihood test worlds, refinement never
made a model worse and correctly refused the world with no interaction signal (6 of 6).
Real-data efficacy is not established.

## 1. Question

The staged pipeline (`PIPELINE.md`, stages A–F) fits the Version-4 model in sequence: price
evidence, the exact additive model with \(\Phi=0\), the interaction basis \(U\), the interaction
strengths \(C\) with everything additive frozen, and the household-size block. Each stage is sound
on its own, but a two-step estimator is not the joint maximum-likelihood estimator. Parameters
fitted first cannot adjust to parameters fitted later. For example, product utilities fitted with
\(\Phi=0\) partly absorb co-purchase structure that \(\Phi\) should carry. The gradient of the
interaction term vanishes at \(\Phi=0\), which is why the pipeline starts \(\Phi\) from the
spectral residual rather than training it jointly from scratch.

The question here is whether the staged optimum can be moved to the joint optimum by one extra
stage, with the same frozen-parameter contract for the likelihood, generation and counterfactual
APIs, and at a cost that scales.

## 2. The stage

### 2.1 What moves and what stays

Refinement starts from the staged checkpoint and moves, jointly, the product utilities
\(\lambda\), household tastes \(\theta\) and product tastes \(\alpha\), the category penalties
\(\rho_c\), the size curve \(\rho_0\), and the interaction strengths \(C\) (with \(K=UCU^\top\)).
The price, promotion, season and store blocks and the basis \(U\) stay frozen. So refinement cannot
change price counterfactuals directly; it changes how baskets are composed.

### 2.2 Objective: iterated Monte Carlo maximum likelihood

For a context \(c\) (household, store, week, prices, assortment) the model is
\(P(S\mid c)=\exp E(S)/Z(c)\). The change in log-likelihood from the current parameters to new ones
is \(\Delta E(S^{\rm obs}_c)-\Delta\log Z(c)\), and

\[
\Delta\log Z(c)=\log\,\mathbb E_{S\sim P_{\rm current}(\cdot\mid c)}\,e^{\Delta E(S)} .
\]

Each round draws a **bank** of \(M=32\) baskets per training context from the current model and
maximises

\[
L=\frac1N\sum_c\Big[\Delta E(S^{\rm obs}_c)-\log\frac1M\sum_m e^{\Delta E(S_{cm})}\Big]
-\text{pooling}-\frac{\tau}{N}\,\lVert\text{new}-\text{current}\rVert^2 ,
\]

then redraws the bank from the new model (Geyer and Thompson 1992; Geyer 1994). The bank is
fixed during the solve, so \(L\) is deterministic. Its gradient is the usual observed-minus-expected
statistic, with the bank supplying the expectation. Gradients come from reverse-mode autodiff
through the closed-form energy only. Nothing is differentiated through the sampler or the
normalizer.

**Bank.** Blocked Gibbs on \((S,z)\): \(z\mid S\sim N(\sum_{j\in S}\phi_j, I)\) by the Gaussian
(Hubbard–Stratonovich) identity, and \(S\mid z\) drawn *exactly* by the size/category dynamic
program (size, then category counts, then products within a category by conditional Bernoulli).
Four chains per context start at the observed basket, run 3 burn-in sweeps, then record 4 sweeps × 2
baskets. The sampler is compiled (numba) and parallel over trips, with per-trip counter-based
random streams, so output does not depend on thread count.

**Solve.** Each basket is reduced to sufficient statistics: its product incidence, household,
the upper triangle of \(ss^\top-\sum_j u_ju_j^\top\) (\(s=\sum_j u_j\)), category pair counts and
size. One evaluation of \(L\) and its gradient is then a few sparse products. Blocks
\((C,\lambda,\theta,\rho_c,\rho_0)\) and \(\alpha\) are each concave with the other held fixed and
are solved alternately by full-batch L-BFGS. The statistics are built in chunks of contexts, so
memory does not grow with contexts × assortment.

**Step size.** Each round tries trust weights \(\tau\in\{300,1000\}\). A candidate must pass an
effective-sample-size rule on the reweighted bank (5th percentile ≥ 0.2, median ≥ 0.5). Among the
survivors, the one with the best exact log-likelihood on held-out **selection** trips is kept. If
neither passes the ESS rule, \(\tau=10^4\) and then \(10^5\) are tried. Rounds stop when the
selection score stops improving. Selection trips are disjoint from the gate's validation trips.

**Household-size contract.** The household-size stage (`fit_household_size_rank1.py`) lowers
each household's size coordinate \(\kappa_h\) (the reserved \(\theta\) coordinate) until none of
its training contexts puts more than 35% of its mass on baskets at or above the tail threshold.
Refinement re-fits \(\theta\), so after every round it re-applies that stage's projection, with
the same cap and screen, on every training context: \(\kappa_h\) is lowered only where needed,
and the mean shift moves into \(\rho_0\), so each household's size law is tilted by exactly
\(e^{n\kappa_h}\) and fixed-size composition is unchanged (`--household-size-cap`).

**Acceptance.** Rounds are tried from best validation score down. The first that satisfies all
three conditions is the output; if none does, the staged model stays:
1. the paired validation gain has a 95% interval above zero (household-clustered);
2. the numerical audit one Smolyak level finer agrees within 0.01 nats;
3. it passes the certification population-size audit (`audit_population_size.py`).

### 2.3 Pipeline integration

```
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full \
    --joint-refinement --evaluation-level-offset 3
```

`--joint-refinement` inserts a `refinement` stage between `interaction` and `evaluation`.
Evaluation and certification use `candidate_refined.pt` only when `reports/joint_refinement.json`
records an accepted decision whose parent digest is the staged checkpoint's; otherwise they use
`candidate_rank1.pt`. `--evaluation-level-offset` sets the Smolyak level of the likelihood,
recommendation and population-size gates. The default (rank + 2) is unchanged; the stress world
needs rank + 3.

## 3. Evidence

### 3.1 Exactness

`tests/test_joint_refinement.py`, on an enumerable model with \(\Phi\neq0\):
- basket energy equals enumeration;
- the bank sampler (blocked Gibbs with warm starts) and the compiled \(S\mid z\) sampler draw the
  exact law, and the parallel sampler's output does not depend on thread count;
- the bank's log-ratio converges to the exact normalizer ratio;
- the sufficient-statistic energy equals the slot-level energy in value and gradient (1e-10);
- chunked statistics equal the whole-index statistics;
- a round raises the exact likelihood;
- the household-size shift tilts each household's size law exactly as intended.

On the stress world, a fresh bank estimated the exact change of \(\log Z\) between two
checkpoints to within 0.01 nats (1–2 standard errors), and the energy formulas agreed exactly.

### 3.2 Exact test worlds

Seven worlds with an enumerable basket space (18 products, 100–600 households), each fitted by the
staged procedure. The **gap** is the test log-likelihood of the exact joint MLE minus the staged
fit. Three seeds × {512, 2048} draws per world; production settings (trust 300/1000 with fallback,
at most 5 rounds, validation gate).

| World | Accepted | Test gain over staged (nats/basket) | Share of gap closed |
|---|---|---|---|
| baseline | 6/6 | +0.028 to +0.069 | 67–77% |
| sparse households | 6/6 | +0.028 to +0.100 | 60–73% |
| misspecified taste | 6/6 | +0.026 to +0.075 | 63–73% |
| wrong partition | 6/6 | +0.025 to +0.050 | 69–76% |
| three-way recipes | 6/6 | +0.028 to +0.070 | 64–74% |
| strong interactions | 6/6 | +0.347 to +1.222 | 73–86% |
| **null** (no interactions) | **0/6** | 0 | — |

No accepted case made the test likelihood worse. Draws per context (512 vs 2048) barely matter.
(The exact-world harness is research code and is not part of this repository.)

### 3.3 One complete pipeline on synthetic data

World: `data/stress_world` (476 products, 42,231 trips, mean basket 9.05). It was generated by a
procedural shopping model (missions, nested choice, loyalty, stockpiling, loss-averse reference
prices) that is **not** the Version-4 model, with declared price scenarios whose causal effects
come from re-simulation with common random numbers.

| Check | Staged | Refined (round 4) | |
|---|---|---|---|
| Validation log-likelihood gain (refinement gate) | — | +0.097 [0.058, 0.136] | accepted |
| Test log-likelihood, 4,096 trips, paired | −29.036 | −28.923 | **+0.113** [0.092, 0.134] |
| Numerical audit (level 11 vs 12) | pass | 0.003, pass | |
| Recommendation MRR / recall@10, 1,996 test baskets | 0.237 / 46.4% | 0.240 / 46.9% | same |
| Price scenarios vs truth, own effect: corr / MAE | 0.875 / 0.152 | 0.875 / 0.153 | same |
| same subcategory: corr / MAE | 0.488 / 0.061 | 0.494 / 0.061 | same |
| rest of category: corr / MAE | 0.871 / 0.034 | 0.852 / 0.034 | same |
| Generation: mean size (observed 9.0), category TV, invalid baskets | — | 9.70, 0.087, 0 | pass |
| Population size (26,140 contexts): worst low-observed context's P(20+ items) | 48.3% | 47.1% | pass (< 50%) |
| Certification (all stages) | — | **pass** | |

Refinement ran 5 rounds; the household-size projection lowered \(\kappa_h\) for 4–7 of 2,802
households per round (largest decrease 0.03, 12 s per round), and round 4 was accepted. Wall time
on a 15-core laptop: staged stages 18 min; refinement with gates 16 min; evaluation and
certification 20 min. The commands to reproduce the run are in the repository `README.md`.

### 3.4 Cost and scaling

One refinement round on the stress world (19,513 contexts × 32 baskets) takes about 3 minutes:
bank ≈ 50 s, solve 70–200 s, validation ≈ 12 s. The original implementation took about 14 minutes.
Every speed-up is exact: the compiled parallel sampler (~130× the reference sampler), the
sufficient-statistic solve (3.4×, identical to five digits), and validation batches bounded by
nodes × products × trips.

Staged training time on stress worlds with larger catalogues, about 3,000 households each:

| Products | 476 | 1,039 | 2,511 | 4,789 |
|---|---|---|---|---|
| Staged pipeline | 11 min | 22 min | 46 min | 88 min |
| Live normalizer, one context | 18 ms | 31 ms | 83 ms | — |
| Generate one basket | 5.5 ms | 8 ms | 15 ms | — |

Time grows about linearly with products. The exception is the price stage, which grows faster
because its L-BFGS iteration count grows with the number of products. Refinement memory was
contexts × assortment until the chunked statistics; peak memory at ~5k products with chunking has
not been measured.

## 4. Findings along the way

These shaped the stage. The experiment code is not part of this repository.

1. **Steps sized only by the bank's ESS overfit.** On the stress world, held-out gain went +0.064,
   +0.015, −0.034, −0.095, −0.154 over five rounds. A fresh bank matched the exact normalizer, so
   the sampler was not at fault. Choosing the step on held-out selection trips fixed it: gains then
   rise and level off.
2. **Bank reuse gives nothing.** The chosen step always uses up the bank's headroom, so a
   reweighted old bank never qualified for the next round. Removed.
3. **Extra shrinkage on \(\theta\) never won** a selection.
4. **Refinement undid the household-size stage's cap.** Without the household-size projection,
   rounds 2–4 (+0.092 to +0.097 on validation) put more than half of one or two contexts' mass on
   baskets of 20+ items where the household bought fewer, failed certification, and only round 1
   (+0.077; +0.087 on test) could be accepted. Re-applying the stage's cap every round keeps the
   same validation gains and round 4 passes (§3.3).
5. **Clipping \(C\) after the solve discards most of a step.** The solve wants eigenvalues of
   \(C\) up to 2.8, against the contract \(0\preceq C\preceq I\). Clipping afterwards removed 59% of
   the step's bank gain and shrank baskets by 0.41 items, because the other parameters had been
   fitted for the unclipped \(C\). Enforcing the cap inside the solve removed the clip effect and
   cut the size drift to 0.09 items, but without the household-size cap it still failed the size
   gate after round 1 and was accepted at +0.067, so it was not adopted.
6. **A size anchor did not help.** A penalty keeping each context's expected size and tail
   probability at the staged values was accepted at +0.071 (round 2), not better than +0.077.

## 5. Limits

- **Counterfactuals are unchanged by refinement.** Price parameters are frozen, and the largest
  counterfactual errors come from the model form:
  - substitution within a subcategory is not represented (\(\Phi\) only expresses complements;
    \(\rho_c\) acts per merchandise category), giving correlation 0.49 and a predicted mean of 0;
  - price response is symmetric in log-price, so price rises are under-predicted (true −28% for
    ×1.2, predicted −20%).
- **Rank-8 interactions are capped at \(C\preceq I\)** for normalizer accuracy, and the data
  pushes against the cap.
- The evidence is synthetic: one procedural world, one seed for the full run, about 3,000
  households, merchandise categories as groups.

## 6. Open decisions

1. **Model form** (each is a model change): subcategory substitution, a larger interaction cap
   audited by the finer quadrature rule, and asymmetric price response.
2. **Real data:** none of this has been run on a real retailer's data.

## References

- Geyer, C. J., and Thompson, E. A. (1992). Constrained Monte Carlo maximum likelihood for
  dependent data. *JRSS B* 54, 657–699.
- Geyer, C. J. (1994). On the convergence of Monte Carlo maximum likelihood calculations.
  *JRSS B* 56, 261–274.
- Murphy, K. M., and Topel, R. H. (1985). Estimation and inference in two-step econometric models.
  *JBES* 3, 370–379.
- Newey, W. K., and McFadden, D. (1994). Large sample estimation and hypothesis testing.
  *Handbook of Econometrics* IV, ch. 36.
- Bickel, P. J. (1975). One-step Huber estimates in the linear model. *JASA* 70, 428–434.
- Keshavan, R. H., Montanari, A., and Oh, S. (2010). Matrix completion from a few entries.
  *IEEE Trans. Inf. Theory* 56, 2980–2998.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence.
  *Neural Computation* 14, 1771–1800.
- Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the
  likelihood gradient. *ICML*.
