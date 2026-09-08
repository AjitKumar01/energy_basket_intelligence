# Joint additive--interaction polish: theory and experimental audit

**Branch:** `joint_interaction_polish`

**Base:** `architecture-hardening-cleanup`

**Status:** synthetic efficacy passed; 5,455-product smoke integration passed by safe fallback;
full-data efficacy is not yet established.

## 1. Question

The established pipeline first fits the exact additive Version-4 model and then fits an
interaction matrix in a cross-fitted spectral basis. This is statistically and
computationally stable, but freezing every additive parameter means that an interaction
increment cannot redistribute a small amount of utility back into the additive block.

The experiment asks whether a restricted joint refinement can permit that redistribution
without reopening the non-convex household--product factorization or introducing a
catalogue-sized dense optimization problem.

## 2. Restricted joint model

Let (U\in\mathbb R^{J\times r}) be the accepted interaction basis and write

\[
K=U C U^\top,\qquad C\succeq0.
\]

The existing interaction stage fits (C) and two safe size corrections. The experimental
model also constructs a centred orthonormal basis (V) from the span of (U) and adds

\[
\Delta b_j = v_j^\top a,
\]

where (a\in\mathbb R^r). The basket correction becomes

\[
\Delta E(S)
=\operatorname{tr}\{C F_U(S)\}
+a^\top\sum_{j\in S}v_j
-s_1\frac{|S|}{10}
-s_2\left(\frac{|S|}{10}\right)^2.
\]

Every term is linear in the natural parameters ((C,a,s_1,s_2)). Consequently the
fixed-proposal likelihood-ratio objective remains concave. The PSD and safe-tail
projections are unchanged. After acceptance, the existing product intercept is updated by

\[
\lambda_j\leftarrow\lambda_j+v_j^\top a.
\]

No new term is added to the Version-4 probability law. This only refines an existing
product-utility parameter in a predeclared low-dimensional subspace.

Centred (V) is essential: (sum_j v_j=0) prevents the utility correction from becoming
an unidentified common size tilt already represented by (ho_0).

## 3. Fail-closed selection

The additive polish cannot be accepted merely because the combined interaction model
beats the additive parent. That would allow an additive correction to masquerade as
interaction evidence. The implemented nested procedure is:

1. Select the interaction ridge using the original interaction-plus-size feature set.
2. Fit the restricted model independently on cross-fit halves A and B.
3. At the already selected ridge, fit the joint model on halves A and B.
4. Evaluate joint-minus-restricted paired likelihood gains from A on B and B on A.
5. Accept the polish only when both directional means are positive and the combined paired
   95% lower bound is positive.
6. If either joint solve fails or the evidence gate fails, set (a=0) exactly and emit the
   original restricted interaction solution.

Interaction acceptance, ESS, convergence, complete-support Smolyak evaluation and all
downstream certification gates remain mandatory after this nested decision.

## 4. Synthetic Version-4 basket test

The end-to-end unit experiment uses:

| Quantity | Value |
|---|---:|
| Products | 10 |
| Interaction rank | 2 |
| Nonempty size support | 1--4 |
| Exhaustively enumerated baskets | 385 |
| Training contexts | 1,200 |
| Locked test contexts | 1,200 |
| Additive-parent draws per training context | 64 |

Data are generated from an additive parent plus a known PSD Gram interaction, a known
centred additive redistribution and safe size corrections. Training uses fixed proposal
draws, while test likelihood is computed by exact summation over all 385 baskets. Thus the
reported test comparison contains no partition-function approximation error.

The joint model improved exact held-out likelihood over the interaction-plus-size-only
fit by

\[
0.09876\pm0.01124\quad\text{nats/basket},
\]

with paired 95% interval

\[
[0.07672,\;0.12079].
\]

A second solver-level experiment with 1,800 training and 1,800 held-out contexts recovered
the declared additive coefficients ((0.75,-0.60)) within absolute tolerance 0.08 and
increased held-out likelihood by more than 0.30 nats relative to the restricted fit.

These tests establish that the joint natural coordinates are algebraically correct, the
solver can recover them, and the combined model can outperform a frozen-additive staged
fit when the declared redistribution is genuinely present.

## 5. Null and failure behavior

Automated tests verify that the nested gate rejects a correction when either cross-fit
direction is negative or when the paired uncertainty interval crosses zero.

The repository smoke pipeline was then executed through the interaction and residual-size
stages using the real 5,455-product data representation. The deliberately underpowered
smoke configuration has only 64 contexts and four draws. Its optional joint solve could
not meet the optimizer condition, so the gate rejected it safely, wrote four exactly zero
additive coefficients, retained the original interaction solution, and completed the
stage successfully. This is the intended behavior; smoke mode is a software-path test and
cannot establish efficacy.

The smoke log is written to
`artifacts/joint_interaction_polish_smoke_retry2.log`, and the machine-readable gate is in
`artifacts/candidate.json`. Both directories are intentionally ignored because they contain
machine-specific run artifacts.

## 6. Scalability

The established natural solve stores (r^2+2) sufficient statistics per proposal basket.
The joint polish stores (r^2+r+2). It does not store a (J\times J) matrix.

| Rank | Old width | Joint width | Increment |
|---:|---:|---:|---:|
| 4 | 18 | 22 | 22.2% |
| 5 | 27 | 32 | 18.5% |
| 6 | 38 | 44 | 15.8% |
| 7 | 51 | 58 | 13.7% |
| 8 | 66 | 74 | 12.1% |

For 12,000 contexts, 64 draws and rank 8, the dense sufficient-statistic array increases
from approximately 386.7 MiB to 433.6 MiB in float64. The new catalogue basis requires
(O(Jr)) storage: at (J=5,455,r=8), one basis is about 0.33 MiB. Applying the final
intercept correction costs (O(Jr)) once.

Per proposal basket, the additional statistic costs (O(|S|r)). The existing pair
statistic already requires the selected product rows and an (r\times r) accumulation.
The catalogue therefore enters linearly through basis storage and basket lines; no
quadratic catalogue object is introduced.

## 7. What is and is not established

Established:

- the joint correction remains inside the original Version-4 energy;
- optimization remains concave in the restricted natural coordinates;
- the correction improves exact held-out likelihood when present synthetically;
- unsupported or numerically failed corrections revert automatically to the established
  interaction solution; and
- the additional cost is (O(Jr)) storage and (O(MDr)) sufficient-statistic work.

Not yet established:

- that the real data contain a repeatable additive redistribution after a fully converged
  additive parent;
- that the correction passes the full 12,000-context, 64-draw nested gate;
- that the final combined checkpoint improves locked q-certified validation and test
  likelihood; or
- that recommendation and generation improve.

A full run is justified only after the synthetic and smoke gates above pass—which they now
do. Its acceptance decision must use the same complete-support likelihood, recommendation,
generation and tail-certification pipeline as the established model.
