# Exact synthetic audit of the Version-4 interaction block

## Question

This audit originally asked whether the Version-4 interaction mechanism or its fitting
code can create useful held-out gains when an old 512-trip external-baseline comparison
was underpowered. That external comparison predates the corrected preprocessing and is
retained below only as historical provenance.

The answer from the controlled audit is **no**. The interaction code recovers likelihood,
recommendation accuracy, and the true pair kernel when the data contain detectable
low-rank interactions. The completed corrected-data pipeline now supplies the relevant
isolated real-data comparison: rank-5 interactions improve over the matched exact additive
parent by \(0.021630\pm0.001581\) nats/basket on validation and
\(0.023908\pm0.001647\) on test. Both gains are numerically certified. See
[CORRECTED_PIPELINE_RESULTS.md](CORRECTED_PIPELINE_RESULTS.md).

## Exact synthetic law

The audit uses 14 products, six contexts, four categories, rank three, and support
(1\le |S|\le5). Every one of the

\[
\sum_{n=1}^{5}{14\choose n}=3472
\]

baskets is enumerated. Synthetic data are drawn from the original Version-4 law

\[
p(S\mid x)
\propto
\exp\left\{
\sum_{j\in S}b_j(x)
+\sum_{j<k\in S}\phi_j^\top\phi_k
-\sum_c\rho_c{n_c(S)\choose2}
-\rho_0(|S|)
\right\}.
\]

Consequently, the normalizer, likelihood, and gradients have no Smolyak, QMC, particle,
or Monte Carlo error. Training, validation, and test samples are disjoint. Every fitted
model starts fresh and validation selects its checkpoint.

The comparison contains:

1. a multinomial with empirical (P(n)) and exact additive composition conditional on
   (n);
2. the matched Version-4 additive parent with (Phi=0);
3. a frozen-base interaction fit, which learns (Phi) after fitting the additive parent;
4. a joint interaction fit, which reoptimizes all parameters after inserting (Phi).

## Why complexity alone cannot guarantee a held-out gain

The interaction family contains the additive family because setting (Phi=0) recovers the
additive law. Therefore the unregularized maximum **training** likelihood satisfies

\[
\max_{\Theta,\Phi}\ell_{\mathrm{train}}(\Theta,\Phi)
\ge
\max_{\Theta}\ell_{\mathrm{train}}(\Theta,0).
\]

There is no corresponding theorem for held-out likelihood. When the true interaction is
zero or too weak relative to sampling noise, estimating (Phi) adds variance and may
slightly reduce test likelihood. A useful implementation must therefore show both:

- null calibration: no artificial gain when (Phi_{\rm true}=0); and
- recovery: increasing held-out gain when genuine interaction strength increases.

## Results

Each row below averages three independent experiments with 20,000 training, 5,000
validation, and 10,000 test baskets.

| True interaction strength | Joint − additive nats | Frozen − additive nats | Joint MRR gain | True-kernel correlation |
|---:|---:|---:|---:|---:|
| 0.00 | -0.00086 | -0.00081 | -0.00045 | not identified |
| 0.35 | +0.00220 | +0.00222 | +0.00058 | 0.781 |
| 0.50 | +0.02514 | +0.02349 | +0.00681 | 0.966 |
| 0.70 | +0.08866 | +0.08447 | +0.03151 | 0.993 |

The individual per-seed estimates and standard errors are in
`reports/synthetic_interaction_staging_audit.json`. The main conclusions are:

- the null arm does not manufacture a gain;
- the detection threshold lies between strengths 0.2 and 0.35 in this design;
- at moderate strength, recovered likelihood and MRR improve together;
- the fitted off-diagonal Gram kernel converges to the true kernel;
- freezing the additive block leaves some likelihood unused, but does not explain the
  entire real-data issue.

## What the real-data numbers actually establish

The clean interaction ablation is not the external multinomial. It is the exact
Version-4 parent with every parameter held equal except (Phi=0). On 1,024 validation
trips the corrected child gives

\[
\ell_{\rm child}-\ell_{\Phi=0}
=0.020129\pm0.003995
\]

nats/basket. Its 95% interval is positive. Thus the learned interactions already have a
real held-out effect under the matched comparison.

The external multinomial changes more than interactions: it uses empirical \(P(n)\), has
no \(\rho_0\) joint size block, and has no category-pair block. Its comparison with
Version-4 therefore mixes interaction value with size-law and structural calibration.

On the common 512-trip panel,

\[
\ell_{\rm V4}-\ell_{\rm multi}=0.1107\pm0.1506.
\]

The paired standard deviation is 3.4067 nats. Approximately

\[
\left(\frac{1.96\times3.4067}{0.1107}\right)^2\approx3638
\]

paired trips are needed merely for a zero-excluding 95% interval if the current mean is
stable. Roughly 7,400 are needed for 80% power in a two-sided 5% test. The 512-trip panel
cannot resolve that effect.

## Historical real-data defect

The following diagnosis refers to the earlier projected checkpoint, not to the current
fresh corrected-data fit.

The population-safe checkpoint is not the solution of the constrained training problem.
It was created by projecting an already trained unsafe checkpoint:

- 138 learned category coefficients were changed;
- maximum implied category attraction fell from 576.5 to 1.5 nats;
- only the global \(\rho_0(n)\) block was subsequently reprofiled;
- utilities, household effects, price coefficients, category coefficients, and \(\Phi\)
  were not jointly allowed to compensate.

The resulting parameter vector need not satisfy the first-order conditions of the safe
likelihood. Its 0.18-nat loss versus the unsafe checkpoint is therefore not evidence that
safe Version-4 fitting must lose 0.18 nats.

## Resolution

The next valid real-data experiment is predeclared as follows.

1. Start from the fresh initialization artifact—never the projected checkpoint.
2. Fit the exact (Phi=0) Version-4 likelihood to convergence while enforcing
   
   \[
   (-\rho_c)_+{m_c\choose2}\le1.5
   \]
   
   after every update.
3. Select rank from split-half pair-score stability and initialize (Phi\ne0).
4. Fit the interaction block, then jointly refine the original Version-4 likelihood with
   the same category constraint. Select only by a fixed validation manifest.
5. Compare the final child with its matched (Phi=0) parent to measure interaction value.
6. Compare with the converged external multinomial on at least 7,500 identical held-out
   trips and report paired intervals.
7. Re-run MRR, generation, price counterfactual, q9/q10 fidelity, and the complete
   population tail audit before accepting the checkpoint.

This procedure changes neither the Version-4 energy nor its H--S/ESP theorem. It corrects
optimization, initialization, safety, and statistical power—the parts that the synthetic
audit identifies as relevant.
