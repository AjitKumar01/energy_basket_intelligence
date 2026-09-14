# Interaction recovery and free-embedding pilot

## Why this experiment

The corrected additive fit finished successfully at update 24,500 and restored
its best checkpoint from update 23,700. The subsequent 50,000-context spectral
check rejected all tested ranks 4–8 at the unchanged stability threshold 0.5.
Rank 5 scored 0.4984705; this is not a reason to relax the threshold. Ranks 1–3
were not tested. The stopped run is retained at
`artifacts/corrected_fit_full_20260913/`; its additive fit is not repeated.

The existing production estimator learns a PSD matrix C in a fixed spectral
subspace, Phi Phi' = U C U'. It cannot move product embeddings outside U.
This pilot implements that additional flexibility, but does not replace the
production estimator or certify a nonconvex optimum.

## Changes

- The spectral builder now tests ranks 1–8. Pair-score multiplication uses binary
  basket incidence: Rv = (X'Xv − diag(X'1)v)/M minus the corresponding additive
  expectation. It stores basket items, not explicit product pairs or a J-by-J
  interaction matrix. This is algebraically the same score estimator.
- The recovery uses eight exact additive draws per context instead of two.
  It selects the largest eligible rank on the primary stream, then requires that
  **same rank** to pass one predeclared independent proposal stream (seed 36602).
  Contexts and split halves stay fixed. This isolates proposal noise; it is not
  an independent-data replication. There is no seed search or rank reselection
  on the confirmation stream.
- Rejected production rank artifacts are archived, not deleted or substituted
  with an earlier basis.
- The pilot first fits the existing convex C/size objective on 2,048 training
  contexts, with ridge 0.001 and 100 size-stratified draws per context. It then
  releases the rows of Phi, retaining the fitted size curve and freezing prices,
  additive utilities and category penalties.
- Free refinement uses monotone projected ascent, anchor penalty
  0.01 ||Phi − Phi_fixed||²/2, Gram ridge 0.001 ||Phi'Phi||²/2 and a singular-value
  cap of 1. If the convex fit returns zero Phi, the predeclared initialization is
  0.05 U: the ordinary Phi gradient at exactly zero cannot start learning. The
  anchor remains the zero fitted baseline in that case.
- Basket pair energy and its gradient use sparse incidence, with the diagonal
  self-interaction removed. Exported factors are rotated into leading active
  coordinates before quadrature. Zero-rank comparison uses one exact base node.

## Validation and limitations

Neither embedding optimizer uses validation observations. Two separate proposal
streams evaluate the same 512 validation contexts, with 32 draws per size band.
Both absolute ESS ≥ 2 and ESS/draws ≥ 0.2 must hold in active bands. Reports
include paired household-cluster uncertainty, mean basket size and tail mass.
The finite-bank screen requires optimizer convergence, positive paired
free-minus-fixed lower confidence bounds in both streams, mean gain versus the
additive parent ≥ 0.005, and a paired proposal-stream discrepancy allowance
≤ 0.002 nats. These are predeclared engineering screens, not a joint statistical
confidence guarantee or proof that the Monte Carlo normalizer is unbiased.

Separate Smolyak evaluations compare both checkpoints on a common validation
manifest, with adjacent-rule checks on 128 contexts. The adjacent-rule allowance
is empirical, not a rigorous integration error bound. Recommendation evaluation
uses exact add-one scores over the complete store assortment, reporting MRR and
recall@5/10/20/100, including paired free-versus-fixed and fitted-parent comparisons.
The MRR screen requires a nonnegative paired mean and reports its clustered
interval; it does not claim statistical noninferiority.

Validation was used upstream for additive early stopping. Households can occur
in both training and validation. The untouched test set and full population
tail/generation audits remain necessary for any final model. No checkpoint is
automatically promoted, no test-set tuning is performed, and no pricing policy
is optimized using this experimental model.

## Execution and evidence

The higher-precision primary rank run is logged at:

`artifacts/interaction_recovery_20260913/01_spectral_precision.log`

The follow-up driver waits for that report, confirms the rank, fits both arms,
then runs likelihood and recommendation audits:

```sh
python -u scripts/run_interaction_recovery.py \
  --parent artifacts/corrected_fit_full_20260913/out/v3_pipeline_additive_best.pt \
  --precision-basis artifacts/interaction_recovery_20260913/spectral_precision.npz \
  --output-dir artifacts/interaction_recovery_20260913/followup \
  --wait-for-precision --threads 4
```

Live status is `followup/manifest.json`; the aggregate log is
`followup/pipeline.log`. Source and checkpoint hashes are recorded. Changing
training/evaluation source during this run stops subsequent stages. The output
directory must be new, so an earlier run cannot be silently overwritten.
The source snapshot covers the follow-up stages; the adopted primary worker did
not record a launch-time source snapshot. Its report and basis digests are checked.

Unit tests cover exact enumeration of basket energies and normalizers, analytic
gradient finite differences, equivalence with the projected natural objective,
rotation invariance, active-rank export, zero-rank quadrature, ESS rejection,
low-rank recovery and preservation of failed artifacts. A synthetic missing-
direction test checks improvement against the known complete basket law, not
just training fit. The tests do not prove that refinement will improve retail
data. `final_unit_tests.log` and `final_unit_tests.xml` record the final
117-test pass. Real-data smoke likelihood checks restored both zero-rank and
rank-one checkpoints successfully; the recommendation smoke check also completed.

The 32-training/16-validation real-data smoke runs are plumbing checks only.
Their tiny spectral basis uses a disabled smoke stability gate and must never
be treated as scientific rank evidence. The first smoke run exposed the zero-
Phi initialization case; its failed log was retained. The second exercises the
explicit nonzero initialization and independent evaluation path.
