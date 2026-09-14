from copy import deepcopy

import pytest

from fit_stratified_natural_interactions import candidate_gate_failures, select_eligible_ridge


GATES = dict(minimum_gain=.005, minimum_half_gain=0., minimum_ess_fraction=.2, minimum_ess=2.)


def candidate(ridge=.01, gain=.012):
    ess = {"minimum_within_band_ess_fraction": .4, "minimum_within_band_ess": 6.4}
    solve = {"converged": True, "accepted_steps_monotone": True}
    return {"ridge": ridge, "mean_crossfit_gain": gain, "minimum_crossfit_gain": gain,
            "a_fit_b": deepcopy(ess), "b_fit_a": deepcopy(ess), "full_fit": deepcopy(ess),
            "solve_a": deepcopy(solve), "solve_b": deepcopy(solve), "full_solve": deepcopy(solve)}


def screen(row):
    row["accepted_for_selection"] = not candidate_gate_failures(row, include_full=True, **GATES)
    return row


def test_unsafe_higher_gain_cannot_displace_safe_regularized_candidate():
    unsafe = candidate(.001, .01988)
    unsafe["a_fit_b"]["minimum_within_band_ess_fraction"] = .1457
    unsafe["b_fit_a"]["minimum_within_band_ess_fraction"] = .1353
    safe = candidate(.01, .012)
    assert select_eligible_ridge([screen(unsafe), screen(safe)]) is safe


@pytest.mark.parametrize("stage", ["a_fit_b", "b_fit_a", "full_fit"])
@pytest.mark.parametrize("field,value", [("minimum_within_band_ess_fraction", .199),
                                          ("minimum_within_band_ess", 1.99)])
def test_absolute_and_fractional_ess_are_required_for_every_stage(stage, field, value):
    row = candidate()
    row[stage][field] = value
    assert candidate_gate_failures(row, include_full=True, **GATES)
    assert select_eligible_ridge([screen(row)]) is None


@pytest.mark.parametrize("stage", ["solve_a", "solve_b", "full_solve"])
def test_unconverged_crossfit_or_full_solver_is_ineligible(stage):
    row = candidate()
    row[stage]["converged"] = False
    assert select_eligible_ridge([screen(row)]) is None


def test_no_eligible_candidate_fails_closed_and_missing_screen_is_not_acceptance():
    assert select_eligible_ridge([candidate()]) is None
    row = candidate(gain=.004)
    assert select_eligible_ridge([screen(row)]) is None
    row = candidate(); row["minimum_crossfit_gain"] = 0
    assert select_eligible_ridge([screen(row)]) is None


def test_valid_rows_are_ranked_by_gain_only_after_all_gates():
    left, right = candidate(.01, .012), candidate(.03, .008)
    assert select_eligible_ridge([screen(right), screen(left)]) is left


def test_reproduction_of_actual_tail_failure_is_fractional_not_absolute():
    row = candidate(gain=.019882)
    row["b_fit_a"] = {"minimum_within_band_ess_fraction": .1352740524,
                       "minimum_within_band_ess": 2.16438484}
    failures = candidate_gate_failures(row, include_full=True, **GATES)
    assert failures == ["b_fit_a: within-band ESS below minimum"]
