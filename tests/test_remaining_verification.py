import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audit_probability_foundations import exact_energy, make_world
from audit_real_price_numerics import action_change, replicate_fidelity
from audit_price_data_provenance import build_events
from evaluate_observational_price_response import exact_parent_incidence
import audit_population_size as population_audit
from audit_population_size import (collect_resilient_size_law,
                                   screen_signature)
from audit_size_drift_decomposition import (
    household_balanced_panel as drift_household_panel,
    load_stage_cache,
    quadrature_chunk,
    save_stage_cache,
)
from run_remaining_verification import run_stage, sources
from interaction_particles import (direct_interaction_particles,
                                   rao_blackwell_expected_size,
                                   rao_blackwell_particle_statistics,
                                   rao_blackwell_selected_incidence)
from run_segment_pricing_mdp import (representative_context_panel, solve_budget_mdp,
                                     independent_policy_evaluation)
from initialize_version4 import resolve_basket_support
from tempered_block_gibbs import conditional_slots_repeated


def action(name, cost, reward):
    return {"action_id": name, "daily_expected_markdown_spend": cost,
            "daily_reward_mean": reward, "daily_reward_lcb95": reward,
            "daily_incremental_list_value_mean": reward,
            "daily_incremental_list_value_lcb95": reward,
            "daily_incremental_post_discount_sales": reward-cost,
            "daily_incremental_distinct_products": reward/10}


def test_zero_utilization_never_forces_a_harmful_promotion():
    answer=solve_budget_mdp([action("none",0,0),action("harm",2,-3)],3,10,100,0)
    assert answer["feasible"]
    assert answer["action_day_counts"]=={"none":3}
    assert answer["expected_markdown_spend"]==0


def test_representative_panel_samples_trips_not_one_trip_per_household():
    data={"trip_split":np.array([2]*7),"trip_nlines":np.ones(7),
          "trip_user":np.array([0,0,0,0,0,1,1])}
    labels=np.array([0,0])
    selected=representative_context_panel(data,labels,0,6,91,120,2)
    assert len(selected)==6
    assert np.all(data["trip_split"][selected]==2)
    assert np.sum(data["trip_user"][selected]==0)>1


def test_price_event_builder_requires_adjacent_observations_and_labels_splits():
    import pandas as pd
    price=pd.DataFrame({"PRODUCT_ID":[10,10,10,20,20],"WEEK_NO":[81,82,84,90,91],
        "n_tx":[5]*5,"n_store":[2]*5,"price":[1,1.2,1.3,2,1.5],
        "base_price":[1,1.2,1.3,2,1.5],"promo_depth":[0,0,0,0,.1]})
    items=pd.DataFrame({"PRODUCT_ID":[10,20],"item_id":[0,1]})
    events=build_events(price,items,{"val_from_week":83,"test_from_week":91,
        "unit_price_source_column":"loyalty_price"})
    assert events[["PRODUCT_ID","WEEK_NO"]].values.tolist()==[[10,82],[20,91]]
    assert events.event_split.tolist()==["train","test"]
    assert events.concurrent_promotion_change.tolist()==[False,True]


def test_independent_policy_evaluation_uses_frozen_counts_and_cluster_se():
    data={"trip_user":np.array([0,0,1,1])}
    row={"bundle":0,"discount":.2,"_per_context":{
        "incremental_size":np.array([1.,2.,3.,4.]),
        "incremental_list_value":np.array([1.,2.,3.,4.]),
        "incremental_post_discount":np.array([.5,1.5,2.5,3.5]),
        "markdown":np.array([.5,.5,.5,.5])}}
    segment={"segment":0,"trips":np.arange(4),"expected_trips_per_day":2.,"actions":[row]}
    scenario={"feasible":True,"budget":20.,
              "action_day_counts":{"segment_0_bundle_0_discount_20":3}}
    answer=independent_policy_evaluation([scenario],[segment],data)[0]
    assert answer["incremental_list_value"]["mean"]==15
    assert answer["markdown_spend"]["mean"]==3


def test_frozen_protocol_has_separate_noncausal_capabilities():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "artifacts" /
            "remaining_verification_20260914" / "verification_protocol.json")
    if not path.is_file():
        pytest.skip("frozen verification protocol artifact is not present in this checkout")
    protocol=json.load(path.open())
    assert protocol["policy"]["profit_capability"] is False
    assert "not_identifiable" in protocol["allowed_statuses"]


def test_uniform_price_actions_are_floating_log_changes():
    # Product identifiers are integer tensors; a full_like call without an explicit
    # floating dtype silently truncates log price changes to zero.
    ix = SimpleNamespace(item=torch.tensor([0, 4, 9], dtype=torch.long))
    for name, multiplier in (("uniform_0.8", .8), ("uniform_1.2", 1.2)):
        change = action_change(name, ix, None, None)
        assert change.dtype == torch.float64
        assert torch.allclose(change, torch.full((3,), math.log(multiplier),
                                                  dtype=torch.float64))
        assert bool((change != 0).all())


def test_particle_estimators_must_match_external_reference_not_only_each_other():
    spec = {"agreement_absolute_items": .02,
            "agreement_combined_mc_se_multiplier": 3}
    # Two identically biased estimators agree with one another, but that is not
    # evidence of fidelity to the fitted-law reference.
    mutually_agreeing = np.full(4, 1.0)
    assert replicate_fidelity(mutually_agreeing-mutually_agreeing, 0.0, spec)["passed"]
    assert not replicate_fidelity(mutually_agreeing, 0.0, spec)["passed"]


def test_parent_incidence_gradient_keeps_context_normalizers_independent(native_dp):
    model, ix, membership, _ = make_world(
        seed=92317, products=7, contexts=3, nmax=4, strength=0)
    item = 0
    with torch.no_grad():
        probability = exact_energy(model, ix, membership).softmax(-1)
        expected = probability @ membership[:, item].to(torch.float64)
    actual = exact_parent_incidence(model, ix, item)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_zero_campaign_budget_returns_no_promotion_without_division_by_zero():
    answer = solve_budget_mdp([action("none", 0, 0)], 28, 0, 4000, 0)
    assert answer["feasible"]
    assert answer["expected_spend_fraction_of_budget"] == 0
    assert answer["action_day_counts"] == {"none": 28}


def test_stage_driver_rejects_success_without_required_report(tmp_path):
    manifest = {"stages": []}
    with pytest.raises(FileNotFoundError, match="required report"):
        run_stage(manifest, tmp_path/"manifest.json", "missing_report",
                  [sys.executable, "-c", "pass"], tmp_path/"stage.log",
                  sources(), report=tmp_path/"absent.json")
    assert manifest["stages"][-1]["status"] == "failed"


def test_tail_cache_signature_binds_data_identity(tmp_path):
    checkpoint = tmp_path/"checkpoint.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    population = np.array([1, 4, 9], dtype=np.int64)
    first = screen_signature(checkpoint, population, 5, [6, 7, 8], "data-a")
    second = screen_signature(checkpoint, population, 5, [6, 7, 8], "data-b")
    assert first != second


def test_confirmation_panel_escalates_only_invalid_context_and_preserves_order(
        monkeypatch):
    def fake_panel(_model, _batcher, trips, rule):
        trips = np.asarray(trips)
        if rule == "q10" and 2 in trips:
            raise FloatingPointError("negative signed size mass")
        observed = trips.astype(np.int64) + 1
        log_probability = np.column_stack((trips, -trips)).astype(np.float64)
        return observed, log_probability

    monkeypatch.setattr(population_audit, "one_size_panel", fake_panel)
    observed, log_probability, used_level = collect_resilient_size_law(
        None, None, np.arange(4), ["q10", "q11"], [10, 11], 4, "test")
    assert observed.tolist() == [1, 2, 3, 4]
    assert log_probability[:, 0].tolist() == [0, 1, 2, 3]
    assert used_level.tolist() == [10, 10, 11, 10]


def test_initialization_support_defaults_to_training_maximum():
    data = {
        "trip_nlines": np.asarray([1, 7, 4, 11, 9]),
        "n_item": 100,
    }
    assert resolve_basket_support(data, np.asarray([0, 1, 2]), None, 120) == (7, 7)
    assert resolve_basket_support(data, np.asarray([0, 1, 2]), 12, 120) == (12, 12)


def test_selected_item_rao_blackwell_matches_full_statistics(native_dp):
    model, ix, _membership, _ = make_world(
        seed=93217, products=7, contexts=3, nmax=4, strength=.7)
    bank = direct_interaction_particles(
        model, ix, 12, torch.Generator().manual_seed(93218))
    selected_slots = torch.stack([
        torch.nonzero((ix.item_trip == b) & (ix.item == 0), as_tuple=True)[0][0]
        for b in range(ix.B)])
    full = rao_blackwell_particle_statistics(
        model, ix, bank.states, bank.log_weights)
    selected = rao_blackwell_selected_incidence(
        model, ix, bank.states, selected_slots, bank.log_weights)
    assert torch.allclose(selected, full.item_incidence[:, 0], atol=1e-12, rtol=1e-12)
    size_axis = torch.arange(1, model.nmax + 1, dtype=torch.float64)
    size_only = rao_blackwell_expected_size(model, ix, bank.states, bank.log_weights)
    assert torch.allclose(size_only, full.size_probability @ size_axis,
                          atol=1e-12, rtol=1e-12)


def test_grouped_repeated_reverse_sampler_matches_exact_base_incidence(native_dp):
    model, ix, membership, _ = make_world(
        seed=94217, products=7, contexts=2, nmax=4, strength=0)
    draws = 20_000
    states = conditional_slots_repeated(
        model, ix, torch.zeros(ix.B, model.Kz), 0.0, draws,
        torch.Generator().manual_seed(94218))
    empirical = torch.zeros(ix.B, model.J, dtype=torch.float64)
    for particle in states:
        for context, slots in enumerate(particle):
            empirical[context, ix.item[slots]] += 1
    empirical /= draws
    exact_probability = exact_energy(model, ix, membership).softmax(-1)
    expected = exact_probability @ membership.to(torch.float64)
    assert torch.allclose(empirical, expected, atol=.02, rtol=0)


def test_drift_household_panel_round_robins_before_reusing_households():
    data = {"trip_user": np.array([0, 0, 0, 1, 1, 2])}
    panel = drift_household_panel(data, np.arange(6), 5, 991)
    first_round = data["trip_user"][panel[:3]]
    counts = np.bincount(data["trip_user"][panel], minlength=3)
    assert set(first_round.tolist()) == {0, 1, 2}
    assert counts.max() - counts.min() <= 1


def test_drift_stage_cache_resumes_only_identical_inputs(tmp_path):
    path = tmp_path / "stage.npz"
    trips = np.array([2, 7, 11], dtype=np.int64)
    expected = np.array([1.5, 2.5, 3.5])
    diagnostic = {"status": "passed", "rank": 5}
    signature = {"checkpoint_sha256": "abc", "tolerance": .02}
    save_stage_cache(path, trips, expected, diagnostic, signature)
    loaded = load_stage_cache(path, trips, signature)
    assert loaded is not None
    assert np.array_equal(loaded[0], expected)
    assert loaded[1] == diagnostic
    assert load_stage_cache(path, trips[::-1], signature) is None
    assert load_stage_cache(path, trips, {**signature, "tolerance": .01}) is None


def test_drift_quadrature_chunk_caps_context_node_product():
    rule = (np.zeros((1341, 5)), np.ones(1341))
    chunk = quadrature_chunk(rule, requested=48, node_context_cap=16_000)
    assert chunk == 11
    assert chunk * len(rule[1]) <= 16_000


def test_size_tail_threshold_follows_training_support_not_a_fixed_size():
    from pipeline_support import size_tail_threshold

    sizes = np.concatenate((np.full(90, 2), np.full(8, 4), np.full(2, 10)))
    data = {
        "trip_split": np.concatenate((np.zeros(100, dtype=int), np.ones(5, dtype=int))),
        "trip_nlines": np.concatenate((sizes, np.full(5, 10))),
    }
    # The 97.5th training percentile is 4, so the tail starts at 5 inside support 1..10.
    assert size_tail_threshold(data, 10) == 5
    # Validation baskets never move the threshold; the support clips it.
    assert size_tail_threshold(data, 4) == 4
    assert size_tail_threshold(data, 10, override=8) == 8
    with pytest.raises(ValueError):
        size_tail_threshold(data, 10, override=60)


def test_heldout_baskets_outside_training_support_are_counted():
    from initialize_version4 import heldout_trips_outside_support
    data = {"trip_nlines": np.array([3, 10, 11, 4, 12]),
            "trip_split": np.array([0, 0, 1, 2, 2])}
    assert heldout_trips_outside_support(data, 10) == {"validation": 1, "test": 1}
    assert heldout_trips_outside_support(data, 12) == {"validation": 0, "test": 0}
