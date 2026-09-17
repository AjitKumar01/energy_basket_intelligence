"""Tests for the store availability input: builder rule, feature lookup and model utility."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from audit_probability_foundations import exact_energy, make_world
from canonical_basket_input import CanonicalBasketModelInputBuilder
from features import Features
from interaction_particles import differentiable_log_size_beta0


ROOT = Path(__file__).resolve().parents[1]


def _availability_fixture():
    items = pd.DataFrame({"item_id": [0, 1, 2, 3],
                          "product_id": ["a", "b", "untracked", "self"], "sub_id": [0] * 4})
    rows = []
    for period in range(1, 41):
        rows.append(("a", 0, period, 5, "retail_aggregate"))      # sells from the start
        rows.append(("a", 1, period, 5, "retail_aggregate"))      # store 1 feed
    for period in range(20, 41):
        rows.append(("b", 0, period, 3, "retail_aggregate"))      # launched at period 20
    rows.append(("self", 1, 25, 1, "retail_aggregate"))            # only the cohort's unit
    for period in range(30, 41):
        rows.append(("self", 1, period, 4, "retail_aggregate"))   # others buy from period 30
    prices = pd.DataFrame(rows, columns=["product_id", "store_id", "period", "units",
                                         "price_source"])
    prices["revenue"] = prices.units * 1.0
    baskets = pd.DataFrame({
        "BASKET_ID": [0, 1, 2, 3, 4, 5],
        "item_id": [0, 1, 3, 0, 1, 2],
        "store_id": [0, 0, 1, 1, 0, 0],
        "WEEK_NO": [2, 10, 25, 3, 22, 5],
        "units": [1, 1, 1, 1, 1, 1],
        "split": ["train", "train", "train", "train", "validation", "train"],
    })
    return items, prices, baskets


def test_first_sale_rule_with_censoring_self_confirmation_and_untracked_items(tmp_path):
    items, prices, baskets = _availability_fixture()
    builder = CanonicalBasketModelInputBuilder(
        tmp_path, tmp_path, price_basis="retail", availability="retail_first_sale",
        availability_left_censor_periods=13)
    audit = builder._write_availability(prices, baskets, items, 2, 40, 30, tmp_path)
    with np.load(tmp_path / "availability.npz") as panel:
        first = panel["first_period"]
        log_floor = float(panel["log_floor"])
    assert first[0, 0] == 0 and first[0, 1] == 0          # censored at feed start
    assert first[1, 0] == 20                               # launched after the window
    assert first[1, 1] == 41                               # never sold at store 1
    assert first[2].tolist() == [0, 0]                     # no feed: always available
    assert first[3, 1] == 30                               # cohort's own unit did not confirm
    assert audit["unconfirmed_lines_by_split"] == {"train": 2}
    assert audit["training_lines_unconfirmed"] == 2
    assert audit["pairs_launched_after_training"] == 0
    assert np.isclose(log_floor, np.log(audit["floor"]))
    assert 1e-4 <= audit["floor"] <= 1.0


def test_disabled_availability_writes_no_panel(tmp_path):
    items, prices, baskets = _availability_fixture()
    builder = CanonicalBasketModelInputBuilder(tmp_path, tmp_path, price_basis="retail")
    audit = builder._write_availability(prices, baskets, items, 2, 40, 30, tmp_path)
    assert audit["enabled"] is False
    assert not (tmp_path / "availability.npz").exists()


def test_builder_rejects_unknown_availability_rule(tmp_path):
    with pytest.raises(ValueError, match="availability"):
        CanonicalBasketModelInputBuilder(tmp_path, tmp_path, price_basis="retail",
                                         availability="guess")


def test_log_availability_lookup():
    features = SimpleNamespace(
        availability_enabled=True, log_avail_floor=float(np.log(0.01)),
        avail_first=torch.tensor([[0, 5], [41, 3]]))
    item = torch.tensor([0, 0, 0, 1, 1])
    store = torch.tensor([0, 1, 1, 0, 1])
    week = torch.tensor([1, 4, 5, 40, 3])
    got = Features.log_availability(features, item, store, week)
    assert torch.allclose(got, torch.tensor([0.0, np.log(0.01), 0.0, np.log(0.01), 0.0],
                                            dtype=torch.float64))
    disabled = SimpleNamespace(availability_enabled=False)
    assert torch.equal(Features.log_availability(disabled, item, store, week),
                       torch.zeros(5, dtype=torch.float64))


def test_availability_offset_enters_energy_and_normalizer_exactly():
    model, ix, membership, _ = make_world(
        seed=411, products=7, contexts=3, nmax=4, strength=0.0, kappa=1.0)
    without = model.b_flat(ix).detach().clone()
    base = exact_energy(model, ix, membership)
    offset = torch.where(torch.arange(len(ix.item)) % 3 == 0,
                         torch.tensor(-4.0, dtype=torch.float64),
                         torch.tensor(0.0, dtype=torch.float64))
    model.ctx = {**model.ctx, "log_avail": offset}
    with_avail = model.b_flat(ix).detach()
    assert torch.allclose(with_avail - without, offset)
    native = torch.logsumexp(differentiable_log_size_beta0(model, ix, model.b_flat(ix)), -1)
    dense_offset = torch.zeros(ix.B, model.J, dtype=torch.float64).index_put(
        (ix.item_trip, ix.item), offset)
    dense = torch.logsumexp(base + dense_offset.matmul(membership.T), dim=-1)
    assert torch.allclose(native, dense, atol=2e-10, rtol=2e-10)


def test_every_context_builder_forwards_availability():
    """Any code that gathers features for the model must also forward log availability."""
    for path in ("scripts/version4/fit.py", "scripts/version4/baselines.py",
                 "retail_api/service.py"):
        source = (ROOT / path).read_text()
        assert source.count(".gather(") <= source.count("log_availability("), path
