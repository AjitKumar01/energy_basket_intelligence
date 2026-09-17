import json
from fastapi.testclient import TestClient
import pytest

from retail_api.app import app, get_service
from retail_api.errors import RetailAPIError


class FakeService:
    checkpoint_sha256 = "a" * 64
    checkpoint_blob = {"data_fingerprint_sha256": "b" * 64}

    def capabilities(self):
        return {
            "available": {"basket_completion": {}, "cross_sell": {},
                          "customer_segmentation": {}},
            "unavailable": {"causal_price_optimization": "not validated"},
        }

    def complete(self, request):
        return {
            "estimand": "test conditional law",
            "protocol": request.protocol,
            "evidence_status": "test_only",
            "checkpoint_sha256": self.checkpoint_sha256,
            "revealed_product_ids": request.revealed_product_ids,
            "stop_probability": .25,
            "expected_additional_items": 1.5,
            "size_distribution": [
                {"additional_items": 0, "probability": .25},
                {"additional_items": 1, "probability": .75},
            ],
            "recommendations": [{
                "product_id": 22,
                "internal_item_index": 3,
                "probability_in_completion": .4,
                "department": "GROCERY",
                "commodity": "BREAD",
                "sub_commodity": "WHITE BREAD",
                "brand": "National",
            }],
            "numerical_certificate": {
                "selected_level": 8,
                "selected_nodes": 341,
                "used_followup": False,
                "adjacent_stop_probability_gap": 1e-7,
                "adjacent_expected_size_gap": 1e-5,
                "adjacent_size_probability_gap": 1e-7,
                "adjacent_item_incidence_gap": 1e-7,
                "item_incidence_sum_identity_gap": 1e-14,
            },
            "limitation": "test limitation",
        }

    def household_segment(self, household_index):
        if household_index == 99:
            raise RetailAPIError("outside cohort")
        return {
            "household_index": household_index,
            "segment": 1,
            "label": "weekly pantry",
            "segment_households": 500,
            "mean_price_coefficient": .1,
            "evidence_status": "descriptive_stratification_only",
            "limitation": "not causal",
        }

    def search_products(self, query, limit):
        return [{
            "product_id": 22,
            "internal_item_index": 3,
            "department": "GROCERY",
            "commodity": query.upper(),
            "sub_commodity": "WHITE BREAD",
            "brand": "National",
        }][:limit]


@pytest.fixture()
def client():
    app.dependency_overrides[get_service] = lambda: FakeService()
    with TestClient(app) as value:
        yield value
    app.dependency_overrides.clear()


def test_live_and_ready_are_separate(client):
    assert client.get("/live").json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["checkpoint_sha256"] == "a" * 64


def test_capabilities_disclose_supported_and_blocked_uses(client):
    value = client.get("/v1/capabilities").json()
    assert set(value["available"]) == {
        "basket_completion", "cross_sell", "customer_segmentation"}
    assert "causal_price_optimization" in value["unavailable"]


def test_completion_contract_and_strict_validation(client):
    body = {
        "context": {"kind": "historical_trip", "trip_index": 10},
        "revealed_product_ids": [818980, 818981],
        "top_k": 5,
        "protocol": "uniform_random_subset",
    }
    response = client.post("/v1/baskets/complete", json=body)
    assert response.status_code == 200
    value = response.json()
    assert value["stop_probability"] == .25
    assert value["recommendations"][0]["product_id"] == 22
    assert value["numerical_certificate"]["selected_nodes"] == 341

    body["revealed_product_ids"] = [818980, 818980]
    assert client.post("/v1/baskets/complete", json=body).status_code == 422


def test_retail_context_accepts_independent_source_day_and_week(client):
    response = client.post("/v1/baskets/complete", json={
        "context": {
            "kind": "retail_context", "household_index": 1,
            "store_index": 2, "day": 650, "week": 92,
        },
        "revealed_product_ids": [818980],
    })
    assert response.status_code == 200

    invalid = client.post("/v1/baskets/complete", json={
        "context": {
            "kind": "retail_context", "household_index": 1,
            "store_index": 2, "day": 650, "week": 0,
        },
        "revealed_product_ids": [818980],
    })
    assert invalid.status_code == 422


def test_context_range_follows_the_fitted_dataset_not_a_fixed_calendar():
    from retail_api.service import validate_context_range
    # ERIM-like panel: 357 days, promotion coverage weeks 1..51.
    validate_context_range(0, 1, n_days=357, week_min=1, week_max=51)
    for day, week in ((357, 10), (10, 52)):
        with pytest.raises(RetailAPIError) as error:
            validate_context_range(day, week, n_days=357, week_min=1, week_max=51)
        assert error.value.code == "context_out_of_range"


def test_product_and_segment_endpoints(client):
    products = client.get("/v1/products/search", params={"q": "bread"})
    assert products.status_code == 200
    assert products.json()[0]["commodity"] == "BREAD"
    assert client.get("/v1/households/4/segment").json()["segment"] == 1
    error = client.get("/v1/households/99/segment")
    assert error.status_code == 400
    assert error.json()["error"]["message"] == "outside cohort"


def test_api_uses_the_levels_its_completion_audit_certified():
    from retail_api.service import certified_levels
    assert certified_levels({"numerical_certification": {}}, 8) == [10, 11, 12]
    audit = {"numerical_certification": {"level_offset": 3, "levels": [11, 12, 13]}}
    assert certified_levels(audit, 8) == [11, 12, 13]
    with pytest.raises(RetailAPIError, match="levels"):
        certified_levels(audit, 7)


def test_default_data_root_respects_explicit_configuration(monkeypatch, tmp_path):
    from retail_api.runtime import DEFAULT_DATA_ROOT, apply_default_data_root
    monkeypatch.delenv("ENERGY_MODEL_DATA_ROOT", raising=False)
    monkeypatch.delenv("RETAIL_API_DATA_ROOT", raising=False)
    assert apply_default_data_root() == DEFAULT_DATA_ROOT.resolve()
    monkeypatch.delenv("ENERGY_MODEL_DATA_ROOT")
    monkeypatch.setenv("RETAIL_API_DATA_ROOT", str(tmp_path))
    assert apply_default_data_root() == tmp_path.resolve()
    monkeypatch.setenv("ENERGY_MODEL_DATA_ROOT", "/explicit")
    assert apply_default_data_root() == __import__("pathlib").Path("/explicit")


def verdict_loader(tmp_path, capabilities, **overrides):
    """A bare service object exercising only the capability-verdict loader."""
    import retail_api.service as service_module
    service = service_module.RetailModelService.__new__(service_module.RetailModelService)
    service.checkpoint = tmp_path / "candidate_rank1.pt"
    service.checkpoint_sha256 = "a" * 64
    service.checkpoint_blob = {"data_fingerprint_sha256": "b" * 64}
    document = {"schema_version": 1, "checkpoint_sha256": "a" * 64,
                "data_fingerprint_sha256": "b" * 64, "capabilities": capabilities}
    document.update(overrides)
    (tmp_path / "capability_verdicts.json").write_text(json.dumps(document))
    return service, service_module


def test_decision_capabilities_are_refused_without_a_verdict_file(tmp_path):
    service, module = verdict_loader(tmp_path, {})
    (tmp_path / "capability_verdicts.json").unlink()
    service._load_capability_verdicts()
    report = module.RetailModelService._decision_capabilities(service)
    assert report["available_decisions"] == {}
    assert set(report["unavailable"]) == set(module.DECISION_CAPABILITIES)
    assert report["capability_verdicts"]["source"] is None


def test_a_verdict_file_opens_only_the_capabilities_it_evidences(tmp_path):
    service, module = verdict_loader(tmp_path, {
        "causal_price_optimization": {"status": "supported", "evidence": {"worst_error": 0.03},
                                      "source": "oracle comparison"},
        "promotion_policy": {"status": "untested", "reason": "no policy was scored"}})
    service._load_capability_verdicts()
    report = module.RetailModelService._decision_capabilities(service)
    assert report["available_decisions"]["causal_price_optimization"]["evidence"] == {"worst_error": 0.03}
    assert report["unavailable"]["promotion_policy"] == "no policy was scored"
    assert "assortment_optimization" in report["unavailable"]           # untouched default
    assert report["capability_verdicts"]["source"].endswith("capability_verdicts.json")


@pytest.mark.parametrize("capabilities, overrides, message", [
    ({"causal_price_optimization": {"status": "supported"}}, {}, "without evidence"),
    ({"causal_price_optimization": {"status": "great"}}, {}, "needs a status"),
    ({"teleportation": {"status": "untested"}}, {}, "unknown capabilities"),
    ({}, {"checkpoint_sha256": "0" * 64}, "another checkpoint"),
    ({}, {"data_fingerprint_sha256": "0" * 64}, "another data_fingerprint"),
    ({}, {"schema_version": 2}, "schema_version 1"),
])
def test_bad_verdict_files_are_rejected(tmp_path, capabilities, overrides, message):
    service, _ = verdict_loader(tmp_path, capabilities, **overrides)
    with pytest.raises(RetailAPIError, match=message):
        service._load_capability_verdicts()
