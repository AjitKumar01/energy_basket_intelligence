import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def recovery_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("recovery_test", ROOT / "scripts/run_interaction_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_cannot_reselect_a_different_rank_or_reuse_rng(monkeypatch):
    recovery = recovery_module(monkeypatch)
    report = {"predeclared_stability_threshold": .5,
              "rank_stability": {"1": {"accepted": True}, "2": {"accepted": True}},
              "parent_sha256": "parent", "data_fingerprint_sha256": "data",
              "context_seed": 26601, "contexts": 50000, "draw_seed": 26602}
    second = copy.deepcopy(report)
    second["draw_seed"] = 36602
    assert recovery.confirmed_rank(report, second, 2)
    second["rank_stability"]["2"]["accepted"] = False
    assert not recovery.confirmed_rank(report, second, 2)
    assert recovery.confirmed_rank(report, second, 1)
    second["draw_seed"] = report["draw_seed"]
    assert not recovery.confirmed_rank(report, second, 1)
    second["draw_seed"] = 36602
    second["predeclared_stability_threshold"] = .49
    assert not recovery.confirmed_rank(report, second, 1)


def test_rank_resume_supports_rank_one(monkeypatch, tmp_path):
    from test_pipeline_resurrection import pipeline, install_data_fingerprint
    digest = install_data_fingerprint(monkeypatch, tmp_path)
    basis = tmp_path / "basis.npz"
    np.savez(basis, eigenvectors=np.eye(3)[:, :1])
    basis.with_suffix(".json").write_text(json.dumps({
        "basis_sha256": pipeline.file_sha256(basis), "data_fingerprint_sha256": digest,
        "rank_stability": {"1": {"accepted": True}}}))
    assert pipeline.selected_rank_from_report(basis, maximum_rank=8, dry_run=False) == 1


def test_no_positive_directions_have_zero_explained_product_mass():
    from build_spectral_phi_initialization import mass_counts
    counts, mass = mass_counts(np.empty((5, 0)), np.empty(0))
    assert set(counts.values()) == {0}
    np.testing.assert_array_equal(mass, np.zeros(5))


def test_failed_rank_preserves_new_rejection_evidence(monkeypatch, tmp_path):
    from test_pipeline_resurrection import pipeline
    monkeypatch.setattr(pipeline, "ART", tmp_path)
    canonical = tmp_path / "interaction_basis_rank8.npz"
    canonical.write_bytes(b"previous")

    class Driver:
        dry_run = False

        @staticmethod
        def run(command, allow_failure=False):
            assert allow_failure
            pending = Path(command[command.index("--output") + 1])
            pending.write_bytes(b"rejected new basis")
            pending.with_suffix(".json").write_text('{"largest_stable_rank": null}')
            return 2

    with pytest.raises(SystemExit, match="no prior basis was reused"):
        pipeline.rank_selection(Driver(), tmp_path / "parent.pt", 100)
    assert canonical.read_bytes() == b"previous"
    directories = list(tmp_path.glob("rejected_rank_*"))
    assert len(directories) == 1
    assert len(list(directories[0].iterdir())) == 2
    assert next(directories[0].glob("*.npz")).read_bytes() == b"rejected new basis"


def test_spectral_builder_assesses_small_ranks_without_pair_materialization(monkeypatch, tmp_path):
    import build_spectral_phi_initialization as builder
    import torch

    class Model:
        phi = torch.zeros((16, 4), dtype=torch.float64)
        Kz = 4

    class Batcher:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def make(trips):
            class Index:
                B = len(trips)
                item = torch.arange(16)
            return Index(), {}, {}, np.arange(len(trips))

    data = {"n_item": 16, "n_store": 1,
            "line_ptr": np.arange(17), "line_item": np.arange(16)}
    monkeypatch.setattr(builder, "build", lambda: data)
    monkeypatch.setattr(builder, "load_checkpoint", lambda *_a, **_k: (
        Model(), {"iter": 2, "data_fingerprint_sha256": "data"}, {"nmax": 16}))
    monkeypatch.setattr(builder, "supported_trips", lambda *_a: np.arange(16))
    monkeypatch.setattr(builder, "Features", lambda *_a, **_k: None)
    monkeypatch.setattr(builder, "Batcher", Batcher)
    monkeypatch.setattr(builder, "conditional_slots_repeated", lambda _m, ix, _z, _b, draws, _g:
                        [[torch.tensor([i]) for i in range(ix.B)] for _ in range(draws)])
    monkeypatch.setattr(builder, "leading", lambda *_a: (np.arange(12, 0, -1), np.eye(16)[:, :12]))
    parent = tmp_path / "parent.pt"; parent.write_bytes(b"parent")
    output = tmp_path / "basis.npz"
    monkeypatch.setattr(sys, "argv", ["spectral", "--parent", str(parent), "--trips", "16",
                                     "--rank", "3", "--output", str(output), "--threads", "1"])
    builder.main()
    report = json.loads(output.with_suffix(".json").read_text())
    assert set(report["rank_stability"]) == {"1", "2", "3"}
    assert report["selected_rank"] == 3
    assert report["operator"] == "matrix_free_binary_basket_incidence"
