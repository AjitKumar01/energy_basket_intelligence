import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "raw_path_for_test", ROOT / "scripts" / "data" / "raw_path.py")
raw_path = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(raw_path)


def test_standalone_data_stage_finds_repository_local_raw(tmp_path, monkeypatch):
    expected = tmp_path / raw_path.RELEASE_SUBDIRECTORY
    expected.mkdir(parents=True)
    monkeypatch.delenv("NF_RAW_DIR", raising=False)
    monkeypatch.setattr(raw_path, "ROOT", tmp_path)
    assert raw_path.resolve_raw_directory() == expected.resolve()


def test_standalone_data_stage_honors_explicit_raw_path(tmp_path, monkeypatch):
    expected = tmp_path / "licensed-data"
    expected.mkdir()
    monkeypatch.setenv("NF_RAW_DIR", str(expected))
    assert raw_path.resolve_raw_directory() == expected.resolve()
