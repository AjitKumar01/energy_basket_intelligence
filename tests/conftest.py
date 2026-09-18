"""Shared test fixtures.

The exact dynamic program is a compiled C++ extension that a fresh clone does not have:
it is built into the git-ignored ``artifacts/native/`` by any pipeline run, or directly with

    python scripts/version4/setup_poly_degree_native.py build_ext \
        --build-lib artifacts/native/lib --build-temp artifacts/native/temp

Tests that exercise the native path skip with that instruction instead of failing, so a
fresh clone gets a clean signal from ``pytest -q``.
"""
import pytest

BUILD_COMMAND = ("python scripts/version4/setup_poly_degree_native.py build_ext "
                 "--build-lib artifacts/native/lib --build-temp artifacts/native/temp")


def native_extension():
    from poly_degree_native import _extension
    return _extension


@pytest.fixture()
def native_dp():
    """Skip the test unless the compiled dynamic program is available."""
    extension = native_extension()
    if extension is None:
        pytest.skip(f"native dynamic program is not built; run: {BUILD_COMMAND}")
    return extension
