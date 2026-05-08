"""Smoke test — proves CI runs Python and can import the project package."""

import importlib


def test_can_import_src() -> None:
    src = importlib.import_module("src")
    assert src.__doc__ is not None


def test_python_version_is_modern_enough() -> None:
    import sys

    assert sys.version_info >= (3, 10)
