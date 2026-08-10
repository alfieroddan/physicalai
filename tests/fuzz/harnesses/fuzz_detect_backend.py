# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz _detect_backend — random file names in a temp dir must either raise ValueError
or return a registered backend name; empty dir must always raise ValueError.
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import atheris

with atheris.instrument_imports():
    from physicalai.inference.model import InferenceModel
    from physicalai.inference.adapters import adapter_registry


def _call_detect_backend(export_dir: Path) -> str:
    """Call InferenceModel._detect_backend via a lightweight mock self."""
    mock = types.SimpleNamespace()
    mock.export_dir = export_dir
    return InferenceModel._detect_backend(mock)  # type: ignore[arg-type]


# Registered backends and extensions — used to validate return values and seed file names
_REGISTERED_BACKENDS: frozenset[str] = frozenset(adapter_registry.names())
_ALL_EXTENSIONS: list[str] = [
    ext
    for name in adapter_registry.names()
    for ext in adapter_registry.extensions_of(name)
]


@atheris.instrument_func
def _sub_random_files(fdp: atheris.FuzzedDataProvider, export_dir: Path) -> None:
    """Fuzz-derived file names placed in the export dir."""
    n_files = fdp.ConsumeIntInRange(0, 8)
    for _ in range(n_files):
        filename = fdp.ConsumeUnicodeNoSurrogates(32)
        # Restrict to printable ASCII and skip path-separator characters to
        # avoid creating subdirectories inside the temp dir.
        safe = "".join(
            c for c in filename if c.isprintable() and c not in r"/\:" and c != "\x00"
        )
        if not safe:
            continue
        try:
            (export_dir / safe).touch()
        except (OSError, ValueError):
            pass  # OS may reject some names; that's fine


@atheris.instrument_func
def _sub_extension_variants(fdp: atheris.FuzzedDataProvider, export_dir: Path) -> None:
    """Files built from real adapter extensions with crafted prefixes."""
    if not _ALL_EXTENSIONS:
        return

    ext = fdp.PickValueInList(_ALL_EXTENSIONS)
    prefix = fdp.ConsumeUnicodeNoSurrogates(16) or "model"
    safe_prefix = "".join(c for c in prefix if c.isalnum() or c in "-_")
    if not safe_prefix:
        safe_prefix = "model"

    filename = f"{safe_prefix}{ext}"
    try:
        (export_dir / filename).touch()
    except (OSError, ValueError):
        pass


def test_one_input(data: bytes) -> None:
    if len(data) < 2:
        return

    fdp = atheris.FuzzedDataProvider(data)
    sub = fdp.ConsumeIntInRange(0, 2)

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp)

        if sub == 0:
            _sub_random_files(fdp, export_dir)
        elif sub == 1:
            _sub_extension_variants(fdp, export_dir)
        # sub == 2: empty directory (no files added)

        try:
            result = _call_detect_backend(export_dir)
        except ValueError:
            return  # Expected for unrecognised / empty dirs

        # Oracle: return value must be a registered backend name.
        assert result in _REGISTERED_BACKENDS, (
            f"_detect_backend returned {result!r} which is not a registered backend. "
            f"Registered: {sorted(_REGISTERED_BACKENDS)}"
        )

    # Sub-target C oracle: empty directory must always raise ValueError.
    if sub == 2:
        with tempfile.TemporaryDirectory() as empty_dir:
            try:
                _call_detect_backend(Path(empty_dir))
            except ValueError:
                return
            raise AssertionError(
                "_detect_backend did not raise ValueError for an empty directory"
            )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
