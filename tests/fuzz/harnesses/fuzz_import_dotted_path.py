# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz import_dotted_path — arbitrary dotted strings must not crash Python;
strings without a dot must raise ValueError.
"""
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from physicalai.inference._importing import import_dotted_path

# Fuzz only known-safe roots to avoid module-level side effects from arbitrary imports
_SAFE_ROOTS = [
    "physicalai.inference.manifest",
    "physicalai.inference.runners",
    "physicalai.inference.preprocessors",
    "physicalai.inference.postprocessors",
    "physicalai.inference.adapters",
    "physicalai.runtime.smoothers",
    "physicalai.runtime.events",
    "json",
    "os.path",
    "collections",
    "abc",
]


def test_one_input(data: bytes) -> None:
    if len(data) < 2:
        return

    fdp = atheris.FuzzedDataProvider(data)
    sub = fdp.ConsumeIntInRange(0, 2)

    if sub == 0:
        # Completely arbitrary string — exercises error paths
        path = fdp.ConsumeUnicodeNoSurrogates(128)
        try:
            import_dotted_path(path)
        except ValueError:
            pass  # Expected for missing dot or non-importable prefix

    elif sub == 1:
        # Rooted at a known-safe module + fuzz suffix
        root = fdp.PickValueInList(_SAFE_ROOTS)
        suffix = fdp.ConsumeUnicodeNoSurrogates(64)
        suffix_clean = "".join(c if c.isidentifier() or c == "." else "_" for c in suffix)
        path = f"{root}.{suffix_clean}" if suffix_clean else root
        try:
            import_dotted_path(path)
        except ValueError:
            pass
        except AttributeError as exc:
            raise AssertionError(
                f"import_dotted_path({path!r}) raised AttributeError ({exc}); expected ValueError"
                ) from None
    else:
        # No-dot string — must always raise ValueError
        path = fdp.ConsumeUnicodeNoSurrogates(64).replace(".", "_")
        try:
            import_dotted_path(path)
        except ValueError:
            return
        # If no ValueError was raised for a string without dots, that's a bug
        if "." not in path:
            raise AssertionError(
                f"import_dotted_path({path!r}) did not raise ValueError for a no-dot string"
            )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
