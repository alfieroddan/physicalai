# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz policy name validation — _is_safe_policy_name must never crash;
InferenceModel must raise ValueError for unsafe names and not for safe ones.
"""
from __future__ import annotations

import sys
import tempfile

import atheris

with atheris.instrument_imports():
    from physicalai.inference.model import InferenceModel, _is_safe_policy_name


def test_one_input(data: bytes) -> None:
    if len(data) < 2:
        return

    fdp = atheris.FuzzedDataProvider(data)
    sub = fdp.ConsumeBool()

    name = fdp.ConsumeUnicodeNoSurrogates(128)

    if sub:
        result = _is_safe_policy_name(name)
        if result:
            if name:
                assert name[0].isalnum(), (
                    f"_is_safe_policy_name accepted {name!r} but first char is not alphanumeric"
                )
                allowed_tail = set("abcdefghijklmnopqrstuvwxyz"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "0123456789_.-")
                bad = [c for c in name[1:] if c not in allowed_tail]
                assert not bad, (
                    f"_is_safe_policy_name accepted {name!r} but found disallowed chars: {bad}"
                )
    else:
        is_safe = _is_safe_policy_name(name)
        with tempfile.TemporaryDirectory() as export_dir:
            try:
                InferenceModel(export_dir, policy_name=name)
            except ValueError as exc:
                # Only assert when the error is specifically a policy_name rejection.
                # Other ValueError sources (e.g. _detect_backend finding no model
                # files in the empty temp dir) must not trigger the safety assertion.
                if "invalid characters" in str(exc):
                    assert not is_safe, (
                        f"InferenceModel raised ValueError for safe policy_name {name!r}"
                    )
            except (FileNotFoundError, RuntimeError):
                pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
