# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trace a 30 Hz control loop with threshold-based refill on a background thread.

Run from the Runtime repository root:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' python proposal/demos/robotgraph_async_trace_demo.py

Open robotgraph-async-trace.json in https://ui.perfetto.dev (Open trace file).
"""

from _robotgraph_trace_common import AsyncPolicy, ThreadWorker, run_demo


if __name__ == "__main__":
    run_demo(AsyncPolicy(ThreadWorker()), "robotgraph-async-trace.json")
