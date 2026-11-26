# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Send async inference requests and replies through Zenoh on a private Unix socket.

Run from the Runtime repository root on Linux:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' --with 'eclipse-zenoh==1.10.2' python proposal/demos/robotgraph_zenoh_trace_demo.py

Open robotgraph-zenoh-trace.json in https://ui.perfetto.dev (Open trace file).
"""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import zenoh
from _robotgraph_trace_common import AsyncPolicy, run_demo
from _robotgraph_zenoh import ZenohWorker, declare_service, peer_config, serve_query

from physicalai.graph import NodeContext


class _LocalZenohWorker(ZenohWorker):
    def start(self, ctx: NodeContext) -> None:
        with ExitStack() as stack:
            server = stack.enter_context(zenoh.open(peer_config(self.endpoint, listen=True)))
            stack.enter_context(declare_service(server, lambda query: serve_query(ctx, query)))
            super().start(ctx)
            self.server = stack.pop_all()

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self.server.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="robotgraph-zenoh-") as directory:
        endpoint = f"unixsock-stream/{Path(directory) / 'peer.sock'}"
        run_demo(AsyncPolicy(_LocalZenohWorker(endpoint)), "robotgraph-zenoh-trace.json")
