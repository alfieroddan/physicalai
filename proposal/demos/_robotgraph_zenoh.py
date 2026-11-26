# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The observation and action-chunk exchange shared by the Zenoh demos."""

import json
from collections.abc import Callable
from contextlib import ExitStack
from math import isfinite
from struct import Struct

import zenoh
from _robotgraph_trace_common import CHUNK_SIZE, ActionChunk, ThreadWorker, infer

from physicalai.graph import NodeContext

_KEY = "robotgraph/demo/inference"
_OBSERVATION = Struct("!Qd")  # Frame number and robot position.
_CHUNK = Struct(f"!{CHUNK_SIZE}d")


def peer_config(endpoint: str, *, listen: bool) -> zenoh.Config:
    config = zenoh.Config()
    for key, value in {
        "mode": "peer",
        "scouting/multicast/enabled": False,
        "scouting/gossip/enabled": False,
        "listen/endpoints": [endpoint] if listen else [],
        "connect/endpoints": [] if listen else [endpoint],
    }.items():
        config.insert_json5(key, json.dumps(value))
    return config


def serve_query(ctx: NodeContext, query: zenoh.Query) -> None:
    payload = query.payload
    if payload is None or len(payload) != _OBSERVATION.size:
        query.reply_err(b"Invalid observation size")
        return
    frame, position = _OBSERVATION.unpack(payload.to_bytes())
    if not isfinite(position):
        query.reply_err(b"Invalid robot position")
        return
    try:
        chunk = infer(ctx, frame, position)
    except Exception:
        query.reply_err(b"Fake inference failed")
        raise
    query.reply(_KEY, _CHUNK.pack(*chunk))


def declare_service(session: zenoh.Session, handler: Callable[[zenoh.Query], None]) -> zenoh.Queryable:
    return session.declare_queryable(_KEY, handler)


class ZenohWorker(ThreadWorker):
    def __init__(self, endpoint: str) -> None:
        super().__init__(self._exchange)
        self.endpoint = endpoint

    def start(self, ctx: NodeContext) -> None:
        with ExitStack() as stack:
            self.client = stack.enter_context(zenoh.open(peer_config(self.endpoint, listen=False)))
            super().start(ctx)
            self.sessions = stack.pop_all()

    def _exchange(self, ctx: NodeContext, frame: int, position: float) -> ActionChunk:
        # The blocking Zenoh query runs off the control thread.
        with ctx.trace_span("Zenoh round trip"):
            payload = _OBSERVATION.pack(frame, position)
            for reply in self.client.get(_KEY, payload=payload, timeout=1.0):
                sample = reply.ok
                if sample is not None and len(sample.payload) == _CHUNK.size:
                    return _CHUNK.unpack(sample.payload.to_bytes())
        msg = "Zenoh did not return a valid action chunk"
        raise RuntimeError(msg)

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self.sessions.close()
