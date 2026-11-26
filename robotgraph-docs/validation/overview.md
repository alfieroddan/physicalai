# Validation and execution traces

## Before running

[`compile_graph()`](../graph/core.md#compile) checks nodes, ports, edges, cycles, and declared timing and delivery values. It cannot verify hardware readiness or guarantee deadlines.

## During a run

```python
from physicalai.graph import GraphExecutor, compile_graph

GraphExecutor(compile_graph(graph)).run(
    duration_s=60, logging=True, trace_path="robotgraph-trace.json",
)
```

- **Trace metadata:** process/thread IDs, node names, tick duration, start delay, period, input ages for checked ports, outcome, and missed ticks. Unknown or absent ages are `null`.
- **Worker spans:** use `with ctx.trace_span("Model inference"):` in a node-managed thread. Names must contain metadata only.
- **Excluded:** payloads, exception text, and free-text node logs.
- **Buffer:** bounded in memory, with `dropped_events` reported in the JSON. The executor writes the file after the run, including on failure. Tracing is off by default.

## View the trace

Open the JSON in [Perfetto](https://ui.perfetto.dev). A tick spans the scheduled attempt, including input checks and publication. See [the demo screenshots](../../proposal/demos/README.md) for sync blocking, async overlap, and Zenoh request/reply timing. The two-process demo merges traces on the same host's monotonic clock; this does not align clocks across computers.

Proposed Runtime integration: forward `logging=True` through `RobotRuntime.from_graph()` and compare the threaded S1/S2 layouts. That integration is not implemented.
