# Validation and execution traces

## Before running

[`compile()`](../graph/core.md#compile): check nodes, ports, edges, cycles,
placement, and delivery. Timing checks are conditional; no hardware readiness
or deadline guarantee.

## During a run

```python
with RobotRuntime.from_graph(graph) as runtime:
    runtime.run(duration_s=60, logging=True)
```

- **Trace metadata:** topology, run/worker IDs, node/port names, readiness,
  failures, step times, input ages, skips, overruns.
- **Excluded:** payloads and free-text node logs. Callbacks stay unchanged.
- **Buffer:** bounded, off the execution path. Count and report dropped events.

## View the trace

Perfetto flame chart: lane per worker, span per node step. Compare S1/S2 overlap
and delayed or skipped command ticks. Separate host timelines unless clocks
align. Trace path and naming are open.
