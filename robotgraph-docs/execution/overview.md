# Execution: a RobotRuntime session

```python
with RobotRuntime.from_graph(graph) as runtime:
    runtime.run(duration_s=60)  # Blocks until duration, stop(), or fatal failure.
```

## Lifecycle

| Phase | Executor |
| --- | --- |
| Prepare | [Compile](../graph/core.md#compile) before scheduling work |
| Start | Open channels and workers; call `start()` once; step ready nodes without waiting for all nodes |
| Run | [Step the plan](../graph/core.md#step-through-the-plan); measure and emit events |
| Stop | Stop scheduling; clean up started nodes and channels after in-flight calls return |

Each `run()` starts a fresh session with no queued work from the previous one.
Exiting the Runtime context closes the outer connection lifecycle.

## Failures and degraded inputs

| Condition | Default for custom graphs |
| --- | --- |
| Node is not ready yet | Wait for that node; keep other ready nodes running. |
| Required input is missing or stale | Skip that step and publish no new output. Keep the rest of the graph running. |
| Step misses its timing target | Record the miss and warn without treating it as fatal. |
| Unhandled node exception or worker death | Fail the run, report the node or worker, and stop scheduling new work. No silent restart. |

- Failed startup: clean up nodes that started successfully.
- Timeouts report failures; they cannot interrupt arbitrary Python calls.
- No new output ≠ previously issued commands revoked. Command validity and stop
  behavior belong to the command path.
- Built-in sync, async, and RTC preserve existing lifecycle and recovery behavior.

## Callbacks and node logging

| Owner | API / output |
| --- | --- |
| Executor | Lifecycle, skip, timing, and failure events |
| Node | Context logging tagged with run and node |
| RobotRuntime | Existing callback dispatcher and action hooks; [opt-in traces](../validation/overview.md) |
