# Execution: a standalone graph session

```python
from physicalai.graph import GraphExecutor, compile_graph

executor = GraphExecutor(compile_graph(graph))
executor.run(duration_s=60)  # Blocks until duration, stop(), or failure.
```

## Lifecycle

| Phase   | Executor                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| Prepare | [Compile](../graph/core.md#compile) before scheduling work                                                   |
| Start   | Create empty local slots and call each node's `start()` in dependency order. Begin scheduling after startup. |
| Run     | [Step the plan](../graph/core.md#step-through-the-plan); measure and emit events                             |
| Stop    | Stop scheduling and call `stop()` on successfully started nodes in reverse order.                            |

The duration starts after startup. Each `run()` has fresh channels and sequence counters. Nodes own their state and any threads or endpoints they start. `executor.stop()` can wake a waiting executor from another thread; it does not interrupt a hook already running.

## Failures and degraded inputs

| Condition                          | Default for custom graphs                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Node is not ready yet              | Wait for that node; keep other ready nodes running.                                                                     |
| Required input is missing or stale | Skip that step and publish no new output. Keep the rest of the graph running.                                           |
| Step misses its timing target      | Record the miss and warn without treating it as fatal.                                                                  |
| Unhandled node exception           | Fail the run with `NodeError` naming the node and hook. A node-managed worker must surface its own errors to the graph. |

- Failed startup: clean up nodes that started successfully.
- Cleanup attempts every started node and preserves the original failure.
- No new output ≠ previously issued commands revoked. Command validity and stop
  behavior belong to the command path.
- Built-in sync, async, and RTC preserve existing lifecycle and recovery behavior.

## Callbacks and node logging

| Owner        | API / output                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------- |
| Executor     | Overrun warnings and [opt-in tick traces](../validation/overview.md)                                |
| Node         | Optional `ctx.trace_span(name)` for metadata-only timing on its own or a node-managed worker thread |
| RobotRuntime | Existing callback dispatcher and action hooks. `from_graph()` and trace delegation are proposed.    |
