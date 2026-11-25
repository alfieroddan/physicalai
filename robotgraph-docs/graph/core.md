# Graph core: build, compile, run

`Graph` → `ExecutionPlan` → executor. `RobotRuntime.run()` is the public entry point.

## Build

`add()` / `connect()` / `add_edge()` reject duplicate names, unknown endpoints,
wrong directions, and second producers. Incomplete wiring is allowed here.

## Compile

| Pass | Checks / output |
| --- | --- |
| Nodes | Unique names, valid ports and worker assignments; positive rates for periodic nodes |
| Edges | Registered output → compatible input; one producer per required input, at most one per optional input; errors name both endpoints |
| Order | Topological sort: emit nodes with zero incoming edges, decrement counts for their consumers, repeat. Registration order breaks ties; leftover nodes mean a cycle. |
| Delivery | Worker groups, port routing, in-process or local IPC channels; compatible schema/codec and delivery policy at process boundaries |
| Timing | Valid freshness limits; feasibility checks only under declared assumptions |

Output: a fixed plan; recompile after rewiring. No devices, live channels, or
publisher availability checks. ROS 2-connected hosts compile separately.

## Run

```python
with RobotRuntime.from_graph(graph) as runtime:
    runtime.run(duration_s=60)
```

See [Execution](../execution/overview.md) for startup, stop, and failure handling.

### Step through the plan

Executor pseudocode, per worker:

```python
for node, ctx in due_in_dependency_order(plan, worker):
    if not node.ready(ctx):
        skip(node, "not ready")
        continue
    inputs = read_inputs(plan, node)
    if missing_or_stale_required_input(inputs):
        skip(node, "input unavailable")
        continue
    outputs = measure_step(node, ctx, inputs)  # Calls node.step(ctx, inputs).
    publish(plan, node, outputs)
```

- Dependency order for nodes due together; no barrier across workers.
- No catch-up backlog by default; built-in recipes keep existing behavior.
- Unwired `policy.intent` fails compilation; a wired but empty input skips a step.
