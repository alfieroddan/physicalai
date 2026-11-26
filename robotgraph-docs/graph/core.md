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

Executor pseudocode for a periodic worker, after startup:

```python
while not stop.requested:
    clock.wait_until(worker.next_due_time(), interruptible_by=stop)
    if stop.requested:
        break
    due = worker.due_nodes(clock.now())  # Snapshot once per pass.

    for node in plan.dependency_order(due):
        if stop.requested:
            break
        ctx = worker.tick_context(node)
        if not node.ready(ctx):
            record_skip(node, "not ready")
        else:
            inputs = read_inputs(plan, node)
            if missing_or_stale_required_input(inputs):
                record_skip(node, "input unavailable")
            else:
                outputs = measure_step(node, ctx, inputs)
                publish_outputs(plan, node, outputs)

        worker.advance_past(node, clock.now())
```

- Run each due node once per pass. On one worker, a consumer can read an
  output produced earlier in the pass. Across workers, there is no barrier.
- `advance_past()` keeps the original cadence and skips ticks already missed,
  even when the step was skipped. No catch-up backlog by default.
- Unwired `policy.intent` fails compilation; a wired but empty input skips a step.
- If `step()` raises, publish no outputs from that step and fail the run by default.

### Publish outputs

`step()` returns payloads keyed by output port name, e.g. `{"action": action}`.
An empty mapping publishes nothing. `publish_outputs()` is an executor operation:

1. Check that returned names are declared output ports and payloads match their contracts.
2. Wrap each payload in a [`Message[T]`](../transport/primitives.md#message-envelope)
   with its source port, sequence number, and publication timestamp. Preserve
   capture time when known; do not substitute publication time for unknown capture time.
3. Hand the message to each outgoing edge's channel. A local `Latest` channel
   replaces its one stored message; the consumer reads it on its own tick.

Publishing never calls a downstream `step()`. A ROS 2 publisher node forwards
messages outside the graph separately from this internal handoff.

### What drives inference?

| Mode | Work trigger |
| --- | --- |
| Custom periodic graph | Each node's declared `rate=` determines when it is due. |
| Built-in sync | `RobotRuntime(fps=...)` drives the ordered read → get action → send pass. Inference runs inline when needed. |
| Built-in async / RTC | The control pass keeps its cadence; the inference worker runs on queue demand, not a fixed model rate. |

```text
control tick → maybe_request(observation) → take next action → send
                      ↓ queue below threshold
               infer inline (sync) or hand work to a worker (async / RTC)
```

Keep the existing thresholds: `request_threshold` for sync/async and
`queue_threshold` for RTC. RTC hands off the observation; its worker decides
when to re-infer. The model performs inference when called; the execution
strategy owns the refill decision. No periodic `rate=` is needed for this work.

An edge channel moves messages; an action queue decides which action comes next.
