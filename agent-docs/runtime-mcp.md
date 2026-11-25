# Runtime MCP API

Proposed tools for an external harness and an operator-configured session.
No custom `Graph` required. A [VLM node](examples/vlm-node.md) uses graph
ports directly and doesn't need MCP.

See [Robot agent harness](harness.md) for client setup and the model/tool loop.

| Tool | Input | Result |
| --- | --- | --- |
| `observe` | `session_id`, camera names | Observation ID, available images/state, capture times, and missing or stale inputs |
| `act` | `session_id`, registered action name, typed arguments, optional `observation_id` | Accepted request ID or rejection reason |
| `status` | `session_id`, optional `request_id` | Session readiness or request state: queued, running, succeeded, failed, or cancelled |
| `stop` | `session_id` | Acknowledgement of a Runtime stop request; use `status` to confirm shutdown |

## Where the tools connect

| Setup | `observe` reads | `act` calls |
| --- | --- | --- |
| [Built-in Runtime](examples/mcp-runtime.md) | Latest data cached by a Runtime callback | Registered handler, e.g. start a bounded policy run |
| [Custom graph](examples/sim-task.md) | Observation tap | Goal source connected to a skill |

## Tool boundaries

- `observe` reads cached data; no sample yet is a valid result. It does not
  open cameras or force a step. Capture time is not the tool response time.
- `act` submits a registered request. Validate
  arguments and permissions at admission. If `observation_id` is supplied,
  reject stale or unrelated observations. Reject concurrent requests in the
  first version. No arbitrary joint arrays, Python code, or driver calls.
- `queued` is not `succeeded`. The handler or skill defines success; a
  policy run ending normally is not proof that the object moved.
- MCP never sends raw motor commands. `stop` requests a software stop, not
  an emergency stop.
