# Example: use MCP with built-in RobotRuntime

## User prompt

> In simulation, run the configured pick policy for ten seconds. Show me a
> wrist image while it's running, then tell me whether the run finished.

## What the LLM does

Calls the registered `run_pick` action, checks the wrist image, then polls the
request status. It reports whether the run ended normally, not whether the
pick succeeded. If asked to stop early, it calls `stop`.

```python
await session.call_tool("act", {"session_id": "sim", "name": "run_pick", "args": {}})
await session.call_tool("observe", {"session_id": "sim", "cameras": ["wrist"]})
# Use the request_id returned by act; task-17 is an example.
await session.call_tool("status", {"session_id": "sim", "request_id": "task-17"})
```

If the user asks to stop early:

```python
await session.call_tool("stop", {"session_id": "sim"})
```

## What the operator configures

No graph definition. `TickSnapshot` and `RuntimeMCP` are proposed adapters;
`RobotRuntime` and `PolicySource` are existing APIs. The camera, robot, and
policy are preconfigured simulation fixtures.

```python
observations = TickSnapshot()  # Cache state and frames from Runtime tick events.
runtime = RobotRuntime(
    robot=sim_robot,
    cameras={"wrist": sim_camera},
    fps=30,
    action_source=PolicySource(model=pick_policy, execution=AsyncExecution()),
    callbacks=[observations],
)


def run_pick():
    runtime.run(duration_s=10)
    return runtime.last_run_reason


with runtime, RuntimeMCP(
    runtime, session_id="sim", observations=observations, actions={"run_pick": run_pick},
) as mcp:
    mcp.serve_forever()
```

The operator starts the MCP session; `act` starts the registered Runtime run
on a worker so `observe`, `status`, and `stop` remain available. No frame is
available until the first tick. If `observe` reports no sample, retry while
the run is active.
