# Example: observe and run one task in simulation

> Look at the wrist camera in simulation. If the block is visible, move it
> to bin A. Check whether the task succeeded.

The LLM checks the image. If the block is visible, it submits `move_block`
and checks the result; otherwise it doesn't act.

```python
await session.call_tool("observe", {"session_id": "sim", "cameras": ["wrist"]})
# Only after inspecting a usable image; use its returned observation_id.
await session.call_tool("act", {
    "session_id": "sim", "name": "move_block", "args": {"target": "bin_a"},
    "observation_id": "obs-42",
})
# Use the request_id returned by act.
await session.call_tool("status", {"session_id": "sim", "request_id": "task-17"})
```

## Graph setup

Proposed graph API. `sim_camera`, `sim_robot`, `move_block_policy`, and
`sim_limits` are configured simulation fixtures.

```python
graph = Graph()
wrist = graph.add(CameraNode(sim_camera, name="wrist", rate=30))
state = graph.add(StateNode(sim_robot, name="state", rate=30))
goals = graph.add(McpGoalSource(name="goals", rate=10))
skill = graph.add(SkillNode({"move_block": move_block_policy}, name="skill", rate=10))
guard = graph.add(CommandGuard(name="guard", limits=sim_limits, rate=30))
commands = graph.add(RobotNode(sim_robot, name="commands", rate=30))
observations = graph.add(ObservationTap(name="observations", rate=10))
results = graph.add(TaskResultTap(name="results", rate=10))

graph.connect(wrist.output("image"), skill.input("wrist"))
graph.connect(state.output("state"), skill.input("q"))
graph.connect(goals.output("goal"), skill.input("goal"))
graph.connect(skill.output("action"), guard.input("action"))
graph.connect(guard.output("action"), commands.input("action"))

graph.connect(wrist.output("image"), observations.input("wrist"))
graph.connect(state.output("state"), observations.input("q"))
graph.connect(skill.output("result"), results.input("result"))

with RobotRuntime.from_graph(graph) as runtime:
    with RuntimeMCP(
        runtime, session_id="sim", goals=goals, observations=observations, results=results,
    ):
        runtime.run(duration_s=60, logging=True)
```

- `RuntimeMCP` serves the [tools](../runtime-mcp.md); it never calls `step()`.
- `ObservationTap` accepts partial observations; the skill requires usable
  inputs and an accepted goal. `TaskResultTap` reports the skill's outcome.
- The command guard checks actions before delivery. Independent hardware safety
  stays in place.

Once Runtime exposes graph tracing, use it for timing. The [standalone executor trace](../../robotgraph-docs/validation/overview.md) is available now.
For the same tool surface without graph wiring, see [built-in Runtime](mcp-runtime.md).
