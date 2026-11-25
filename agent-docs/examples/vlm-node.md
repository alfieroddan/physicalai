# Example: the VLM as an asynchronous goal source

## User prompt

> Use the configured camera and simulated robot. Write a graph that runs the
> local VLM on its own worker. If it sees a block, send a `move_block` goal
> for bin A. Show me the wiring before launch.

The authoring LLM writes the graph below and leaves launch to the operator.
At runtime the VLM reads the camera and state ports, then emits a goal such
as `{"name": "move_block", "args": {"target": "bin_a"}}`, or no goal if the
block is not visible. It makes no MCP calls.

## Generated graph

Proposed API. `vlm` is configured with a local model artifact and CPU/GPU backend
before launch. It receives observations and emits named goals, not motor commands.
Camera, robot, policy, and command limits are already configured.

```python
graph = Graph()
wrist = graph.add(CameraNode(camera, name="wrist", rate=30))
state = graph.add(StateNode(robot, name="state", rate=30))
planner = graph.add(VlmGoalNode(
    vlm, name="planner", rate=1, allowed_skills=("move_block",),
    instruction="If the block is visible, request move_block with target bin_a.",
))
skill = graph.add(SkillNode({"move_block": move_block_policy}, name="skill", rate=10))
guard = graph.add(CommandGuard(name="guard", limits=limits, rate=30))
commands = graph.add(RobotNode(robot, name="commands", rate=30))

graph.connect(wrist.output("image"), planner.input("wrist"))
graph.connect(state.output("state"), planner.input("q"))
graph.connect(wrist.output("image"), skill.input("wrist"))
graph.connect(state.output("state"), skill.input("q"))
graph.connect(planner.output("goal"), skill.input("goal"))
graph.connect(skill.output("action"), guard.input("action"))
graph.connect(guard.output("action"), commands.input("action"))

with RobotRuntime.from_graph(graph, placement={"planner": Thread(), "skill": Thread()}) as runtime:
    runtime.run(duration_s=60)
```

- The VLM and skill run on separate workers; robot I/O stays on its own loop.
- The skill needs a valid goal and fresh observations. The command guard checks actions.
- No MCP server or graph rewriting at runtime.
