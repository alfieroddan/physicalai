# RobotGraph integration plan

Order: **standalone graph → built-in RobotRuntime modes → custom pipelines and transport → agentic workflows**. Issues 1–2 are implemented in `physicalai.graph`. The [standalone demos](demos/README.md) validate the direction with fake devices; they are not Runtime integration or a hardware baseline. Issues 3–17 remain proposed unless noted below.

## Epic 1 — Run a graph without RobotRuntime

### Issue 1 — Build the primitives and compile the wiring

**Implemented** in [`physicalai.graph`](../src/physicalai/graph/). The core types do not depend on camera or robot plugins:

| Primitive                                                                       | What to build                                                                                                                                                                                    |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`Graph`](../robotgraph-docs/graph/primitives.md#wire-a-graph)                  | `add()` registers a named node. `connect(src, dst)` is shorthand for `add_edge(Edge(src, dst))`. A graph can be incomplete while you build it.                                                   |
| [`Node`](../robotgraph-docs/graph/primitives.md#node-lifecycle)                 | Declare inputs, outputs, and a positive periodic `rate` for scheduled nodes. `step()` returns outputs keyed by port name.                                                                        |
| [`Port`](../robotgraph-docs/graph/primitives.md#declare-ports-on-a-custom-node) | A read-only endpoint with a node, name, direction, and payload type. `input(name)` and `output(name)` look up existing ports. Required inputs need one producer; optional inputs can be unwired. |
| [`Edge`](../robotgraph-docs/graph/primitives.md#build-an-edge-explicitly)       | Connect one named output to one named input. Port names can differ; types must be compatible. An output can feed several inputs.                                                                 |
| [`Latest`](../robotgraph-docs/timing/overview.md#enforce-at-the-point-of-use)   | Read the newest message and reject it if its age exceeds the edge limit. `Latest` is the only implemented delivery policy, not a transport selector.                                             |
| [`Message[T]`](../robotgraph-docs/transport/primitives.md#message-envelope)     | Carry a typed payload with its source port, sequence number, capture time (if known), and publication time.                                                                                      |
| [`ExecutionPlan`](../robotgraph-docs/graph/core.md#compile)                     | Compile the wiring into a fixed dependency order, node periods, and routes for one worker. Recompile after rewiring.                                                                             |

```python
from physicalai.graph import Graph, Node

class Counter(Node):
    outputs = {"count": int}

    def __init__(self):
        super().__init__("counter", rate=30)
        self.count = 0

    def step(self, ctx, inputs):
        self.count += 1
        return {"count": self.count}


class Recorder(Node):
    inputs = {"sample": int}

    def step(self, ctx, inputs):
        print(inputs["sample"].payload)
        return {}


graph = Graph()
counter = graph.add(Counter())
recorder = graph.add(Recorder("recorder", rate=10))
graph.connect(counter.output("count"), recorder.input("sample"))
# Or use this instead of connect() to set a delivery policy:
# graph.add_edge(Edge(counter.output("count"), recorder.input("sample"),
#                     policy=Latest(max_age=0.2)))
```

**Verification:** compiler tests cover wiring and fan-out. The [counter demo](demos/README.md#1-counter-and-fan-out) runs the plan on one thread.

### Issue 2 — Run the plan on one thread

**Implemented** in [`executor.py`](../src/physicalai/graph/executor.py). Call `start()` once, check `ready()` before reading inputs, and run due nodes once per batch in dependency order. Skip missed ticks rather than replaying them. On exit, stop successfully started nodes in reverse order.

```python
# Standalone runner; no RobotRuntime or hardware.
from physicalai.graph import GraphExecutor, compile_graph

plan = compile_graph(graph)
GraphExecutor(plan).run(duration_s=1, logging=True, trace_path="counter-trace.json")
```

Validate the returned ports and payloads, then send a `Message[T]` to each outgoing edge. Include source port, sequence, and publication time. Keep the original capture time when known; don't substitute the publication time for it.

**Verification:** fake-clock tests cover scheduling, freshness, and failure cleanup. The [trace guide](../robotgraph-docs/validation/overview.md) documents the implemented timing fields.

### Issue 3 — Add basic nodes that wrap existing plugins

Build the first camera → policy → robot pipeline on the standalone single-thread runner. Nodes take already-configured components; they don't implement device drivers or load a second copy of the model.

| Node         | Reuses                                                                                                                                                                                                                                   | Ports                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `CameraNode` | Existing [`Camera`](../src/physicalai/capture/camera.py) implementation or plugin. Read a frame and keep its capture time.                                                                                                               | Output: `image`                                           |
| `StateNode`  | Existing [`Robot`](../src/physicalai/robot/interface.py) plugin. Call `get_observation()` when the policy needs robot state.                                                                                                             | Output: `state`                                           |
| `PolicyNode` | Existing [`InferenceModel`](../src/physicalai/inference/model.py). Build the model observation from the input ports and use `select_action()` for this simple pipeline. It can return a buffered action without running inference again. | Inputs: configured camera names and `q`; output: `action` |
| `RobotNode`  | Same robot plugin as `StateNode`. Check action validity before calling `send_action()`.                                                                                                                                                  | Input: `action`                                           |

```python
# camera, sim_robot, and model are existing configured components.
graph = Graph()
wrist = graph.add(CameraNode(camera, name="wrist", rate=30))
state = graph.add(StateNode(sim_robot, name="state", rate=30))
policy = graph.add(PolicyNode(model, name="policy", rate=30, cameras=["wrist"]))
commands = graph.add(RobotNode(sim_robot, name="commands", rate=30))

graph.connect(wrist.output("image"), policy.input("wrist"))
graph.connect(state.output("state"), policy.input("q"))
graph.connect(policy.output("action"), commands.input("action"))

GraphExecutor(compile_graph(graph)).run(duration_s=1)
```

**Done when:** the graph runs with a configured simulated robot plugin, no `RobotRuntime`, and no new drivers. State and command nodes share one robot connection, and missing or expired actions are not sent. This first pipeline uses a periodic policy node; built-in sync, async, and RTC refill rules come in Epic 2.

### Issue 4 — Run the same graph on separate threads

Keep placement separate from wiring and rate. A local `Latest` edge holds one message; the producer replaces it and the consumer reads it on its own tick. Workers don't wait for each other. No ROS 2, Zenoh, or DDS dependency.

The [async and ring demos](demos/README.md) manage their own model worker. They do not implement executor-managed `Thread()` placement or cross-thread graph edges.

```python
plan = compile_graph(graph, placement={"recorder": Thread()})
GraphExecutor(plan).run(duration_s=1)
```

**Done when:** the same graph runs in either layout, a slow recorder does not stall the counter worker, and an unhandled worker failure ends the run. Also run the plugin example with `placement={"policy": Thread()}` and keep state reads and commands on the default I/O worker.

Design: [Primitives](../robotgraph-docs/graph/primitives.md) · [Compiler and worker loop](../robotgraph-docs/graph/core.md) · [Lifecycle](../robotgraph-docs/execution/overview.md)

## Epic 2 — Move the existing RobotRuntime modes onto the graph

### Issue 5 — Capture the current Runtime baseline

Run sync, async, and RTC on the current backend before changing it. Use the same model, observations, and `fps` for both backends. Use a deterministic fixture to compare action order and a representative simulation run to compare timing.

The fake-device Perfetto traces are illustrations, not this baseline.

**Done when:** there is one repeatable comparison for all three modes. Record control tick delay, inference latency, queue underruns, and stop behavior. Agree on acceptable timing differences before switch-over.

### Issue 6 — Move sync first

Build the graph internally from the existing constructor. Keep the control pass ordered: read observations → request inference if the queue needs refilling → take and send the next action. Sync calls the model inline and keeps `request_threshold`.

```python
# Existing user API: no graph and no new arguments.
with RobotRuntime(
    robot=robot, cameras=cameras, fps=30,
    action_source=PolicySource(
        model=model, execution=SyncExecution(request_threshold=0.5),
    ),
) as runtime:
    runtime.run(duration_s=60)
```

**Done when:** the sync comparison passes with the same public constructor, callback behavior, and control-loop cadence rules.

### Issue 7 — Move async without changing refill behavior

Keep inference off the control worker. The control pass checks `request_threshold` and hands off an observation when the queue needs refilling and the worker is free. The model runs when called, not at a fixed graph rate.

```python
# Existing alternative to SyncExecution in the same Runtime call.
execution = AsyncExecution(request_threshold=0.5)
```

**Done when:** tests cover threshold crossings and a busy inference worker, and the async comparison passes. File any async behavior fixes separately with a reproducer and a regression test.

### Issue 8 — Move RTC without changing chunk handoff

Keep the observation slot and `queue_threshold`. The RTC worker decides when to re-infer and how to hand off chunks. The graph moves observations and work between workers; its edge channels do not replace the action queue.

```python
# Existing API; match RTC fps to RobotRuntime fps.
execution = RTCExecution(fps=30, queue_threshold=queue_threshold)
```

**Done when:** tests cover chunk handoff, reset, and worker failure, and the RTC comparison passes.

### Issue 9 — Switch the built-in modes to the graph backend

Run the same comparison on both backends. Check existing callbacks, config, and affected Studio call sites. Make graph-backed execution the default only after each mode meets the agreed checks; users should not have to change their setup.

Design: [Inference triggers](../robotgraph-docs/graph/core.md#what-drives-inference) · [Timing metrics](../robotgraph-docs/timing/overview.md#measure-and-report)

## Epic 3 — Run custom pipelines and add transport as needed

### Issue 10 — Add `from_graph()` and prove it with local S1/S2

Reuse existing camera and robot plugins. Give S2 the camera image, give S1 the image, robot state, and S2 intent, then send S1 actions to the command node. Run the same graph first on one thread, then with one worker per model. The rates are per node; there is no graph-wide `fps`.

```python
graph = Graph()
wrist = graph.add(CameraNode(camera, name="wrist", rate=30))
state = graph.add(StateNode(robot, name="state", rate=30))
planner = graph.add(S2PlannerNode(planner_model, name="planner", rate=1))
policy = graph.add(S1PolicyNode(policy_model, name="policy", rate=10))
commands = graph.add(RobotNode(robot, name="commands", rate=30))

graph.connect(wrist.output("image"), planner.input("wrist"))
graph.connect(wrist.output("image"), policy.input("wrist"))
graph.connect(state.output("state"), policy.input("q"))
graph.connect(planner.output("intent"), policy.input("intent"))
graph.connect(policy.output("action"), commands.input("action"))

# Omit placement for the one-thread version.
placement = {"planner": Thread(), "policy": Thread()}
with RobotRuntime.from_graph(graph, placement=placement) as runtime:
    runtime.run(duration_s=60)
```

**Done when:** both layouts run in simulation, missing or stale required intent skips S1, and the command path checks whether prior actions are still valid. Keep the full wiring examples in Markdown.

### Issue 11 — Expose the graph trace through RobotRuntime

The standalone executor already writes a bounded, metadata-only Perfetto trace with tick timing, input ages, skips, and overruns. This issue is to expose it through `RobotRuntime.from_graph()` and use it to compare S1/S2 placements.

```python
with RobotRuntime.from_graph(graph, placement=placement) as runtime:
    runtime.run(duration_s=60, logging=True)
```

**Done when:** a Perfetto-style trace shows one lane per worker with step times, input ages, skips, and overruns. Use it to compare the two S1/S2 layouts. Keep buffering bounded and record metadata only, not payloads.

### Issue 12 — Add same-host process delivery

Run the local pipeline with S2 in another process. Use built-in IPC and typed payload codecs; keep the ports and delivery policies the same. Add a shared-memory ring for compatible frames only if measurements show copying is a bottleneck.

The [two-process Zenoh demo](demos/two_process/README.md) proves explicit endpoints between independent local graphs. It does not implement `Process()` placement, middleware-free IPC, or cross-process shared memory. The in-process ring demo passes Python references only.

```python
# Placement sketch; the process API is open.
placement = {"planner": Process(), "policy": Thread()}
with RobotRuntime.from_graph(graph, placement=placement) as runtime:
    runtime.run(duration_s=60)
```

**Done when:** local process delivery works without middleware, preserves message metadata, and rejects a process edge that has no usable schema or codec.

### Issue 13 — Connect two local S1/S2 graphs over ROS 2

Start a graph on each computer. The robot host runs the camera, state, S1, and commands. The compute host runs S2. Add publisher and subscription nodes for wrist frames and intents, not a transport option on every edge.

```python
# Robot host: publish wrist frames and receive S2 intent.
frames_out = robot_graph.add(Ros2PublisherNode(
    name="frames_out", rate=30, topic="/s1s2/wrist", message_type=RosImage,
    qos=camera_qos, encode=frame_to_ros_image,
))
intent_in = robot_graph.add(Ros2SubscriptionNode(
    name="intent_in", rate=30, topic="/s1s2/intent", message_type=RosIntent,
    qos=intent_qos, decode=ros_to_intent,
))
robot_graph.connect(wrist.output("image"), frames_out.input("message"))
robot_graph.connect(intent_in.output("message"), policy.input("intent"))

# Compute host: receive frames and publish intent.
frames_in = compute_graph.add(Ros2SubscriptionNode(
    name="frames_in", rate=30, topic="/s1s2/wrist", message_type=RosImage,
    qos=camera_qos, decode=ros_image_to_frame,
))
intent_out = compute_graph.add(Ros2PublisherNode(
    name="intent_out", rate=1, topic="/s1s2/intent", message_type=RosIntent,
    qos=intent_qos, encode=intent_to_ros,
))
compute_graph.connect(frames_in.output("message"), planner.input("wrist"))
compute_graph.connect(planner.output("intent"), intent_out.input("message"))
```

**Done when:** topics, types, and QoS match; capture timestamps survive the hop; and only frames and intents cross hosts. No remote launcher. Add Zenoh or DDS adapters as follow-up issues when a pipeline needs them.

Design: [Full S1/S2 wiring](../robotgraph-docs/examples/s1s2/index.md) · [Transport](../robotgraph-docs/transport/primitives.md) · [Trace](../robotgraph-docs/validation/overview.md)

## Epic 4 — Let agents author setups and operate them

### Issue 14 — Add authoring skills that reuse existing plugins

Teach the agent to choose between the built-in Runtime and a custom graph. Find the configured camera, robot, and policy first. Write an adapter only if one is missing. Return reviewable Python, config, and a simulation test; leave launch to the operator.

> Use the configured wrist camera and simulated robot. Put a local VLM on its own worker. If it sees a block, send a `move_block` goal for bin A. Show me the wiring before launch.

**Done when:** this prompt produces a setup that reuses the configured plugins and can be reviewed and run in simulation.

### Issue 15 — Add a local VLM goal node

The VLM reads camera and state on its own worker. It returns an allowed goal or nothing. A skill runs the policy, and a command guard checks actions before the robot node sees them. No MCP server needed.

```python
# In a simulation graph with wrist, state, guard, and commands already added.
planner = graph.add(VlmGoalNode(
    vlm, name="planner", rate=1, allowed_skills=("move_block",),
    instruction="If the block is visible, request move_block with target bin_a.",
))
skill = graph.add(SkillNode({"move_block": move_block_policy}, name="skill", rate=10))
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

**Done when:** the simulation test covers no goal, a valid goal, and a rejected goal. The VLM has no motor-command output.

### Issue 16 — Add MCP for built-in Runtime, without a custom graph

Use a Runtime callback to cache observations and register a bounded simulation action. Run the action on a worker so the MCP tools stay available while it runs.

```python
observations = TickSnapshot()
runtime = RobotRuntime(
    robot=sim_robot, cameras={"wrist": sim_camera}, fps=30,
    action_source=PolicySource(model=pick_policy, execution=AsyncExecution()),
    callbacks=[observations],
)

def run_pick():
    runtime.run(duration_s=10)
    return runtime.last_run_reason

with runtime, RuntimeMCP(
    runtime, session_id="sim", observations=observations,
    actions={"run_pick": run_pick},
) as mcp:
    mcp.serve_forever()
```

```python
# Calls from an MCP client.
await session.call_tool("act", {"session_id": "sim", "name": "run_pick", "args": {}})
await session.call_tool("observe", {"session_id": "sim", "cameras": ["wrist"]})
# Use the request_id returned by act, not a hard-coded ID.
await session.call_tool("status", {"session_id": "sim", "request_id": request_id})
# If asked to stop early:
await session.call_tool("stop", {"session_id": "sim"})
```

**Done when:** a client can start a run, read a cached wrist image, check its status, and request a software stop. An empty cache is a valid response. A run ending normally does not prove a pick succeeded.

### Issue 17 — Connect the same MCP tools to a custom graph

Use a goal source for requests, an observation tap for images and state, and a result tap for task outcomes. Runtime owns scheduling; MCP never calls `step()` or sends raw motor commands.

```python
goals = graph.add(McpGoalSource(name="goals", rate=10))
observations = graph.add(ObservationTap(name="observations", rate=10))
results = graph.add(TaskResultTap(name="results", rate=10))
graph.connect(goals.output("goal"), skill.input("goal"))  # Instead of the local VLM.
graph.connect(wrist.output("image"), observations.input("wrist"))
graph.connect(state.output("state"), observations.input("q"))
graph.connect(skill.output("result"), results.input("result"))

with RobotRuntime.from_graph(graph) as runtime:
    with RuntimeMCP(
        runtime, session_id="sim", goals=goals, observations=observations, results=results,
    ):
        runtime.run(duration_s=60)
```

**Done when:** the client can observe, submit an allowed simulation goal, and get the skill's outcome. Reject invalid or unauthorized requests, concurrent requests, and stale or unrelated observation IDs if supplied. `stop` is a software stop, not an emergency stop.

Design: [Authoring](../agent-docs/authoring.md) · [Local VLM](../agent-docs/examples/vlm-node.md) · [MCP tools](../agent-docs/runtime-mcp.md) · [Built-in Runtime](../agent-docs/examples/mcp-runtime.md) · [Custom graph](../agent-docs/examples/sim-task.md)
