# RobotGraph component overview

RobotGraph is the execution engine underneath RobotRuntime. RobotRuntime
assembles common robot workflows from graph components. Users can also assemble
a custom graph and run it on the same engine.

## Graph engine

| Component | Responsibility |
| --- | --- |
| Graph definition | Nodes, typed input and output ports, and the connections between them. |
| Timing contracts | Node update rates, message timestamps, input freshness limits, and the policy for missing or stale inputs. |
| Graph validation | Check required connections and port compatibility before starting. Check timing constraints where the graph declares enough information. |
| Local executor | Run nodes in order on one thread or place nodes on separate workers. Start and stop the workers and manage data handoff between them. |
| Transport adapters | Bind selected edges to external transports such as Zenoh or ROS 2. Local graphs do not require these adapters. |

## RobotRuntime API

### Built-in sync, async, and RTC

Keep the existing public API. Callers still choose a mode through
`PolicySource(..., execution=...)`; they don't have to construct a graph or
pass a new backend argument.

```python
from physicalai.runtime import (
    AsyncExecution, PolicySource, RobotRuntime, RTCExecution, SyncExecution,
)

execution = SyncExecution()
# execution = AsyncExecution()
# execution = RTCExecution(fps=30)  # With an RTC-compatible model.

runtime = RobotRuntime(
    robot=robot,
    cameras=cameras,
    fps=30,
    action_source=PolicySource(model=model, execution=execution),
)

with runtime:
    runtime.run(duration_s=60)
```

Internally, RobotRuntime uses the built-in mode and its settings to assemble a
graph for observation reads, inference, and command delivery. Sync preserves
inline inference; async uses a worker and an action queue; RTC uses a worker
and preserves its chunk handoff and timing. The graph executor replaces the
loop for these runs; it doesn't run a second loop inside `PolicySource`.

Existing Python and config inputs, lifecycle, and callbacks stay in place.
Custom `ActionSource` and `Execution` implementations continue on the existing
path unless explicitly adapted; we don't reinterpret them as built-in modes.

### The sync graph underneath RobotRuntime

This is the wiring the built-in `SyncExecution()` recipe would assemble for a
30 Hz run with one wrist camera. On each tick, the loop reads observations,
gets the next policy action, and sends it to the robot. If the policy needs
to refill its action queue, inference runs inline before the command is sent.
All four nodes share one thread.

```python
# Proposed API; camera, robot, and model are already configured.
graph = Graph()
camera_node = graph.add(CameraNode(camera, name="wrist", rate=30))
state_node = graph.add(StateNode(robot, name="state", rate=30))
policy_node = graph.add(PolicyNode(model, name="policy", rate=30, cameras=["wrist"]))
action_node = graph.add(RobotNode(robot, name="commands", rate=30))

graph.connect(camera_node, policy_node, "wrist")
graph.connect(state_node, policy_node, "q")
graph.connect(policy_node, action_node, "action")

loop = Shared("sync")
placement = {node.name: loop for node in (camera_node, state_node, policy_node, action_node)}

# Only needed when supplying the graph yourself.
runtime = RobotRuntime.from_graph(graph, placement=placement)
with runtime:
    runtime.run(duration_s=60)
```

In this proposed API, `RobotRuntime.from_graph()` takes a fully wired graph.
The nodes already bind the camera, robot, and model, so the caller doesn't pass
those again. `RobotRuntime` still owns the session lifecycle and delegates
execution to the graph engine. A built-in recipe supplies the wiring; a custom
graph lets the caller supply it. Both use `runtime.run()` and `runtime.stop()`.
