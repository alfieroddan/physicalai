# RobotGraph component overview

`physicalai.graph` is a standalone implementation on this branch. Existing `RobotRuntime` still uses its existing backend.

| Status                                        | Scope                                                                                                                                                               |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implemented                                   | Typed graph, side-effect-free compiler, one calling-thread executor, local `Latest` delivery, opt-in Perfetto traces                                                |
| [Runnable demos](../proposal/demos/README.md) | Fan-out, sync and async queue refill, an in-process ring, and Zenoh within and across processes. The demos use fake devices and manage their own workers/endpoints. |
| [Proposed](../proposal/integration.md)        | Plugin-backed nodes, executor-managed placement and IPC, Runtime migration and `from_graph()`, S1/S2 over ROS 2, agent authoring and MCP                            |

## Graph engine

| Component                                       | API / responsibility                             |
| ----------------------------------------------- | ------------------------------------------------ |
| [Graph](graph/primitives.md)                    | `Graph`, `Node`, `Port`, `Edge`                  |
| [Core](graph/core.md)                           | Compile and step an execution plan               |
| [Timing](timing/overview.md)                    | Ticks, deadlines, freshness                      |
| [Validation and traces](validation/overview.md) | Preflight checks; `run(logging=True)`            |
| [Execution](execution/overview.md)              | Run lifecycle and failures                       |
| [Transport](transport/primitives.md)            | `Message[T]`, local delivery, external endpoints |

## RobotRuntime API

### Built-in sync, async, and RTC today

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

| Execution          | Current behavior to preserve         |
| ------------------ | ------------------------------------ |
| `SyncExecution()`  | Inline inference on queue refill     |
| `AsyncExecution()` | Inference worker + action queue      |
| `RTCExecution()`   | Inference worker + RTC chunk handoff |

No second loop in `PolicySource`. Preserve config, lifecycle, and callbacks.
Custom `ActionSource` and `Execution` stay on the existing path unless adapted.

### Proposed: the sync graph underneath RobotRuntime

One thread, dependency order. The built-in recipe uses Runtime's `fps` for the
control loop. The equivalent custom graph declares node rates explicitly.

```python
# Proposed API; camera, robot, and model are already configured.
graph = Graph()
camera_node = graph.add(CameraNode(camera, name="wrist", rate=30))
state_node = graph.add(StateNode(robot, name="state", rate=30))
policy_node = graph.add(PolicyNode(model, name="policy", rate=30, cameras=["wrist"]))
action_node = graph.add(RobotNode(robot, name="commands", rate=30))

graph.connect(camera_node.output("image"), policy_node.input("wrist"))
graph.connect(state_node.output("state"), policy_node.input("q"))
graph.connect(policy_node.output("action"), action_node.input("action"))

# Only needed when supplying the graph yourself.
with RobotRuntime.from_graph(graph) as runtime:
    runtime.run(duration_s=60)
```

More API examples: [S1/S2 — local, threaded, and ROS 2](examples/s1s2/index.md).
Agent-facing APIs: [Agentic Runtime](../agent-docs/overview.md).
