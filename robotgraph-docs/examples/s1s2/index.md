# System 1 / System 2: user API examples

Proposed API. Camera, robot, and models are already configured.

## 1. Local: one process, one thread

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

with RobotRuntime.from_graph(graph) as runtime:
    runtime.run(duration_s=60)
```

One thread, no transport setup. Blocking S2 delays S1 and I/O.

## 2. Local: one process, separate threads

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

placement = {"planner": Thread(), "policy": Thread()}
with RobotRuntime.from_graph(graph, placement=placement) as runtime:
    runtime.run(duration_s=60)
```

`Thread()` per model; default I/O thread for camera, state, and commands.
Compare with [`run(logging=True)`](../../validation/overview.md#view-the-trace).

## 3. Separate computers: start a local graph on each host

Start each program on its own host. ROS 2 connects the graphs; no remote launcher.

### Robot computer: camera, state, S1, and commands

```python
graph = Graph()
wrist = graph.add(CameraNode(camera, name="wrist", rate=30))
state = graph.add(StateNode(robot, name="state", rate=30))
policy = graph.add(S1PolicyNode(policy_model, name="policy", rate=10))
commands = graph.add(RobotNode(robot, name="commands", rate=30))

frames_out = graph.add(Ros2PublisherNode(
    name="frames_out", topic="/s1s2/wrist", message_type=RosImage,
    qos=camera_qos, encode=frame_to_ros_image,
))
intent_in = graph.add(Ros2SubscriptionNode(
    name="intent_in", topic="/s1s2/intent", message_type=RosIntent,
    qos=intent_qos, decode=ros_to_intent,
))

graph.connect(wrist.output("image"), frames_out.input("message"))
graph.connect(wrist.output("image"), policy.input("wrist"))
graph.connect(state.output("state"), policy.input("q"))
graph.connect(intent_in.output("message"), policy.input("intent"))
graph.connect(policy.output("action"), commands.input("action"))

with RobotRuntime.from_graph(graph) as runtime:
    runtime.run()
```

### Compute computer: S2 planner

```python
graph = Graph()
planner = graph.add(S2PlannerNode(planner_model, name="planner", rate=1))

frames_in = graph.add(Ros2SubscriptionNode(
    name="frames_in", topic="/s1s2/wrist", message_type=RosImage,
    qos=camera_qos, decode=ros_image_to_frame,
))
intent_out = graph.add(Ros2PublisherNode(
    name="intent_out", topic="/s1s2/intent", message_type=RosIntent,
    qos=intent_qos, encode=intent_to_ros,
))

graph.connect(frames_in.output("message"), planner.input("wrist"))
graph.connect(planner.output("intent"), intent_out.input("message"))

with RobotRuntime.from_graph(graph) as runtime:
    runtime.run()
```

- Match topics, message types (`RosIntent` is shared), and QoS.
- Preserve timestamps. S1 needs a valid intent before publishing actions.
- Only frames and intents cross ROS 2; state and commands stay local.
