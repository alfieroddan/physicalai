# MHS and Agentic Runtime

**Model Hardware Standard (MHS)** lets software find devices, read sensors, write controls, call procedures, and subscribe to events. The driver decides what each operation does.

## Start here: model → MHS → hardware

Wrap existing camera and robot plugins in `MhsDriver` adapters. The model picks discrete operations; the device or its controller executes them. **No `RobotRuntime`, policy, or graph is needed.** The model does not send a command every tick.

```text
Frontier model → tool client (MCP or Python) → MHS → camera / robot adapters
```

### Expose an operation in Python (WIRE)

```python
from mhs.mhs_api import MhsDriver, rpc, run_device


class SimRobot(MhsDriver):
    device_type = "robot"
    pose = "home"

    @rpc()
    def move_to_named_pose(self, pose: str) -> dict:
        if pose not in {"home", "inspection"}:
            raise ValueError("Unknown pose")
        self.pose = pose  # Simulation. Hardware uses a controller's move operation.
        return {"pose": self.pose}


# With a configured broker and tenant:
# run_device(SimRobot(), broker=broker, tenant=tenant, device_id="robot-01")
```

A real adapter can also expose robot status or a preview from a camera plugin. A named move needs a controller or plugin that supports it; it is not part of the generic `Robot` API. For a long move, return a request ID and expose `status` separately.

### Call it with or without MCP

```python
from mhs.mhs_api import connect_fleet


async def inspect(broker, tenant):
    async with connect_fleet(broker=broker, tenant=tenant, discovery="presence") as fleet:
        print(await fleet.discover("device(robot-01)"))
        return await fleet.invoke("robot-01", "move_to_named_pose", pose="inspection")
```

```python
# MCP client sketch: exact tool names depend on the bridge configuration.
preview = await mcp.call_tool("get_snapshot", {"camera": "wrist"})
move = await mcp.call_tool("move_to_named_pose", {"pose": "inspection"})
```

A Python tool client can run the first example for a model, or a script can call it directly. With the [MHS MCP bridge](https://github.com/modelhardwarestandard/mcp-server), expose only approved operations, not raw joint writes. See the [SDK fleet examples](https://github.com/modelhardwarestandard/python-sdk/tree/main/examples/fleet) for mounting and subscriptions.

### Local: read or write a slot (BUFFER)

```python
from mhs.format_layer.formats import SafetyFloat
from mhs.mhs_api import MhsDict

# In-memory demo. No hardware is attached.
d = MhsDict("demo", storage="inmem")
d["speed"] = SafetyFloat(0.0, range=(0.0, 100.0), safe=(0.0, 20.0))
d["speed"] = 10.0
print(d["speed"])
```

An attached driver can expose `@sensor` for reads and `@control` for writes. A write to `speed` can call the driver's `set_speed()`. Use shared memory to share data such as camera frames between local processes.

## Add local policy execution only when needed

These are proposed integrations. All three setups work with a Python client or with MCP.

| Need                                                                   | Use                            |
| ---------------------------------------------------------------------- | ------------------------------ |
| Read a camera or ask a device to perform a discrete action             | MHS + plugins                  |
| Run a learned policy that continuously turns observations into actions | MHS + `RobotRuntime` + plugins |
| Run a custom pipeline with different rates or freshness rules          | MHS + RobotGraph + plugins     |

### Standard policy: `RobotRuntime`, no graph

Reuse the configured robot plugin, cameras, and policy. The MHS driver wraps the session, not the hardware loop:

```python
class RuntimeDriver(MhsDriver):
    device_type = "physicalai_runtime"

    @rpc()
    def act(self, action_name: str, args: dict) -> dict:
        return session.submit(action_name, args)  # Proposed: validate and enqueue.
```

```python
# Python client: no MCP.
request = await fleet.invoke("runtime-01", "act", action_name="run_pick", args={})

# Model client: same registered action via MCP.
request = await mcp.call_tool("act", {
    "session_id": "sim", "name": "run_pick", "args": {},
})
```

Both paths should support [`observe`, `act`, `status`, and `stop`](runtime-mcp.md). `observe` reads cached frames and state. An accepted `act` is not a completed task. Reuse the [MHS MCP bridge](https://github.com/modelhardwarestandard/mcp-server) if it can expose only these approved operations; otherwise use a thin Runtime MCP server.

### Custom pipeline: RobotGraph

```text
MHS act or local VLM → Goal node ──┐
                                  ├→ Policy → SendActionNode → Robot plugin
Camera + robot state ─────────────┘
```

```python
# Sketch: configured graph nodes, not built-in MHS adapters.
graph.connect(goal.output("goal"), policy.input("goal"))
graph.connect(camera.output("image"), policy.input("wrist"))
graph.connect(state.output("state"), policy.input("state"))
graph.connect(policy.output("action"), commands.input("action"))
```

The goal node reads a local inbox. The command node checks action validity and calls the same robot plugin's `send_action()`; it is not another driver. Missing or expired actions need a local fallback. If the graph owns the loop, don't run a separate `RobotRuntime` loop. A [local VLM](examples/vlm-node.md) can supply goals with neither MHS nor MCP.

**One hardware owner:** If MHS hosts the plugin, its adapter owns the connection. If Runtime or the graph owns it, MHS reads cached state and submits requests. Don't open the device twice or use fleet RPC per servo tick.
