# Graph primitives

Implemented in `physicalai.graph`. See [Core](core.md) for compilation and stepping.

```python
class Graph:
    def add(self, node: Node) -> Node: ...
    def connect(self, source: Port, destination: Port) -> Edge: ...
    def add_edge(self, edge: Edge) -> Edge: ...
    def compile(self) -> ExecutionPlan: ...

class Node:
    def __init__(self, name: str, *, rate: float | None = None) -> None: ...
    def input(self, name: str) -> Port: ...
    def output(self, name: str) -> Port: ...
    def start(self, ctx: NodeContext) -> None: ...
    def ready(self, ctx: NodeContext) -> bool: ...
    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]]) -> Mapping[str, object]: ...
    def stop(self, ctx: NodeContext) -> None: ...
```

`Port`: read-only `(node, name, direction, data_type, required)`. `input()` / `output()`
look up declarations and raise `KeyError` for unknown names; they don't create ports or read data.

### Node lifecycle

| Hook                | Contract                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `start(ctx)`        | Once per session. Default: no-op. On failure, clean up partial setup and raise.                                  |
| `ready(ctx)`        | Non-blocking, default `True`. `False` delays only this node; no second `start()`.                                |
| `step(ctx, inputs)` | One executor-owned [tick](../timing/overview.md). Inputs and outputs keyed by port name; readiness ≠ fresh data. |
| `stop(ctx)`         | Release resources after a successful `start()`.                                                                  |

### Declare ports on a custom node

```python
class WristPolicy(Node):
    inputs = {"wrist": Image, "q": JointState}
    outputs = {"action": Action}
    optional_inputs = frozenset({"q"})  # Only if this policy can run without state.

policy = WristPolicy("policy", rate=30)
```

- Compile requires a finite, positive `rate` for every scheduled node. `rate=None` has no implicit cadence.
- Payload types are ordinary Python classes. The producer's type must be a subclass of the consumer's type; `object` deliberately accepts any local payload. Richer contracts and process codecs are proposed.
- Inputs are `Message` objects. Outputs are payloads or `Captured(payload, timestamp)` if known. `ctx.now()` provides the executor's clock.

## Wire a graph

```python
graph = Graph()
camera = graph.add(camera_node)
state = graph.add(state_node)
policy = graph.add(policy_node)
commands = graph.add(command_node)

graph.connect(camera.output("image"), policy.input("wrist"))
graph.connect(state.output("state"), policy.input("q"))
graph.connect(policy.output("action"), commands.input("action"))
```

- Register both nodes first. Port names can differ; types must be compatible.
- Outputs can fan out. Required inputs need one producer; optional inputs zero or one.
- Use a selector or merge node for multiple producers.

## Build an edge explicitly

```python
class Edge:
    def __init__(
        self,
        source: Port,
        destination: Port,
        policy: Latest | None = None,
    ) -> None: ...
```

Read-only fields. `connect(src, dst)` is shorthand for
`add_edge(Edge(src, dst))`. To customize delivery, replace the relevant
`connect()` call:

```python
wrist_edge = Edge(
    source=camera.output("image"),
    destination=policy.input("wrist"),
    policy=Latest(max_age=0.10),  # Seconds.
)
graph.add_edge(wrist_edge)
```

`Latest` is the implemented delivery policy; omitting it keeps the newest message without an age limit.
Delivery policy controls what the consumer gets; [transport](../transport/primitives.md) controls how it gets there.
The standalone executor has no transport or buffer arguments on `Edge`.
