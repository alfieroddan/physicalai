# Authoring skills

Skills teach the model to write ordinary Python, config, and tests. Reuse
existing components first; generate adapters only when needed.

| Need | Reference for the skill |
| --- | --- |
| Configure an existing camera or Runtime | [`pai-configure-runtime`](../skills/using/pai-configure-runtime/SKILL.md) |
| Write a missing camera adapter | [Camera API](../docs/reference/camera-api.md) |
| Write a missing robot plugin | [Robot API](../docs/reference/robot-api.md); [plugin skill](../skills/using/pai-create-robot-plugin/SKILL.md) for Studio registration |
| Build a custom graph | [Graph primitives](../robotgraph-docs/graph/primitives.md) and [VLM node example](examples/vlm-node.md) for a proposed graph-authoring skill |

> Use the configured camera and simulated robot. Write a graph that runs the
> VLM from a local model artifact on a separate GPU worker. Route its goals to
> the pick policy. Show the wiring and checks before launching.

Review before launch. Authoring skills do not become live MCP tools or grant
the running VLM permission to install plugins or rewire the graph.
