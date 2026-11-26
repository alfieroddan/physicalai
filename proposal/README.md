# RobotGraph proposal

The standalone `physicalai.graph` compiler and one-thread executor are implemented. The demos use fake devices and show what the design can do without changing `RobotRuntime` or introducing a transport requirement for local graphs.

| Start here                                     | What it covers                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [Runnable demos](demos/README.md)              | Wiring diagrams and real Perfetto snapshots, including action-chunk refill and S1/S2 across processes |
| [Integration backlog](integration.md)          | Delivered standalone work and proposed plugin, Runtime, transport, and agent integration              |
| [Graph design](../robotgraph-docs/overview.md) | The current API and the proposed Runtime surface                                                      |
| [Agentic design](../agent-docs/overview.md)    | Proposed authoring, local VLM, and MCP workflows                                                      |

## Presentations

- [RobotGraph core](robotgraph-core.pdf): implemented standalone executor, demo evidence, and proposed integration.
- [Agentic Runtime](agentic-runtime.pdf): proposed workflows; not part of the current demo implementation.
- [Short-term proposal](short-term.pdf): original direction and earlier prototype API, retained for context. Use the core deck and design for the current API.
