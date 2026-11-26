# Two RobotGraphs over Zenoh

One Python file runs both roles. The control process starts a second Python process for inference. Each process compiles and executes its own graph.

```mermaid
flowchart LR
    subgraph Control["Process 1 · Control graph · 30 Hz target"]
        camera[Fake camera]
        state[Fake robot state]
        policy[Async policy + action queue]
        send[Send one action to fake robot]
        client[Zenoh client worker]
        camera -->|frame| policy
        state -->|position| policy
        policy -->|action| send
        policy -->|refill when fewer than 4 actions remain| client
        client -->|completed chunk| policy
    end

    subgraph Inference["Process 2 · Inference graph"]
        server[Zenoh request inbox]
        model[Inference node · about 100 ms]
        server -->|next request| model
    end

    client ==>|observation over private Unix socket| server
    model ==>|eight-action chunk over Zenoh| client
```

The model runs only in the inference process. The control graph keeps reading while waiting for the first chunk, then sends one queued action per control tick. Neither process needs a Zenoh router.

Run [the demo](zenoh_two_process_demo.py) using the command in its docstring, then open `robotgraph-two-process-trace.json` in [Perfetto](https://ui.perfetto.dev) to see both processes on the same timeline.
