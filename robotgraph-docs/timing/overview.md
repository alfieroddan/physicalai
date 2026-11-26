# Timing: ticks, deadlines, and freshness

| API                                       | Controls                                                           |
| ----------------------------------------- | ------------------------------------------------------------------ |
| `RobotRuntime(..., fps=30)`               | Control-loop target for built-in recipes; inference runs on demand |
| `GraphExecutor(compile_graph(graph))`     | No `fps` argument; each node declares its own rate                 |
| Proposed `RobotRuntime.from_graph(graph)` | Retains the graph's node rates                                     |

No graph-wide `fps`. A 10 Hz policy and 30 Hz command node are not a conflict.
Keep `RTCExecution(fps=...)` matched to the Runtime cycle rate: those two
settings refer to the same cadence.

Period and placement are separate. The standalone executor runs one schedule on the calling thread.
The demo workers compute on request; the 30 Hz control target does not make the model run at 30 Hz.

## Measure and report

| Metric        | Calculation                                                                        |
| ------------- | ---------------------------------------------------------------------------------- |
| Start delay   | `actual_start - scheduled_start`                                                   |
| Tick duration | `actual_finish - actual_start`, including readiness, input reads, and publication  |
| Deadline miss | `actual_finish > scheduled_start + period`                                         |
| Missed ticks  | Scheduled ticks the executor advances past without replaying                       |
| Input age     | `read_time - captured_at`, or `read_time - published_at` if capture time is absent |

Use a monotonic clock for worker durations. Count skips separately. Sync
overruns can delay later steps: measure and warn, don't preempt. No hard
real-time guarantee, including with separate threads.

## Enforce at the point of use

- At each tick, read the latest available observation on each input. Don't wait
  for a synchronized batch.
- `Latest(max_age=...)`: optional per-edge age limit in seconds. Missing or stale required data skips the step; unavailable optional data is omitted.
- Intent and observation age are separate; an S2 intent can span multiple S1 ticks.
- Command consumers still check previously published actions for validity.
- Preserve capture time. A different clock domain or a future timestamp means unknown age, not zero. Reject unknown age when an edge has an age limit.
