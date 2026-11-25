# Timing: ticks, deadlines, and freshness

| API | Controls |
| --- | --- |
| `RobotRuntime(..., fps=30)` | Control-loop target for built-in recipes; inference runs on demand |
| `RobotRuntime.from_graph(graph)` | No `fps` argument; periodic nodes declare their own rates |
| `S1PolicyNode(..., rate=10)` | Target for this node's periodic steps, not the robot's command rate |

No graph-wide `fps`. A 10 Hz policy and 30 Hz command node are not a conflict.
Keep `RTCExecution(fps=...)` matched to the Runtime cycle rate: those two
settings refer to the same cadence.

Threads affect placement, not rate. Built-in async/RTC workers infer when the
action queue needs refilling, not in a busy loop. The executor owns periodic
ticks; ROS 2 receipt updates the available input.

## Measure and report

| Metric | Calculation |
| --- | --- |
| Start delay | `actual_start - scheduled_start` |
| Step duration | `actual_finish - actual_start` |
| Deadline miss | `actual_finish > completion_deadline`, if declared |
| Input age | `read_time - capture_or_publication_time`, per contract |

Use a monotonic clock for worker durations. Count skips separately. Sync
overruns can delay later steps: measure and warn, don't preempt. No hard
real-time guarantee, including with separate threads.

## Enforce at the point of use

- At each tick, read the latest available observation on each input. Don't wait
  for a synchronized batch.
- `Latest(max_age=...)`: optional per-edge age limit; skip a step with unusable
  required data.
- Intent and observation age are separate; an S2 intent can span multiple S1 ticks.
- Command consumers still check previously published actions for validity.
- Preserve capture time. Unrelated clock domains mean unknown age, not zero.
