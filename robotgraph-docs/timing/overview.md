# Timing: ticks, deadlines, and freshness

The node declares a target rate; the executor calls `step(ctx, inputs)`.
Tick context: number, scheduled time, actual start. No node-owned timer or
loop; ROS 2 receipt doesn't schedule a graph step.

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

- `Latest(max_age=...)`: check age at read time; skip when required data is unusable.
- Intent and observation age are separate; an S2 intent can span multiple S1 ticks.
- Command consumers still check previously published actions for validity.
- Preserve capture time. Unrelated clock domains mean unknown age, not zero.
