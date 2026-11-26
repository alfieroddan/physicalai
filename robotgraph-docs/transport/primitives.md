# Transport primitives

## What runs today

An executor uses one in-memory `Latest` slot per edge on its own thread. Edges do not cross threads or processes directly. The [demos](../../proposal/demos/README.md) keep transport outside the scheduler:

| Demo         | Boundary                                  | What crosses it                                                                       |
| ------------ | ----------------------------------------- | ------------------------------------------------------------------------------------- |
| Async thread | Same process                              | Observation arguments and a future for the result                                     |
| Shared rings | Same process, two threads                 | Immutable Python tuples through bounded request and reply rings                       |
| Zenoh        | Private Unix socket, one or two processes | Fixed-size observation and action-chunk messages, not the graph `Message[T]` envelope |

In the [two-process demo](../../proposal/demos/two_process/README.md), the launcher waits for the inference graph to register its Zenoh service before starting the control graph. Each process owns its own plan. The control graph keeps the queue and refill rule; the inference graph handles one request at a time. No router or remote launcher is required.

## Proposed transport requirements

- Integrate with ROS 2, Zenoh, and DDS; don't replace them.
- Built-in thread and same-host process delivery; no external dependency.
- Configure external endpoints on nodes, not on every graph edge.

## Message envelope

The local envelope is implemented in `physicalai.graph`. This example is not a wire format:

```python
from dataclasses import dataclass

from physicalai.graph import Message, Timestamp

@dataclass(frozen=True)
class ImageFrame:
    width: int
    height: int
    pixel_format: str
    pixels: bytes


frame = Message(
    source=("wrist", "image"),
    sequence=42,
    captured_at=Timestamp(12_000_000, "demo-clock"),
    published_at=Timestamp(13_000_000, "demo-clock"),
    payload=ImageFrame(2, 2, "gray8", b"\x00\x40\x80\xff"),
)
```

`captured_at`: original observation time, if known. `published_at`: graph
publication time. Receipt time is neither. See [Timing](../timing/overview.md).

### Payloads, encodings, and delivery

**Port** = payload type. **Codec** = representation. **Transport** = delivery.

| Payload        | Possible representations                       | Delivery considerations                                |
| -------------- | ---------------------------------------------- | ------------------------------------------------------ |
| Image frame    | Raw array, shared-memory buffer, encoded video | Latest fresh frame; lossy only if allowed              |
| Joint state    | Local object or typed message                  | Latest for control; ordered samples for some recorders |
| Action / chunk | Local object or typed timed command            | Preserve ordering and expiry; no silent drops          |

### Proposed: mix protocols in the same graph

```text
camera.image ─shared memory→ policy.wrist   (same host)
camera.image → video_publisher ─WebRTC→ viewer
speech.text  → text_publisher ─ROS 2 or Zenoh→ subscriber
```

Unknown capture age may be fine for preview, not for a capture-age limit.
WebRTC is unrelated to RobotRuntime's `RTCExecution` (Real-Time Chunking).

### Proposed: cross-process shared-memory ring buffer

Latest-frame edge, same host, fixed-size slots. This is different from the demo's in-process rings, which pass Python references:

1. Write pixels, then publish `(slot_index, sequence, metadata)` via IPC.
2. Lease the matching `(slot_index, sequence)` before reading the newest
   complete frame. No overwrite until the reader releases the slot.
3. If all slots are leased, drop the incoming frame and count it. Ordered
   action queues need a separate overflow policy.

## Proposed TransportProtocol

Adapter interface (signatures open): deliver `Message[T]` without changing
`DeliveryPolicy`. Runtime selects in-process delivery or same-host IPC
automatically. Shared memory is optional for compatible payloads.

## Proposed ROS 2 endpoint nodes

```python
# Proposed API. The policy is already registered in graph.
camera = graph.add(
    Ros2SubscriptionNode(
        name="wrist_camera",
        rate=30,
        topic="/wrist/image_raw",
        message_type=RosImage,
        qos=camera_qos,
        decode=ros_image_to_frame,
    )
)
graph.connect(camera.output("message"), policy.input("wrist"))
```

`RosImage` = `sensor_msgs.msg.Image`. Use compatible QoS. The remote publisher
runs independently; `Ros2PublisherNode` handles the reverse direction.

## Proposed worker and host connections

For separate computers, start a [local graph per host](../examples/s1s2/index.md#3-separate-computers-start-a-local-graph-on-each-host) and connect explicit endpoints. Remote launch is out of scope.
