# Transport primitives

## Requirements

- Integrate with ROS 2, Zenoh, and DDS; don't replace them.
- Built-in thread and same-host process delivery; no external dependency.
- Configure external endpoints on nodes, not on every graph edge.

## Message envelope

One `Message[T]`; payload types vary. Example, not a wire format:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Timestamp:
    ns: int
    clock_domain: str


@dataclass(frozen=True)
class Message(Generic[T]):
    source: tuple[str, str]  # Node name and output port name.
    sequence: int          # Increments within this source's run.
    captured_at: Timestamp | None
    published_at: Timestamp
    payload: T


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

| Payload | Possible representations | Delivery considerations |
| --- | --- | --- |
| Image frame | Raw array, shared-memory buffer, encoded video | Latest fresh frame; lossy only if allowed |
| Joint state | Local object or typed message | Latest for control; ordered samples for some recorders |
| Action / chunk | Local object or typed timed command | Preserve ordering and expiry; no silent drops |

### Mix protocols in the same graph

```text
camera.image ─shared memory→ policy.wrist   (same host)
camera.image → video_publisher ─WebRTC→ viewer
speech.text  → text_publisher ─ROS 2 or Zenoh→ subscriber
```

Unknown capture age may be fine for preview, not for a capture-age limit.
WebRTC is unrelated to RobotRuntime's `RTCExecution` (Real-Time Chunking).

### Shared-memory ring buffer

Latest-frame edge, same host, fixed-size slots:

1. Write pixels, then publish `(slot_index, sequence, metadata)` via IPC.
2. Lease the matching `(slot_index, sequence)` before reading the newest
   complete frame. No overwrite until the reader releases the slot.
3. If all slots are leased, drop the incoming frame and count it. Ordered
   action queues need a separate overflow policy.

## TransportProtocol

Adapter interface (signatures open): deliver `Message[T]` without changing
`DeliveryPolicy`. Runtime selects in-process delivery or same-host IPC
automatically. Shared memory is optional for compatible payloads.

## Connect to an existing ROS 2 node

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

## Connect graph workers

[Local graph per host](../examples/s1s2/index.md#3-separate-computers-start-a-local-graph-on-each-host).
Managed cross-host workers are separate: one transport per worker link;
deployment API open.
