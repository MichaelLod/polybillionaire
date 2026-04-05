"""Message bus for inter-agent communication.

``Bus`` is the original append-only log with synchronous listeners.
``EventBus`` extends it with thread-safe topic queues for the swarm.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Message:
    sender: str
    to: str
    kind: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    seq: int = 0


class Bus:
    def __init__(self) -> None:
        self._log: list[Message] = []
        self._seq = 0
        self._listeners: list[Callable[[Message], None]] = []

    def post(
        self,
        sender: str,
        to: str,
        kind: str,
        content: str,
        data: dict[str, Any] | None = None,
    ) -> Message:
        self._seq += 1
        msg = Message(
            sender=sender,
            to=to,
            kind=kind,
            content=content,
            data=data or {},
            seq=self._seq,
        )
        self._log.append(msg)
        for fn in self._listeners:
            fn(msg)
        return msg

    def on_message(self, fn: Callable[[Message], None]) -> None:
        self._listeners.append(fn)

    @property
    def log(self) -> list[Message]:
        return list(self._log)


class EventBus(Bus):
    """Thread-safe bus with topic-based queues for the agent swarm.

    Usage::

        bus = EventBus()
        q = bus.subscribe("finding")        # Reasoning subscribes
        bus.emit("finding", "Research-1", {"lead_id": 42, ...})  # Research posts
        msg = q.get(timeout=10)             # Reasoning receives
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._topic_queues: dict[str, list[queue.Queue[Message]]] = {}

    def post(
        self,
        sender: str,
        to: str,
        kind: str,
        content: str,
        data: dict[str, Any] | None = None,
    ) -> Message:
        with self._lock:
            return super().post(sender, to, kind, content, data)

    def subscribe(self, topic: str) -> queue.Queue[Message]:
        """Get a queue that receives all messages emitted to *topic*."""
        q: queue.Queue[Message] = queue.Queue()
        with self._lock:
            self._topic_queues.setdefault(topic, []).append(q)
        return q

    def unsubscribe(self, topic: str, q: queue.Queue[Message]) -> None:
        """Remove a queue from a topic's subscriber list."""
        with self._lock:
            subs = self._topic_queues.get(topic, [])
            if q in subs:
                subs.remove(q)

    def emit(
        self,
        topic: str,
        sender: str,
        data: dict[str, Any],
        content: str = "",
    ) -> Message:
        """Post to the log AND push to all subscribers of *topic*."""
        msg = self.post(sender, "all", topic, content or str(data)[:200], data)
        with self._lock:
            subscribers = list(self._topic_queues.get(topic, []))
        for q in subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # drop if consumer is overwhelmed
        return msg
