"""Message bus for inter-agent communication."""

from __future__ import annotations

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
