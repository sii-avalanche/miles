import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535


def _configured_port_range() -> tuple[int, int] | None:
    value = os.environ.get("MILES_DYNAMIC_PORT_RANGE")
    if value is None:
        return None
    if re.fullmatch(r"[0-9]+-[0-9]+", value):
        lower, upper = map(int, value.split("-"))
        if 1 <= lower <= upper <= _MAX_PORT:
            return lower, upper
    raise ValueError("MILES_DYNAMIC_PORT_RANGE must be MIN-MAX with 1 <= MIN <= MAX <= 65535")


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _port_range: tuple[int, int] | None = field(default_factory=_configured_port_range, init=False)

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        lower, upper = self._port_range or (_DYNAMIC_PORT_START, _MAX_PORT)
        if type(consecutive) is not int or not 1 <= consecutive <= upper - lower + 1:
            raise ValueError("consecutive must be a positive integer fitting the dynamic port range")
        async with self._lock:
            start_port = self._next_port_of_ip.get(node_ip, lower)
            if not lower <= start_port <= upper - consecutive + 1:
                start_port = lower
            # Explicit bounds travel with the RPC; actor environments may differ.
            bounds = {} if self._port_range is None else {"min_port": lower, "max_port": upper}
            port: int = await actor._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
                **bounds,
            )
            if type(port) is not int or not lower <= port <= upper - consecutive + 1:
                raise RuntimeError("Remote port probe returned a block outside the dynamic port range")
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
