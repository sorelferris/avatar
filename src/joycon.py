from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
import threading
import logging

logger = logging.getLogger(__name__)


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class JoyconState:
    """Current state of a Joycon controller."""
    buttons: dict[str, bool]
    sticks: dict[str, tuple[float, float]]
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


class JoyconDevice:
    """Wrapper for a single Joycon device."""

    def __init__(self, mac: str, pyjoycon_instance: Any):
        self.mac = mac
        self._instance = pyjoycon_instance
        self.side: Side | None = None
        self._last_state: JoyconState | None = None


class JoyconManager:
    """Manages all Joycon devices and dispatches events via callbacks."""

    def __init__(self, poll_interval: float = 0.01):
        self._poll_interval = poll_interval
        self._devices: dict[str, JoyconDevice] = {}
        self._running = False
        self._thread: threading.Thread | None = None

        # Callbacks
        self.on_button: Callable[[Side, str, bool], None] | None = None
        self.on_stick: Callable[[Side, float, float], None] | None = None
        self.on_imu: Callable[[Side, tuple, tuple], None] | None = None

    def scan(self, timeout: float = 5.0) -> int:
        """Scan for Joycon devices. Returns number of devices found."""
        ...

    def start(self) -> None:
        """Start the event polling thread."""
        ...

    def stop(self) -> None:
        """Stop the event polling thread."""
        ...

    def get_state(self, mac: str) -> JoyconState | None:
        """Get current state of a device by MAC address."""
        ...

    def get_left_mac(self) -> str | None:
        """Get MAC address of left Joycon."""
        ...

    def get_right_mac(self) -> str | None:
        """Get MAC address of right Joycon."""
        ...