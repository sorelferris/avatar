from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
import threading
import logging
import time

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
        self.poll_interval = poll_interval
        self._devices: dict[str, JoyconDevice] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Callbacks
        self.on_button: Callable[[Side, str, bool], None] | None = None
        self.on_stick: Callable[[Side, float, float], None] | None = None
        self.on_imu: Callable[[Side, tuple[float, float, float], tuple[float, float, float]], None] | None = None

    def scan(self, timeout: float = 5.0) -> int:
        """Scan for Joycon devices. Returns number of devices found."""
        with self._lock:
            try:
                from pyjoycon import JoyConManager
            except ImportError:
                logger.warning("pyjoycon not installed, cannot scan")
                return 0

            manager = JoyConManager()
            joycons = manager.get_joycons()
            count = 0

            for mac, instance in joycons.items():
                if mac not in self._devices:
                    device = JoyconDevice(mac, instance)
                    device.side = self._detect_side(instance)
                    self._devices[mac] = device
                    count += 1
                    logger.info(f"Found Joycon: {mac} ({device.side.value})")

            return count

    def _detect_side(self, instance: Any) -> Side:
        """Detect if this is a left or right Joycon."""
        try:
            device_type = instance.get_device_type()
            if "left" in str(device_type).lower():
                return Side.LEFT
            elif "right" in str(device_type).lower():
                return Side.RIGHT
        except Exception:
            pass
        return Side.LEFT # default

    def start(self) -> None:
        """Start the event polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("JoyconManager started")

    def stop(self) -> None:
        """Stop the event polling thread."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("JoyconManager stopped")

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            with self._lock:
                devices = list(self._devices.values())
            for device in devices:
                self._poll_device(device)
            time.sleep(self.poll_interval)

    def _poll_device(self, device: JoyconDevice) -> None:
        """Poll a single device and dispatch events on state change."""
        try:
            status = device._instance.get_status()
        except Exception as e:
            logger.warning(f"Failed to poll {device.mac}: {e}")
            return

        current_state = self._parse_status(status, device.side)

        if device._last_state is not None:
            self._dispatch_events(device, device._last_state, current_state)

        device._last_state = current_state

    def _parse_status(self, status: dict, side: Side | None) -> JoyconState:
        """Parse pyjoycon status dict into JoyconState."""
        buttons = status.get("buttons", {})
        sticks = status.get("sticks", {})
        accel = status.get("accel", (0, 0, 0))
        gyro = status.get("gyro", (0, 0, 0))
        return JoyconState(
            buttons=buttons,
            sticks=sticks,
            accel=tuple(accel),
            gyro=tuple(gyro),
        )

    def _dispatch_events(self, device: JoyconDevice, old: JoyconState, new: JoyconState) -> None:
        """Compare states and dispatch callback events."""
        side = device.side or Side.LEFT

        # Button events
        for key, pressed in new.buttons.items():
            was_pressed = old.buttons.get(key, False)
            if pressed != was_pressed and self.on_button:
                self.on_button(side, key, pressed)

        # Stick events (dispatch when value changes significantly)
        for stick_name, (x, y) in new.sticks.items():
            old_x, old_y = old.sticks.get(stick_name, (0, 0))
            if abs(x - old_x) > 0.01 or abs(y - old_y) > 0.01:
                if self.on_stick:
                    self.on_stick(side, x, y)

        # IMU events
        if self.on_imu:
            self.on_imu(side, new.accel, new.gyro)

    def get_state(self, mac: str) -> JoyconState | None:
        """Get current state of a device by MAC address."""
        with self._lock:
            device = self._devices.get(mac)
            return device._last_state if device else None

    def get_left_mac(self) -> str | None:
        """Get MAC address of left Joycon."""
        with self._lock:
            for device in self._devices.values():
                if device.side == Side.LEFT:
                    return device.mac
            return None

    def get_right_mac(self) -> str | None:
        """Get MAC address of right Joycon."""
        with self._lock:
            for device in self._devices.values():
                if device.side == Side.RIGHT:
                    return device.mac
            return None