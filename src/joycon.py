import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

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
        self.on_imu: (
            Callable[
                [Side, tuple[float, float, float], tuple[float, float, float]], None
            ]
            | None
        ) = None

    def scan(self, timeout: float = 5.0) -> int:
        """Scan for Joycon devices. Returns number of devices found."""
        # pyjoycon imports gyro helpers from glm at package import time.
        # Keep this import local so module import stays resilient.
        print("Scanning for Joycons...")
        try:
            from pyjoycon import PythonicJoyCon, get_L_id, get_R_id
        except Exception as e:
            logger.error(f"Failed to import pyjoycon: {e}")
            return 0

        print(f"{get_L_id()=}, {get_R_id()=}")
        with self._lock:
            count = 0
            candidates = [
                (Side.LEFT, get_L_id()),
                (Side.RIGHT, get_R_id()),
            ]

            for side, (vendor_id, product_id, serial) in candidates:
                if vendor_id is None or product_id is None:
                    continue
                mac = serial or f"{vendor_id:04x}:{product_id:04x}:{side.value}"
                if mac not in self._devices:
                    try:
                        instance = PythonicJoyCon(vendor_id, product_id, serial)
                    except Exception as e:
                        logger.warning(f"Failed to initialize Joycon {mac}: {e}")
                        continue

                    device = JoyconDevice(mac, instance)
                    device.side = side
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
        return Side.LEFT  # default

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
            current_state = self._read_state(device)
        except Exception as e:
            logger.warning(f"Failed to poll {device.mac}: {e}")
            return

        if device._last_state is not None:
            self._dispatch_events(device, device._last_state, current_state)

        device._last_state = current_state

    def _read_state(self, device: JoyconDevice) -> JoyconState:
        """Read current state from a PythonicJoyCon instance."""
        instance = device._instance
        side = device.side or Side.LEFT

        # joycon-python returns stick values around [0, 4095]; normalize to [-1, 1].
        def _normalize_stick(raw_x: float, raw_y: float) -> tuple[float, float]:
            x = max(-1.0, min(1.0, (float(raw_x) - 2048.0) / 2048.0))
            y = max(-1.0, min(1.0, (float(raw_y) - 2048.0) / 2048.0))
            return (x, y)

        if side == Side.LEFT:
            stick_x, stick_y = _normalize_stick(*instance.stick_l)
            buttons = {
                "l": bool(instance.l),
                "zl": bool(instance.zl),
                "minus": bool(instance.minus),
                "up": bool(instance.up),
                "down": bool(instance.down),
                "left": bool(instance.left),
                "right": bool(instance.right),
                "stick_l_btn": bool(instance.stick_l_btn),
                "capture": bool(instance.capture),
                "left_sr": bool(instance.left_sr),
                "left_sl": bool(instance.left_sl),
            }
            sticks = {"left": (stick_x, stick_y)}
        else:
            stick_x, stick_y = _normalize_stick(*instance.stick_r)
            buttons = {
                "a": bool(instance.a),
                "b": bool(instance.b),
                "x": bool(instance.x),
                "y": bool(instance.y),
                "r": bool(instance.r),
                "zr": bool(instance.zr),
                "plus": bool(instance.plus),
                "stick_r_btn": bool(instance.stick_r_btn),
                "home": bool(instance.home),
                "right_sr": bool(instance.right_sr),
                "right_sl": bool(instance.right_sl),
            }
            sticks = {"right": (stick_x, stick_y)}

        # Use latest IMU samples for event stream.
        accel = tuple(float(v) for v in instance.accel_in_g[-1])
        gyro = tuple(float(v) for v in instance.gyro_in_deg[-1])
        return JoyconState(
            buttons=buttons,
            sticks=sticks,
            accel=accel,
            gyro=gyro,
        )

    def _dispatch_events(
        self, device: JoyconDevice, old: JoyconState, new: JoyconState
    ) -> None:
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
