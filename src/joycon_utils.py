import time

import pyjoycon
from rich import print
from rich.live import Live
from rich.pretty import Pretty


class JoyCon:
    def __init__(self, calibration_seconds=2.0):
        l_id = pyjoycon.get_L_id()
        r_id = pyjoycon.get_R_id()
        if None in l_id:
            raise RuntimeError(f"Could not find Joy-Con (L). ID: {l_id}")
        if None in r_id:
            raise RuntimeError(f"Could not find Joy-Con (R). ID: {r_id}")

        self.joycon_L = pyjoycon.GyroTrackingJoyCon(*l_id)
        self.joycon_R = pyjoycon.GyroTrackingJoyCon(*r_id)
        self.calibrate(seconds=calibration_seconds)

    def calibrate(self, seconds=2.0, timeout=5.0):
        print(
            f"[cyan]Calibrating Joy-Con (L/R) for {seconds:.1f}s. Keep both controllers still..."
        )
        self.joycon_L.calibrate(seconds=seconds)
        self.joycon_R.calibrate(seconds=seconds)

        deadline = time.monotonic() + max(timeout, seconds + 1.0)
        while time.monotonic() < deadline:
            if not self.joycon_L.is_calibrating and not self.joycon_R.is_calibrating:
                print("[green]Calibration complete.")
                return
            time.sleep(0.05)

        print("[yellow]Calibration timeout. Continue with current calibration values.")

    def get_status(self):
        return {
            "L": self.joycon_L.get_status(),
            "R": self.joycon_R.get_status(),
        }


def main():
    joycon = JoyCon()

    with Live(refresh_per_second=10) as live_display:
        while True:
            try:
                joycon.get_status()
                live_display.update(Pretty(joycon.get_status()), refresh=True)
                time.sleep(0.1)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
