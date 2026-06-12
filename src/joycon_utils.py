import time

import pyjoycon
from rich import print
from rich.live import Live
from rich.pretty import Pretty


# Combining multiple inheritance to create a JoyCon class that supports both gyro tracking and button events
class MyJoyCon(pyjoycon.GyroTrackingJoyCon, pyjoycon.ButtonEventJoyCon): ...


class JoyCon:
    def __init__(self, calibration_seconds=2.0):
        l_id = pyjoycon.get_L_id()
        r_id = pyjoycon.get_R_id()
        assert None not in l_id, "Left Joy-Con not found"
        assert None not in r_id, "Right Joy-Con not found"

        self.joycon_L = self._create_with_retry(l_id, side="L")
        self.joycon_R = self._create_with_retry(r_id, side="R")
        self.calibrate(seconds=calibration_seconds)

    @staticmethod
    def _create_with_retry(
        device_id: tuple, side: str, retries: int = 5, delay: float = 0.1
    ):
        """Create a Joy-Con instance with retry for intermittent pyjoycon init race."""
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                return MyJoyCon(*device_id)
            except AssertionError as e:
                # pyjoycon occasionally raises this during SPI subcmd handshake.
                if "THREAD carefully" not in str(e):
                    raise
                last_err = e
            except OSError as e:
                # Temporary HID I/O failures can happen right after reconnect.
                last_err = e

            if attempt < retries:
                time.sleep(delay)

        raise RuntimeError(
            f"Failed to initialize Joy-Con ({side}) after {retries} retries"
        ) from last_err

    def calibrate(self, seconds=2.0, timeout=5.0):
        print(f"[blue]Calibrating Joy-Con (L&R). Keep still...[/blue]")
        self.joycon_L.calibrate(seconds=seconds)
        self.joycon_R.calibrate(seconds=seconds)
        deadline = time.monotonic() + max(timeout, seconds + 1.0)
        while time.monotonic() < deadline:
            if not self.joycon_L.is_calibrating and not self.joycon_R.is_calibrating:
                print("[green]Calibration complete.[/green]")
                return
            time.sleep(0.05)
        print("[yellow]Calibration timeout. Use current calibration values.[/yellow]")

    def get_status(self):
        """
        Example return value:
        {
            'L': {
                'battery': 4,
                'buttons': {'down': 0, 'up': 0, 'right': 0, 'left': 0, 'sl': 0, 'sr': 0, 'l': 0, 'zl': 0, 'minus': 0, 'l-stick': 0, 'capture': 0},
                'stick': [2003, 2337],  # [H, V]
                'accel': [-105, -32, 3877],
                'gyro': [-1.0015, -1.1970, -0.8760]
            },
            'R': {
                'battery': 4,
                'buttons': {'y': 0, 'x': 0, 'b': 0, 'a': 0, 'sr': 0, 'sl': 0, 'r': 0, 'zr': 0, 'plus': 0, 'r-stick': 0, 'home': 0},
                'stick': [2101, 1820],  # [H, V]
                'accel': [339, -149, -4133],
                'gyro': [0.2398, -2.8484, -0.2777]
            }
        }
        """
        raw_L = self.joycon_L.get_status()
        raw_R = self.joycon_R.get_status()

        # Clean up left hand controller status
        btn_L = {}
        # Merge left hand specific directional buttons, shoulder buttons, and shared buttons
        btn_L.update(raw_L["buttons"]["left"])
        btn_L.update(
            {
                k: v
                for k, v in raw_L["buttons"]["shared"].items()
                if k in ["minus", "l-stick", "capture"]
            }
        )

        clean_L = {
            "battery": raw_L["battery"]["level"],
            "buttons": btn_L,
            "stick": [
                raw_L["analog-sticks"]["left"]["horizontal"],
                raw_L["analog-sticks"]["left"]["vertical"],
            ],
            "accel": [raw_L["accel"]["x"], raw_L["accel"]["y"], raw_L["accel"]["z"]],
            "gyro": [raw_L["gyro"]["x"], raw_L["gyro"]["y"], raw_L["gyro"]["z"]],
        }

        # Clean up right hand controller status
        btn_R = {}
        # Merge right hand specific action buttons, shoulder buttons, and shared buttons
        btn_R.update(raw_R["buttons"]["right"])
        btn_R.update(
            {
                k: v
                for k, v in raw_R["buttons"]["shared"].items()
                if k in ["plus", "r-stick", "home"]
            }
        )

        clean_R = {
            "battery": raw_R["battery"]["level"],
            "buttons": btn_R,
            "stick": [
                raw_R["analog-sticks"]["right"]["horizontal"],
                raw_R["analog-sticks"]["right"]["vertical"],
            ],
            "accel": [raw_R["accel"]["x"], raw_R["accel"]["y"], raw_R["accel"]["z"]],
            "gyro": [raw_R["gyro"]["x"], raw_R["gyro"]["y"], raw_R["gyro"]["z"]],
        }

        return {"L": clean_L, "R": clean_R}

    def get_imu(self):
        """
        Example return value:
        {
            'L': {
                'direction': vec3( 0.998765, 0.00501661, -0.0494945 ),
                'rotation': vec3( -0.013572, 0.0494452, 0.00568786 ),
                'accel': [(-107, 44, -3876), (-107, 42, -3878), (-108, 44, -3879)],
                'gyro': [
                    (-0.8907046731924599, 2.2625047713255784, -0.9855178830520237),
                    (-0.8907046731924599, 0.5186508424048086, -0.9855178830520237),
                    (-0.8907046731924599, 1.3905778068651935, -0.11359091859163895)
                ]
            },
            'R': {
                'direction': vec3( 0.980642, -0.188617, 0.0525824 ),
                'rotation': vec3( 0.008294, -0.0551612, -0.189301 ),
                'accel': [(352, -152, -4135), (350, -148, -4133), (351, -149, -4130)],
                'gyro': [
                    (1.9629631042480469, -2.1580238342285156, -0.08889007568359375),
                    (0.9629631042480469, -2.1580238342285156, -1.0888900756835938),
                    (0.9629631042480469, -2.1580238342285156, -2.0888900756835938)
                ]
            }
        }
        """
        return {
            "L": {
                "direction": self.joycon_L.direction,  # by integrating gyro
                "rotation": self.joycon_L.rotation,  # by integrating gyro
                "accel": self.joycon_L.accel,
                "gyro": self.joycon_L.gyro,
            },
            "R": {
                "direction": self.joycon_R.direction,
                "rotation": self.joycon_R.rotation,
                "accel": self.joycon_R.accel,
                "gyro": self.joycon_R.gyro,
            },
        }

    def events(self):
        # Yield events from both controllers, preferring the right hand for shared buttons
        for event in self.joycon_R.events():
            yield event
        for event in self.joycon_L.events():
            if event[0] not in ["plus", "r-stick", "home"]:
                yield event

    def get_L_analog(self, deadzone: int = 300, center: int = 2048) -> tuple:
        """
        Left stick horizontal/vertical values, normalized to -1 to 1 with a deadzone ±deadzone around the center (center).
        """
        status = self.get_status()
        h, v = status["L"]["stick"]  # [Horizontal, Vertical]
        # Apply deadzone
        h = center if abs(h - center) < deadzone else h
        v = center if abs(v - center) < deadzone else v
        # Normalize to -1 to 1
        x = (h - center) / center
        y = (v - center) / center
        return x, y

    def get_R_analog(self, deadzone: int = 300, center: int = 2048) -> tuple:
        """
        Right stick horizontal/vertical values, normalized to -1 to 1 with a deadzone ±deadzone around the center (center).
        """
        status = self.get_status()
        h, v = status["R"]["stick"]  # [Horizontal, Vertical]
        # Apply deadzone
        h = center if abs(h - center) < deadzone else h
        v = center if abs(v - center) < deadzone else v
        # Normalize to -1 to 1
        x = (h - center) / center
        y = (v - center) / center
        return x, y

    def display_dashboard(self):
        """Display a live 3D dashboard of Joy-Con IMU direction using matplotlib."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        fig.text(0.5, 0.01, "Ctrl+C to exit", ha="center", fontsize=9, color="gray")

        with Live(refresh_per_second=30) as live_display:
            while True:
                try:
                    imu = self.get_imu()
                    status = self.get_status()
                    events = list(self.events())

                    dir_L = imu["L"]["direction"]
                    dir_R = imu["R"]["direction"]

                    ax.clear()

                    # Draw world coordinate axes (gray, subtle)
                    ax.quiver(
                        0,
                        0,
                        0,
                        1.2,
                        0,
                        0,
                        color="gray",
                        alpha=0.4,
                        arrow_length_ratio=0.1,
                    )
                    ax.quiver(
                        0,
                        0,
                        0,
                        0,
                        -1.2,
                        0,
                        color="gray",
                        alpha=0.4,
                        arrow_length_ratio=0.1,
                    )
                    ax.quiver(
                        0,
                        0,
                        0,
                        0,
                        0,
                        1.2,
                        color="gray",
                        alpha=0.4,
                        arrow_length_ratio=0.1,
                    )
                    ax.text(1.3, 0, 0, "X", color="gray", fontsize=9)
                    ax.text(0, -1.3, 0, "Y", color="gray", fontsize=9)
                    ax.text(0, 0, 1.3, "Z", color="gray", fontsize=9)

                    # Left Joy-Con direction (cyan)
                    ax.quiver(
                        0,
                        0,
                        0,
                        dir_L[1],
                        dir_L[0],
                        -dir_L[2],
                        color="blue",
                        arrow_length_ratio=0.15,
                        linewidth=2.5,
                        label="L",
                    )

                    # Right Joy-Con direction (magenta)
                    ax.quiver(
                        0,
                        0,
                        0,
                        dir_R[1],
                        dir_R[0],
                        -dir_R[2],
                        color="red",
                        arrow_length_ratio=0.15,
                        linewidth=2.5,
                        label="R",
                    )

                    ax.set_xlim([-1.5, 1.5])
                    ax.set_ylim([-1.5, 1.5])
                    ax.set_zlim([-1.5, 1.5])
                    ax.set_xlabel("X", fontsize=10)
                    ax.set_ylabel("Y", fontsize=10)
                    ax.set_zlabel("Z", fontsize=10)
                    ax.legend(loc="upper left")

                    # Button status as title
                    btn_L_pressed = [k for k, v in status["L"]["buttons"].items() if v]
                    btn_R_pressed = [k for k, v in status["R"]["buttons"].items() if v]
                    L_btns = ", ".join(btn_L_pressed) if btn_L_pressed else "—"
                    R_btns = ", ".join(btn_R_pressed) if btn_R_pressed else "—"

                    ax.set_title(f"L: {L_btns} | R: {R_btns}", fontsize=9)

                    plt.draw()
                    plt.pause(0.001)

                    live_display.update(
                        Pretty(
                            {
                                "L direction": list(dir_L),
                                "R direction": list(dir_R),
                                "L buttons": btn_L_pressed,
                                "R buttons": btn_R_pressed,
                                "event count": len(events),
                            }
                        ),
                        refresh=True,
                    )
                    time.sleep(0.033)

                except KeyboardInterrupt:
                    plt.close(fig)
                    print("[yellow]Exiting 3D dashboard...[/yellow]")
                    break
                except Exception as e:
                    print(f"[bright_red]Error: {e}[/bright_red]")
                    break


def main():
    joycon = JoyCon()
    button_events_queue = []
    button_events_queue_size = 10  # Display the last n events
    # joycon.display_dashboard()
    # quit()

    with Live(refresh_per_second=10) as live_display:
        while True:
            try:
                imu = joycon.get_imu()
                status = joycon.get_status()
                button_events_queue.extend(joycon.events())
                # Keep only the most recent events in the queue
                button_events_queue = button_events_queue[-button_events_queue_size:]
                L_analog = joycon.get_L_analog()
                R_analog = joycon.get_R_analog()
                live_display.update(
                    Pretty(
                        {
                            "imu": imu,
                            "status": status,
                            "button_events_history": button_events_queue,
                            "L_analog": L_analog,
                            "R_analog": R_analog,
                        }
                    ),
                    refresh=True,
                )

                direction_L = imu["L"]["direction"]
                direction_R = imu["R"]["direction"]

                time.sleep(0.1)

            except KeyboardInterrupt:
                print("[yellow]Exiting...")
                break
            except Exception as e:
                print(f"[bright_red]Error: {e}[/bright_red]")
                break


if __name__ == "__main__":
    main()
