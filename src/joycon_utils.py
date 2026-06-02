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
        """
        example return value:
        {
            'L': {
                'battery': {'charging': 0, 'level': 4},
                'buttons': {
                    'right': {'y': 0, 'x': 0, 'b': 0, 'a': 0, 'sr': 0, 'sl': 0, 'r': 0, 'zr': 0},
                    'shared': {'minus': 0, 'plus': 0, 'r-stick': 0, 'l-stick': 0, 'home': 0, 'capture': 0, 'charging-grip': 0},
                    'left': {'down': 0, 'up': 0, 'right': 0, 'left': 0, 'sr': 0, 'sl': 0, 'l': 0, 'zl': 0}
                },
                'analog-sticks': {'left': {'horizontal': 2003, 'vertical': 2337}, 'right': {'horizontal': 0, 'vertical': 0}},
                'accel': {'x': -105, 'y': -32, 'z': 3877},
                'gyro': {'x': -1.0015141093773683, 'y': -1.1970182296621088, 'z': -0.8760858835208521}
            },
            'R': {
                'battery': {'charging': 0, 'level': 4},
                'buttons': {
                    'right': {'y': 0, 'x': 0, 'b': 0, 'a': 0, 'sr': 0, 'sl': 0, 'r': 0, 'zr': 0},
                    'shared': {'minus': 0, 'plus': 0, 'r-stick': 0, 'l-stick': 0, 'home': 0, 'capture': 0, 'charging-grip': 0},
                    'left': {'down': 0, 'up': 0, 'right': 0, 'left': 0, 'sr': 0, 'sl': 0, 'l': 0, 'zl': 0}
                },
                'analog-sticks': {'left': {'horizontal': 0, 'vertical': 0}, 'right': {'horizontal': 2101, 'vertical': 1820}},
                'accel': {'x': 339, 'y': -149, 'z': -4133},
                'gyro': {'x': 0.239898681640625, 'y': -2.8484840393066406, 'z': -0.27777862548828125}
            }
        }
        """
        return {
            "L": self.joycon_L.get_status(),
            "R": self.joycon_R.get_status(),
        }

    # 右摇杆归一化值（死区过滤后）
    # 返回 (x, y)，范围约 -1 到 1
    def get_R_analog(self) -> tuple[float, float]:
        status = self.get_status()
        R = status["R"]["analog-sticks"]["right"]
        h = R["horizontal"]
        v = R["vertical"]
        # 中性值约 2048，死区 ±100
        DEADZONE = 100
        if abs(h - 2048) < DEADZONE:
            h = 2048
        if abs(v - 2048) < DEADZONE:
            v = 2048
        x = (h - 2048) / 2048.0  # -1 到 1
        y = (v - 2048) / 2048.0  # -1 到 1（Y 轴反向）
        return x, y

    # R/ZR 肩键状态：ZR=1, R=1 都按=0, 否则=0
    # 返回 1=上升, -1=下降, 0=静止
    def get_R_shoulder(self) -> int:
        status = self.get_status()
        R = status["R"]["buttons"]["right"]
        zr = R["zr"]
        r = R["r"]
        if zr and not r:
            return 1  # ZR pressed = 上升
        if r and not zr:
            return -1  # R pressed = 下降
        return 0

    # R A 键单次按下检测（边缘触发）
    # 首次调用返回 False，之后按一次返回 True，然后等松开再按才返回 True
    def get_A_pressed(self) -> bool:
        status = self.get_status()
        current = status["R"]["buttons"]["right"]["a"]
        pressed = current and not getattr(self, "_A_last", 0)
        self._A_last = current
        return pressed

    # 获取夹爪命令（A 键切换）
    def get_gripper_toggle(self) -> bool:
        return self.get_A_pressed()


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
