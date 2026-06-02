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
                "direction": self.joycon_L.direction,
                "rotation": self.joycon_L.rotation,
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

    def get_R_imu_delta(self) -> dict:
        """获取按住 ZR 期间的 IMU 相对偏移（相对于基准姿态）。

        首次调用（ZR 刚按下）记录基准姿态，返回全零。
        之后每次返回相对基准姿态的欧拉角变化。
        """
        status = self.get_status()
        R = status["R"]

        # 欧拉角（弧度，ZYX 顺序）
        rotation = R["rotation"]

        if not hasattr(self, "_imu_baseline"):
            self._imu_baseline = {
                "rotation": rotation,
            }
            return {"position": (0.0, 0.0, 0.0), "attitude": (0.0, 0.0, 0.0)}

        # 计算相对偏移
        pos_delta = (0.0, 0.0, 0.0)  # 加速度积分暂用零
        att_delta = (
            rotation.x - self._imu_baseline["rotation"].x,
            rotation.y - self._imu_baseline["rotation"].y,
            rotation.z - self._imu_baseline["rotation"].z,
        )

        return {"position": pos_delta, "attitude": att_delta}

    def is_ZR_pressed(self) -> bool:
        """检查 ZR 肩键是否按下。"""
        status = self.get_status()
        return bool(status["R"]["buttons"]["right"]["zr"])

    def reset_imu_baseline(self) -> None:
        """重置 IMU 基准姿态（松开 ZR 时调用）。"""
        if hasattr(self, "_imu_baseline"):
            delattr(self, "_imu_baseline")


def main():
    joycon = JoyCon()

    with Live(refresh_per_second=10) as live_display:
        while True:
            try:
                live_display.update(Pretty(joycon.get_imu()), refresh=True)
                time.sleep(0.1)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
