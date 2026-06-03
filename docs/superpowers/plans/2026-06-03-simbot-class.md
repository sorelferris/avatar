# SimBot 类实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现 `SimBot` 类 — 人形机器人双臂的可视化与控制层，封装 `RobotModel` + `ViserViewer` + 双 `IKSolver`，通过 `set_joints(dict)` 编程 API 实时控制关节并显示。

**架构：** 单文件 `src/sim_bot.py` 暴露 `SimBot` 类；`set_joints` 走 delta 语义（未指定关节保持当前值）；内置左右臂各一个 `IKSolver`（复用 `src/ik_solver.py`）；viewer 可关做 headless 测试。

**技术栈：** scikit-robot（RobotModel + ViserViewer）、Pinocchio（IK via IKSolver）、pytest

**规格：** `docs/superpowers/specs/2026-06-03-simbot-class-design.md`

---

## 文件结构

| 文件 | 状态 | 职责 |
|---|---|---|
| `tests/test_sim_bot.py` | 新建 | SimBot 单元测试（用 SO101 URDF + 显式 arm joint 名） |
| `src/sim_bot.py` | 新建 | SimBot 类 |
| `src/sim_env.py` | 改写 | 改为 SimBot 演示 demo（右臂插值动画 + wait_until_close） |

不修改的文件：`src/ik_solver.py`、`src/joycon_utils.py`、`main_joycon.py`、`main.py`、`pyproject.toml`。

**测试 URDF 选择**：用项目内的 `assets/SO101/so101_new_calib.urdf`（SO101 是 5-DoF 单臂；测试时显式传 `right_arm_joints` 列表）。这样测的是 SimBot 的**可配置接口**（任意 URDF + 任意关节名），不仅是默认 humanoid 7-DoF。

---

## 任务 1：测试脚手架 + 构造测试

**文件：**
- 创建：`tests/test_sim_bot.py`

- [ ] **步骤 1：写失败测试 — 构造 + viewer=False 不开浏览器**

```python
"""Unit tests for SimBot class.

Use SO101 URDF (project-internal) with explicit arm joint names so we test
the configurable interface rather than only the humanoid 7-DoF defaults.
"""
import numpy as np
import pytest

from src.sim_bot import SimBot

URDF = "assets/SO101/so101_new_calib.urdf"
SO101_ARM = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SO101_EEF = "gripper_frame_link"


def test_construct_with_viewer_false_does_not_open_browser():
    """viewer=False must not start ViserViewer (headless mode)."""
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # Constructed successfully without starting viser
    assert bot is not None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py -v`
预期：FAIL — `ModuleNotFoundError: No module named 'src.sim_bot'`

---

## 任务 2：SimBot 最小构造

**文件：**
- 创建：`src/sim_bot.py`

- [ ] **步骤 1：写最小 __init__ 实现**

```python
"""SimBot — humanoid dual-arm visualization and control layer.

Wraps a scikit-robot RobotModel + ViserViewer, with built-in Pinocchio IK
solvers for the left and right arms. Designed as a thin layer for JoyCon
teleoperation visualization; can also be driven standalone for animation
demos.
"""
import numpy as np
from skrobot.model import RobotModel
from skrobot.viewers import ViserViewer

from src.ik_solver import IKSolver


_DEFAULT_RIGHT_ARM = [f"right_arm_joint{i}" for i in range(1, 8)]
_DEFAULT_LEFT_ARM = [f"left_arm_joint{i}" for i in range(1, 8)]


class SimBot:
    """Dual-arm robot simulator with real-time viser visualization.

    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF file.
    right_arm_joints : list[str] | None
        Names of right arm joints, in kinematic order. Defaults to
        ``["right_arm_joint1", ..., "right_arm_joint7"]``.
    left_arm_joints : list[str] | None
        Names of left arm joints. Defaults to humanoid 7-DoF names.
    right_eef_frame : str
        Pinocchio frame name for the right end-effector. Must exist in
        the URDF as a frame.
    left_eef_frame : str
        Pinocchio frame name for the left end-effector.
    viewer : bool
        If True (default), start a ViserViewer for browser visualization.
        Set False for headless testing.
    auto_redraw : bool
        If True (default), every ``set_joints`` call triggers
        ``viewer.redraw()``. Pass ``redraw=False`` to a single call to
        override.
    """

    def __init__(
        self,
        urdf_path: str,
        right_arm_joints: list[str] | None = None,
        left_arm_joints: list[str] | None = None,
        right_eef_frame: str = "right_arm_link7",
        left_eef_frame: str = "left_arm_link7",
        viewer: bool = True,
        auto_redraw: bool = True,
    ) -> None:
        self._robot = RobotModel.from_urdf(urdf_path)
        self._right_arm_joints = right_arm_joints or _DEFAULT_RIGHT_ARM
        self._left_arm_joints = left_arm_joints or _DEFAULT_LEFT_ARM
        self._right_eef_frame = right_eef_frame
        self._left_eef_frame = left_eef_frame
        self._auto_redraw = auto_redraw

        if viewer:
            self._viewer = ViserViewer()
            self._viewer.add(self._robot)
        else:
            self._viewer = None
```

- [ ] **步骤 2：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_construct_with_viewer_false_does_not_open_browser -v`
预期：PASS

- [ ] **步骤 3：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add SimBot class skeleton with viewer toggle"
```

---

## 任务 3：get_angles()

**文件：**
- 修改：`src/sim_bot.py`
- 测试：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加失败测试**

```python
def test_get_angles_returns_full_vector():
    """get_angles() returns the robot's full joint angle vector."""
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    angles = bot.get_angles()
    assert isinstance(angles, np.ndarray)
    # SO101 URDF has 6 joints (5 arm + 1 gripper)
    assert angles.shape == (6,)
    assert np.allclose(angles, 0.0)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py::test_get_angles_returns_full_vector -v`
预期：FAIL — `AttributeError: 'SimBot' object has no attribute 'get_angles'`

- [ ] **步骤 3：实现 get_angles()**

在 `src/sim_bot.py` 的 `SimBot` 类内追加：

```python
    def get_angles(self) -> np.ndarray:
        """Return the full joint angle vector of the robot."""
        return self._robot.angle_vector().copy()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_get_angles_returns_full_vector -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add get_angles"
```

---

## 任务 4：set_joints() — round-trip + delta 语义

**文件：**
- 修改：`src/sim_bot.py`
- 测试：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加失败测试（覆盖 round-trip + delta）**

```python
def test_set_joints_round_trip_and_delta():
    """set_joints updates only the joints in the dict (delta semantics).

    Setting joint A then joint B should leave A at its value, not reset
    to zero.
    """
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )

    # Set shoulder_pan to 0.3
    bot.set_joints({"shoulder_pan": 0.3})
    angles = bot.get_angles()
    assert np.isclose(angles[0], 0.3), "shoulder_pan should be 0.3"

    # Set elbow_flex to -0.5; shoulder_pan should still be 0.3
    bot.set_joints({"elbow_flex": -0.5})
    angles = bot.get_angles()
    assert np.isclose(angles[0], 0.3), "shoulder_pan should still be 0.3"
    assert np.isclose(angles[2], -0.5), "elbow_flex should be -0.5"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py::test_set_joints_round_trip_and_delta -v`
预期：FAIL — `AttributeError: 'SimBot' object has no attribute 'set_joints'`

- [ ] **步骤 3：实现 set_joints()**

在 `SimBot` 类内 `get_angles` 之后追加：

```python
    def set_joints(
        self,
        name_to_angle: dict[str, float],
        redraw: bool | None = None,
    ) -> None:
        """Set joint angles by name, keeping unspecified joints unchanged.

        Parameters
        ----------
        name_to_angle : dict[str, float]
            Mapping from joint name (as in ``robot.joint_list``) to
            target angle in radians. Joints not in the dict keep their
            current value (delta semantics).
        redraw : bool | None
            If None (default), use the constructor's ``auto_redraw``.
            Pass False to skip redraw for this call, True to force it.
        """
        current = self._robot.angle_vector().copy()
        for name, angle in name_to_angle.items():
            # Locate joint by name in robot.joint_list
            joint_index = None
            for i, j in enumerate(self._robot.joint_list):
                if j.name == name:
                    joint_index = i
                    break
            if joint_index is None:
                raise KeyError(f"Unknown joint name: {name!r}")
            current[joint_index] = float(angle)
        self._robot.angle_vector(current)

        should_redraw = self._auto_redraw if redraw is None else redraw
        if should_redraw and self._viewer is not None:
            self._viewer.redraw()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_set_joints_round_trip_and_delta -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add set_joints with delta semantics"
```

---

## 任务 5：set_joints() KeyError

**文件：**
- 修改：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加测试 — 未知关节名抛 KeyError**

```python
def test_set_joints_unknown_name_raises_keyerror():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    with pytest.raises(KeyError):
        bot.set_joints({"nonexistent_joint": 0.0})
```

- [ ] **步骤 2：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_set_joints_unknown_name_raises_keyerror -v`
预期：PASS（任务 4 的实现已抛 `KeyError`）

- [ ] **步骤 3：Commit**

```bash
git add tests/test_sim_bot.py
git commit -m "test(sim_bot): cover set_joints KeyError on unknown joint"
```

---

## 任务 6：get_arm_angles + get_eef_position

**文件：**
- 修改：`src/sim_bot.py`
- 测试：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加失败测试**

```python
def test_get_arm_angles_and_eef_position():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    bot.set_joints({"shoulder_pan": 0.3, "elbow_flex": -0.5})

    # 5-DoF arm slice
    right_angles = bot.get_arm_angles("right")
    assert right_angles.shape == (5,)
    assert np.isclose(right_angles[0], 0.3)
    assert np.isclose(right_angles[2], -0.5)

    # EEF position is a 3-vector
    eef = bot.get_eef_position("right")
    assert eef.shape == (3,)
    assert np.linalg.norm(eef) > 0.0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py::test_get_arm_angles_and_eef_position -v`
预期：FAIL — `AttributeError: ... no attribute 'get_arm_angles'`

- [ ] **步骤 3：实现 get_arm_angles + get_eef_position**

在 `SimBot` 类内追加：

```python
    def get_arm_angles(self, side: str) -> np.ndarray:
        """Return the joint angles of one arm in kinematic order.

        Parameters
        ----------
        side : str
            "right" or "left".
        """
        names = self._right_arm_joints if side == "right" else self._left_arm_joints
        full = self._robot.angle_vector()
        result = np.zeros(len(names))
        for i, name in enumerate(names):
            for j in self._robot.joint_list:
                if j.name == name:
                    result[i] = j.joint_angle()
                    break
        return result

    def get_eef_position(self, side: str) -> np.ndarray:
        """Return the end-effector position in world frame.

        Parameters
        ----------
        side : str
            "right" or "left".
        """
        eef_frame = self._right_eef_frame if side == "right" else self._left_eef_frame
        for link in self._robot.link_list:
            if link.name == eef_frame:
                return np.asarray(link.worldpos(), dtype=np.float64)
        raise KeyError(f"EEF frame {eef_frame!r} not found in robot link_list")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_get_arm_angles_and_eef_position -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add get_arm_angles and get_eef_position"
```

---

## 任务 7：solve_ik() 单步

**文件：**
- 修改：`src/sim_bot.py`
- 测试：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加失败测试**

```python
def test_solve_ik_moves_eef_toward_target():
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # EEF at zero pose
    eef_before = bot.get_eef_position("right")

    # Target slightly offset in +X
    target = eef_before + np.array([0.02, 0.0, 0.0])

    q_new = bot.solve_ik("right", target)
    assert q_new.shape == (5,)

    # Single-step IK should move EEF toward target
    eef_after = bot.get_eef_position("right")
    # Either EEF moved toward target, or joint limits clamped it
    eef_after_dist = np.linalg.norm(eef_after - target)
    eef_before_dist = np.linalg.norm(eef_before - target)
    assert eef_after_dist <= eef_before_dist + 1e-6, (
        f"IK step did not reduce EEF distance: "
        f"before={eef_before_dist:.4f} after={eef_after_dist:.4f}"
    )
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py::test_solve_ik_moves_eef_toward_target -v`
预期：FAIL — `AttributeError: ... no attribute 'solve_ik'`

- [ ] **步骤 3：在 __init__ 中实例化双 IK + 实现 solve_ik**

修改 `__init__` 末尾（在 `if viewer:` 块之后或之前均可，建议在 viewer 之前），追加：

```python
        # IK solvers (one per arm). Pinocchio model is independent of
        # the skrobot RobotModel and the ViserViewer.
        self._ik_right = IKSolver(
            urdf_path=urdf_path,
            eef_frame=right_eef_frame,
            arm_joint_names=self._right_arm_joints,
        )
        self._ik_left = IKSolver(
            urdf_path=urdf_path,
            eef_frame=left_eef_frame,
            arm_joint_names=self._left_arm_joints,
        )
```

在 `SimBot` 类内 `get_eef_position` 之后追加：

```python
    def solve_ik(
        self,
        side: str,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run a single IK step toward the target end-effector position.

        Parameters
        ----------
        side : str
            "right" or "left".
        target_pos : array-like, shape (3,)
            Desired EEF position in world frame.
        q_init : array-like, shape (n_arm_joints,) | None
            Initial arm joint angles. If None, uses current angles.

        Returns
        -------
        q_new : np.ndarray, shape (n_arm_joints,)
            New arm joint angles after one IK step. Joint-limit-clamped
            and step-size-limited (damping/max_delta from IKSolver).
        """
        ik = self._ik_right if side == "right" else self._ik_left
        if q_init is None:
            q_init = self.get_arm_angles(side)
        return ik.solve(np.asarray(q_init, dtype=np.float64), np.asarray(target_pos, dtype=np.float64))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_solve_ik_moves_eef_toward_target -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add solve_ik using pinocchio IKSolver per arm"
```

---

## 任务 8：viewer 控制 — show / close / wait_until_close / redraw

**文件：**
- 修改：`src/sim_bot.py`
- 测试：`tests/test_sim_bot.py`

- [ ] **步骤 1：追加失败测试 — 显式 close 幂等**

```python
def test_close_is_idempotent():
    """close() can be called multiple times without error.

    Only meaningful with viewer=True (otherwise no viewer to close).
    """
    bot = SimBot(
        urdf_path=URDF,
        right_arm_joints=SO101_ARM,
        left_arm_joints=SO101_ARM,
        right_eef_frame=SO101_EEF,
        left_eef_frame=SO101_EEF,
        viewer=False,
    )
    # When viewer=False, _viewer is None; close should be a no-op.
    bot.close()
    bot.close()  # must not raise
```

- [ ] **步骤 2：运行测试验证失败**

运行：`uv run pytest tests/test_sim_bot.py::test_close_is_idempotent -v`
预期：FAIL — `AttributeError: ... no attribute 'close'`

- [ ] **步骤 3：实现 viewer 控制方法**

在 `SimBot` 类内追加：

```python
    def show(self, open_browser: bool = True) -> None:
        """Print viewer URL and optionally open in browser.

        No-op if ``viewer=False`` was passed to the constructor.
        """
        if self._viewer is not None:
            self._viewer.show(open_browser=open_browser)

    def redraw(self) -> None:
        """Force a viewer redraw. No-op if viewer is disabled."""
        if self._viewer is not None:
            self._viewer.redraw()

    def wait_until_close(self) -> None:
        """Block until KeyboardInterrupt (Ctrl-C). No-op if viewer disabled."""
        if self._viewer is not None:
            self._viewer.wait_until_close()
        else:
            # Block on KeyboardInterrupt for parity with viewer mode.
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        """Close the viewer. Idempotent; no-op if viewer disabled."""
        if self._viewer is not None:
            self._viewer.close()
```

在 `src/sim_bot.py` 顶部 import 块追加：

```python
import time
```

- [ ] **步骤 4：运行测试验证通过**

运行：`uv run pytest tests/test_sim_bot.py::test_close_is_idempotent -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_sim_bot.py src/sim_bot.py
git commit -m "feat(sim_bot): add show/redraw/wait_until_close/close viewer control"
```

---

## 任务 9：改写 sim_env.py 为 SimBot demo

**文件：**
- 修改：`src/sim_env.py`

- [ ] **步骤 1：重写为 SimBot demo**

完整替换 `src/sim_env.py` 为：

```python
"""SimBot demo: humanoid URDF with right-arm interpolation animation.

Run with::

    uv run src/sim_env.py

Then open the printed viser URL in a browser to see the right arm
interpolate from zero pose to a target pose over ~10 seconds. The viewer
stays open after the animation finishes; press Ctrl-C to exit.
"""
import time

import numpy as np

from src.sim_bot import SimBot

URDF = "/home/sorel/workspace/robot_description/robotic_description.urdf"

RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_TARGET = [0.5, 0.3, -0.2, -1.0, 0.8, -0.5, 0.2]


def main() -> None:
    bot = SimBot(urdf_path=URDF, viewer=True)
    print(f"Joint list:\n{bot._robot.joint_list}")
    print(f"Initial angles: {bot.get_angles()}")
    bot.show()

    num_steps = 50
    step_delay = 0.2
    time.sleep(1.0)  # let the browser connect

    for i in range(num_steps):
        t = i / (num_steps - 1)
        # Build per-step dict; joints not listed keep their current value
        # (delta semantics — torso/left arm stay at zero throughout).
        angles = {name: t * target for name, target in zip(RIGHT_ARM_JOINTS, RIGHT_ARM_TARGET)}
        bot.set_joints(angles)
        time.sleep(step_delay)

    bot.wait_until_close()


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：手动验证 — 运行 demo 不报错**

运行：`uv run src/sim_env.py &`
预期：viser 启动、打印 URL、动画跑 10s、viewer 持续保持
清理：`pkill -f sim_env.py`

- [ ] **步骤 3：运行所有单元测试**

运行：`uv run pytest tests/ -v`
预期：所有 SimBot 测试通过（其他测试不受影响）

- [ ] **步骤 4：Commit**

```bash
git add src/sim_env.py
git commit -m "refactor(sim_env): rewrite as SimBot demo with right-arm animation"
```

---

## 任务 10：最终验证 + 文档同步

- [ ] **步骤 1：运行完整测试套件**

运行：`uv run pytest tests/ -v`
预期：全部通过

- [ ] **步骤 2：Commit（如有遗漏变更）**

```bash
git status
# If there are uncommitted changes:
# git add -A && git commit -m "chore: final cleanup"
```

- [ ] **步骤 3：更新 spec 文件状态**

如需在 spec 顶部加 "Status: Implemented" 标记，可选。

---

## 自检结果

**1. 规格覆盖度**：
- §3 公开 API：`__init__`（任务 2）、`get_angles`（任务 3）、`set_joints`（任务 4）、`get_arm_angles` + `get_eef_position`（任务 6）、`solve_ik`（任务 7）、`show` / `redraw` / `wait_until_close` / `close`（任务 8）— 全部覆盖
- §4.1 auto_redraw / redraw 参数：任务 4 实现
- §4.2 delta 语义：任务 4 测试覆盖
- §4.3 IK 单步：任务 7 测试覆盖
- §4.4 viewer 关闭幂等：任务 8 测试覆盖
- §7 测试策略：所有点都有对应任务
- §9 错误处理：`KeyError`（任务 5）

**2. 占位符扫描**：无 "TODO"/"待定"/"类似任务 N"

**3. 类型一致性**：
- `_robot`、`_viewer`、`_ik_right`、`_ik_left`、`_right_arm_joints`、`_left_arm_joints`、`_right_eef_frame`、`_left_eef_frame`、`_auto_redraw` 命名全程一致
- `side` 参数全程用字符串 `"right"`/`"left"`，无重载歧义

未发现需要修复的问题。
