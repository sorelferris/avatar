# SimBot 类设计规格

**日期**: 2026-06-03
**目标**: 封装人形机器人双臂仿真的可视化与控制层，作为 JoyCon 遥操作的可视化层（也可独立运行 demo）。

---

## 1. 背景

`src/sim_env.py` 当前是一段线性脚本（加载 URDF → 创建 ViserViewer → 跑插值动画 → wait），无法被 JoyCon 遥操作等其他模块复用。同时 `main_joycon.py` 仍依赖旧的 `MujocoEnvironment`（已不在 sim_env.py 中），集成入口缺失。

需要一个薄层 `SimBot`，把"机器人 + 可视化 + IK 求解"封装为可复用的类，外部通过编程 API 控制双臂并实时显示。

**约束**：
- 零 MuJoCo 依赖（IK 用 Pinocchio）
- 双臂均支持
- 默认自动 redraw，但高频场景可关

---

## 2. 架构

```
┌──────────────────────────────────────────────┐
│  SimBot (src/sim_bot.py)                     │
│  ┌────────────────┐  ┌────────────────────┐  │
│  │  RobotModel    │  │  ViserViewer       │  │
│  │  (skrobot)     │  │  (skrobot)         │  │
│  │  - joints      │  │  - handles         │  │
│  │  - kinematics  │  │  - browser UI      │  │
│  └────────────────┘  └────────────────────┘  │
│  ┌───────────────────────────────────────┐   │
│  │  IKSolver × 2 (Pinocchio, left/right) │   │
│  └───────────────────────────────────────┘   │
│            ▲                    ▲             │
│            └─── set_*(...) ─────┘             │
│              (auto_redraw 默认 True)          │
└──────────────────────────────────────────────┘
        ▲             ▲             ▲
        │             │             │
   独立 demo      JoyCon 集成     其他脚本
   (sim_env.py)  (main_joycon.py) (其他)
```

---

## 3. 公开 API

```python
class SimBot:
    def __init__(
        self,
        urdf_path: str,
        right_arm_joints: list[str] | None = None,   # 默认 ["right_arm_joint1", ..., "right_arm_joint7"]
        left_arm_joints:  list[str] | None = None,   # 默认 ["left_arm_joint1", ..., "left_arm_joint7"]
        right_eef_frame:  str = "right_arm_link7",   # pinocchio frame 名（与 URDF 一致）
        left_eef_frame:   str = "left_arm_link7",    # pinocchio frame 名（与 URDF 一致）
        viewer: bool = True,
        auto_redraw: bool = True,
    ) -> None:
        """双臂 + ViserViewer。可关 viewer 做 headless 测试。
        eef_frame 必须是 pinocchio model 中存在的 frame 名（默认取末端 link 名）。"""

    # —— 关节设置（auto_redraw 默认 True，可 redraw=False 关单次）——
    def set_joints(self, name_to_angle: dict[str, float], redraw: bool | None = None) -> None
    #   未在 dict 中的关节保持当前值（delta 语义）。设单个关节：set_joints({"right_arm_joint1": 0.5})。
    #   设多关节：set_joints({"right_arm_joint1": 0.5, "left_arm_joint2": -0.3, ...})。

    # —— 读取 ——
    def get_angles(self) -> np.ndarray                    # 完整 N-DoF
    def get_arm_angles(self, side: str) -> np.ndarray     # 7-DoF 单臂
    def get_eef_position(self, side: str) -> np.ndarray   # 单臂 EEF (3,)

    # —— IK（Pinocchio only）——
    def solve_ik(
        self,
        side: str,                                        # "right" 或 "left"
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,                 # 默认当前角
    ) -> np.ndarray                                       # 新 7-DoF

    # —— Viewer 控制 ——
    def show(self, open_browser: bool = True) -> None
    def redraw(self) -> None
    def wait_until_close(self) -> None
    def close(self) -> None
```

**`side` 参数**只接受字符串 `"right"` 或 `"left"`。

---

## 4. 行为细节

### 4.1 auto_redraw 与 redraw 参数
- `auto_redraw=True`（默认）：每次 `set_joints` 内部最后调用 `self._viewer.redraw()`
- `redraw=None` 跟随 `auto_redraw` 行为
- `redraw=False` 跳过本次 redraw（高频场景用）
- 显式 `redraw=True` 总是触发

### 4.2 set_joints delta 语义
- 未在 dict 中的关节**保持当前值**（不重置为 0）
- 内部实现：先 `get_angles()` 拿到当前完整 N-DoF 向量，再按 dict 更新对应位置，再 `robot.angle_vector()`
- 已知关节名必须在 `robot.joint_list` 中；否则 `KeyError`
- 关节值会被 `robot.angle_vector()` 自动 clip 到 joint limits（skrobot 内置行为）

### 4.3 IK
- `SimBot.__init__` 时为左右臂各实例化一个 `IKSolver`（`src/ik_solver.py`）
- `solve_ik(side, target, q_init=None)`：选对应 `_ik_<side>` 调一次 `solve()`（**单步**，适合实时遥操作）
- 调用方如需多步收敛，可直接用 `IKSolver.solve_to_convergence()`（`ik_solver.py` 已暴露）

### 4.4 viewer 关闭与脚本退出
- `close()` 显式关 viewer，幂等
- `wait_until_close()` 阻塞直到 `KeyboardInterrupt`（复用 `_InteractiveViewerMixin`）
- 脚本结束时 viser 的 atexit handler 兜底

---

## 5. 数据流（JoyCon 集成示例）

```
JoyCon L  ──IK──▶  SimBot.solve_ik("left",  target_L)  ──▶  set_joints({"left_arm_joint1":  q_L[0], ...})
JoyCon R  ──IK──▶  SimBot.solve_ik("right", target_R)  ──▶  set_joints({"right_arm_joint1": q_R[0], ...})
                                                                │
                                              auto_redraw=True   │
                                                                ▼
                                                          viser 浏览器
```

---

## 6. 文件结构

```
src/
├── sim_bot.py        ← 新建：SimBot 类
├── sim_env.py        ← 改为 SimBot 演示 demo（首段插值动画 + wait_until_close）
├── ik_solver.py      ← 不变（纯 Pinocchio，SimBot 复用）
├── joycon_utils.py   ← 不变
└── ...
main_joycon.py       ← 不在本次改动范围（可后续用 SimBot 重写）
```

---

## 7. 测试策略

- **单元测试**：`tests/test_sim_bot.py`
  - `set_joints({...})` round-trip：设后 `get_angles()` 读回一致
  - `set_joints` delta 语义：先设 `{"right_arm_joint1": 0.5}`，再设 `{"left_arm_joint1": 0.3}`，然后 `get_angles()` 中 `right_arm_joint1` 仍为 0.5（左关节动不影响右关节）
  - `solve_ik("right", target)`：调一次后 `get_eef_position("right")` 相比调前位置向 target 方向移动（验证单步工作）
  - `viewer=False` 模式：构造不启动浏览器、不抛错
  - `set_joints` 未知关节名：`KeyError`
- **不测**：浏览器端 UI 渲染（依赖外部进程）
- **手动验证**：`sim_env.py` demo 能跑、动画可见

---

## 8. 不做的事（YAGNI）

- ❌ 不集成轨迹规划/插值（SimBot 是薄层，插值由调用方做）
- ❌ 不做力矩/动力学控制（那是控制层）
- ❌ 不做录像/截图（viewer 自己有）
- ❌ 不支持多机器人（一个 SimBot = 一个 robot model）
- ❌ 不在 SimBot 内做多步 IK 收敛（暴露单步 + 调用方用 `solve_to_convergence`）

---

## 9. 错误处理

- `urdf_path` 不存在：`FileNotFoundError`（让 pinocchio 自然抛出）
- `set_joints` 关节名不在 `robot.joint_list` 中：`KeyError`
- `set_joints` 关节值越界：skrobot `robot.angle_vector()` 自动 clip
- `solve_ik` 不可达：返回 `solve_to_convergence` 的最终 `q`（可能未收敛）
- viewer 重复 `close()`：幂等
