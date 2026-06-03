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
    def set_joint(self, name: str, angle: float, redraw: bool | None = None) -> None
    def set_joints(self, name_to_angle: dict[str, float], redraw: bool | None = None) -> None
    def set_arm_angles(
        self,
        side: str | np.ndarray,           # "right"/"left"（str）或 14-DoF 向量（np.ndarray）
        av: np.ndarray | None = None,     # 7-DoF 单臂；当 side 是 np.ndarray 时忽略
        redraw: bool | None = None,
    ) -> None
    # 重载判定规则：isinstance(side, np.ndarray) → 14-DoF 双臂；否则 side 必为 "right"/"left"，av 必为 7-DoF
    def set_full_angles(self, av_n: np.ndarray, redraw: bool | None = None) -> None

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

**`side` 参数**只接受字符串 `"right"` 或 `"left"`；14-DoF 双臂向量通过 `set_arm_angles(av_14)` 重载传入。

---

## 4. 行为细节

### 4.1 auto_redraw 与 redraw 参数
- `auto_redraw=True`（默认）：每次 `set_*` 内部最后调用 `self._viewer.redraw()`
- `redraw=None` 跟随 `auto_redraw` 行为
- `redraw=False` 跳过本次 redraw（高频场景用）
- 显式 `redraw=True` 总是触发

### 4.2 set_arm_angles 重载
- `set_arm_angles("right", av_7)`：只设右臂 7 维
- `set_arm_angles("left",  av_7)`：只设左臂 7 维
- `set_arm_angles(av_14)`：双臂 14 维（顺序：右臂在前，左臂在后）
- 内部统一转成完整 N-DoF 向量后调 `robot.angle_vector()`，避免索引错位

### 4.3 IK
- `SimBot.__init__` 时为左右臂各实例化一个 `IKSolver`（`src/ik_solver.py`）
- `solve_ik(side, target, q_init=None)`：选对应 `_ik_<side>` 调一次 `solve()`（单步）
- 不在 `SimBot` 内做多步收敛；调用方如需多步收敛，调 `IKSolver.solve_to_convergence()`（已在 ik_solver.py 暴露）

### 4.4 viewer 关闭与脚本退出
- `close()` 显式关 viewer，幂等
- `wait_until_close()` 阻塞直到 `KeyboardInterrupt`（复用 `_InteractiveViewerMixin`）
- 脚本结束时 viser 的 atexit handler 兜底

---

## 5. 数据流（JoyCon 集成示例）

```
JoyCon L  ──IK──▶  SimBot.solve_ik("left",  target_L)  ──▶  set_arm_angles("left",  q_L)
JoyCon R  ──IK──▶  SimBot.solve_ik("right", target_R)  ──▶  set_arm_angles("right", q_R)
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
  - `set_joint` / `set_joints` round-trip：设后 `get_angles()` 读回一致
  - `set_arm_angles("right", av)`：`get_eef_position("right")` 与 forward kinematics 一致
  - `set_arm_angles(av_14)`：左右臂 EEF 位置与各自 forward kinematics 一致
  - `solve_ik("right", target)`：调用 `solve_to_convergence(max_iter=50, tol=1e-3)` 后 `forward_kinematics` 误差 < 1e-3 m（用 ik_solver 已有的多步收敛方法）
  - `viewer=False` 模式：构造不启动浏览器、不抛错
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
- `set_joint` 关节名不存在：`KeyError`
- `set_arm_angles` 向量长度不匹配：`ValueError`
- `solve_ik` 不可达：返回 `q_init`（ik_solver 已 clip 到 joint limits）
- viewer 重复 `close()`：幂等
