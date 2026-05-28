# Hand Visual Teleoperation System — Design Spec

## Overview

人手视觉遥操作系统：通过普通单目摄像头实时捕捉双手 3D 位置，以食指指尖作为机器人末端执行器（EEF），控制 Unitree G1 人形机器人双臂各 5 DoF，在 MuJoCo 仿真环境中验证遥操作逻辑。

## Requirements

### Functional

- 实时手部检测：MediaPipe Hands 检测双手 21 个关键点
- EEF 3D 坐标提取：食指指尖 landmark[8] 的 3D 位置
- 深度估计：MiDaS small 模型提供单目深度，替换 MediaPipe 归一化 Z 值
- 坐标变换：相机坐标系→机器人坐标系重映射 + 线性缩放到工作空间
- 坐标平滑：EMA 低通滤波消除高频抖动
- 防超程保护：目标裁剪到臂长边界球内
- IK 求解：MuJoCo 雅可比 DLS 逆运动学
- 仿真控制：PD 力矩控制驱动 G1 手臂关节
- 可视化：OpenCV 窗口（手部关键点叠加 + HUD）+ MuJoCo viewer（机器人状态）

### Non-functional

- 帧率 ≥ 25 FPS（检测管线）
- 仿真物理步 500Hz，控制频率 50Hz
- 检测延迟 < 40ms（MediaPipe ~15ms + MiDaS ~20ms）
- 鲁棒性：抗轻微遮挡、光照变化；手部丢失时冻结位置

## Architecture

单进程多线程，6 个模块通过共享数据结构通信。

```
摄像头(30fps) → 手部检测(MediaPipe+MiDaS) → 坐标处理(变换+平滑) → IK求解(MuJoCo) → 仿真步进 → 双窗口可视化
```

### Modules

| Module | File | Responsibility | Thread |
|--------|------|----------------|--------|
| Camera | `src/camera.py` | OpenCV 摄像头捕获，帧缓冲 | Dedicated |
| Hand Detector | `src/hand_detector.py` | MediaPipe Hands + MiDaS 深度估计 | Detection thread |
| Coordinate Processor | `src/coordinate_processor.py` | 归一化→工作空间变换、EMA 平滑、防超程裁剪 | Detection thread |
| IK Solver | `src/ik_solver.py` | MuJoCo 雅可比 DLS IK + PD 力矩控制 | Main loop |
| Sim Environment | `src/sim_env.py` | MuJoCo 仿真环境：加载 G1、固定基座、步进 | Main loop |
| Visualizer | `src/visualizer.py` | OpenCV 窗口 + MuJoCo viewer | Dedicated |

### Shared Data Structure

```python
@dataclass
class TeleopState:
    lock: threading.Lock
    # Raw landmarks
    left_landmarks: np.ndarray | None   # (21, 3) normalized
    right_landmarks: np.ndarray | None
    # Processed 3D targets (robot frame, meters)
    left_target: np.ndarray             # (3,) smoothed workspace coord
    right_target: np.ndarray
    # Status
    left_hand_detected: bool
    right_hand_detected: bool
    fps: float
```

## Hand Detection & Depth Estimation

### MediaPipe Hands

- `max_num_hands=2`
- `model_complexity=1`
- `min_detection_confidence=0.7`
- `min_tracking_confidence=0.5`
- 食指指尖 = `landmark[8]` (`INDEX_FINGER_TIP`)
- 左手 landmark[8] → 左臂 EEF；右手 landmark[8] → 右臂 EEF
- 手部丢失时保持最后有效位置，设置 `hand_lost` 标志位

### MiDaS Depth Estimation

- 模型：`MiDaS_small` 通过 `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` 加载
- 输入：当前帧 BGR→RGB，resize 384×384
- 输出：相对深度图，归一化到 [0, 1]
- 深度采样：用 landmark[8] 像素坐标在深度图上取值
- 尺度校正：`real_depth = depth_scale * (reference_palm_width / measured_palm_width) * midas_value`
- 参考手掌宽度：启动时校准或使用默认值 8cm

## Coordinate Processing

### Pipeline

```
MediaPipe 归一化坐标 (0~1)
    ↓
MiDaS 深度替换 Z 轴
    ↓
坐标系重映射（相机→机器人）
    ↓
线性缩放到工作空间
    ↓
EMA 平滑
    ↓
防超程裁剪
    ↓
目标 3D 坐标
```

### Coordinate Remapping

- 相机坐标系：X 右、Y 下、Z 前（OpenCV 惯例）
- 机器人坐标系：X 前、Y 左、Z 上（MuJoCo 惯例）
- 映射：`robot_x = cam_z`，`robot_y = -cam_x`，`robot_z = -cam_y`

### Linear Scaling

- 手部活动范围：~30cm × 30cm × 30cm
- G1 臂工作空间：肩关节前方 ~0.4m 半径球体
- 缩放因子：各轴独立可调
- 偏移：启动时食指位置为零点

### EMA Smoothing

- `smoothed = α * new + (1-α) * smoothed_prev`
- α = 0.3（约 3 帧收敛）
- 三轴独立滤波
- 手部丢失超过 0.5s 冻结最后位置

### Anti-overreach

- 裁剪目标到工作空间边界球内
- 边界半径 = 臂长总和 × 0.9
- 裁剪方式：保持方向，限制距离

## IK Solver & Simulation

### MuJoCo Environment

- 加载 `assets/g1_23dof.xml`
- 固定 pelvis：用 `equality/weld` 约束或修改 joint 类型
- 仿真参数：`opt.timestep = 0.002`（500Hz），控制频率 50Hz
- 启用碰撞检测

### Jacobian IK

- 左臂关节：`[left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll]`
- 右臂对称
- `mj_jac(model, data, jac_pos, jac_rot, body_id)` 计算 3×N 位置雅可比
- DLS：`Δq = J^T (J J^T + λ²I)^{-1} (x_target - x_current)`
- λ = 0.1（阻尼因子）
- 步幅限幅：`Δq = clip(Δq, -0.1, 0.1)` rad/step
- 关节限位：`q = clip(q + Δq, q_min, q_max)`

### PD Torque Control

XML actuator 为 `<motor>`（力矩控制），IK 输出角度需 PD 转力矩：

```
torque = Kp * (q_target - q_current) - Kd * qdot
Kp = 50, Kd = 5
```

### Control Loop

```python
while running:
    target_L, target_R = processor.get_targets()
    q_left  = jacobian_ik(left_arm_chain, target_L)
    q_right = jacobian_ik(right_arm_chain, target_R)
    data.ctrl[left_arm_ids] = pd_torque(q_left, qdot_left)
    data.ctrl[right_arm_ids] = pd_torque(q_right, qdot_right)
    mujoco.mj_step(model, data)
```

## Visualization

### OpenCV Window

- 摄像头原始帧 + MediaPipe 21 关键点连线（绿色骨架）
- 食指指尖 [8] 红色大圆点
- HUD：当前 EEF 3D 坐标、FPS、手部状态
- 双手左右分色（蓝/红）

### MuJoCo Viewer

- 独立窗口，`mujoco.viewer.launch_passive(model, data)`
- EEF 目标位置 site（绿色）vs 当前 EEF site（黄色）

## File Structure

```
avatar/
├── pyproject.toml
├── main.py
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── hand_detector.py
│   ├── coordinate_processor.py
│   ├── ik_solver.py
│   ├── sim_env.py
│   └── visualizer.py
├── assets/
│   ├── g1_23dof.xml
│   ├── g1_23dof.urdf
│   └── meshes/
└── tests/
    ├── test_coordinate_processor.py
    └── test_ik_solver.py
```

## Dependencies

- `mediapipe` >= 0.10
- `opencv-python` >= 4.8
- `mujoco` >= 3.0
- `numpy` >= 1.24
- `torch` >= 2.0 (MiDaS inference)
- `timm` >= 0.9 (MiDaS model backbone)

MiDaS 通过 `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` 加载，首次运行自动下载权重。

## Known Limitations

1. **单目深度精度弱**：MiDaS 提供相对深度，Z 轴精度低于 XY；手掌尺度校正可缓解但不消除
2. **遮挡敏感**：严重自遮挡/外物遮挡时关键点精度明显下降
3. **手部尺度差异**：不同人手大小通过手掌宽度校正，仍有残余误差
4. **5 DoF 冗余**：3D 位置目标对 5 DoF 欠约束，IK 会收敛到最近解，不保证唯一
5. **PD 参数调优**：Kp/Kd 需要在仿真中实测调整，初始值为经验值

## Scope

- **In scope**：实时手部检测、3D 坐标提取、坐标变换平滑、MuJoCo 仿真环境、雅可比 IK、双窗口可视化、单元测试
- **Out of scope**：实物机器人通信、抓取/力控、全身平衡控制、多摄像头融合、训练自定义模型
