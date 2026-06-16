# RealSense 手势控制机械臂设计规格

## 概述

结合 `realsense_demo.py` 的手势识别与 `main_realsense.py` 的机械臂 IK 控制逻辑，实现：
- 仅在五指张开（Open Hand）时启用控制
- 通过手腕相对位移（dx/dy/dz）精确控制机械臂末端位置
- 左手控制左臂，右手控制右臂

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      主循环 (30Hz)                           │
│  polling HandDetector.shared_status / shared_motion          │
│     ├── 读取 motion["Left"]  → move_arm(left_arm_eef, ...)   │
│     └── 读取 motion["Right"] → move_arm(right_arm_eef, ...)  │
└─────────────────────────────────────────────────────────────┘
         ▲  threading.Event / dict 共享
         │
┌─────────────────────────────────────────────────────────────┐
│                  HandDetector 后台线程                       │
│  RealSense (30fps) + MediaPipe Hands → 计算 motion/anchor    │
│  仅在 num_fingers==5 时更新 anchor + motion                  │
└─────────────────────────────────────────────────────────────┘
```

## 坐标系映射

| HandDetector 状态 | 映射目标 | 说明 |
|---|---|---|
| `motion[hand_label]["x"]` | 机械臂末端 `dx` | 摄像头 x 差值 → 末端 X |
| `motion[hand_label]["y"]` | 机械臂末端 `dy` | 摄像头 y 差值 → 末端 Y（俯视） |
| `motion[hand_label]["depth_mm"]` | 机械臂末端 `dz` | 手腕深度差 → 末端 Z |

**灵敏度（可变增益）**：
```python
gain = 0.0005 + abs(displacement) * 0.0001
# 位移越大，增益越高，非线性映射
```

## 手势使能逻辑

| 手势 | 行为 |
|---|---|
| `num_fingers == 5` | 稳定性等待（连续 3 帧 + depth > 100mm） → 设置锚点 → 计算 motion |
| `num_fingers != 5` | 清除锚点 + `motion = {x:0, y:0, depth_mm:0}` |

**稳定性等待**：五指张开后，需连续 3 帧检测到 5 指且深度 > 100mm 才设置为锚点，避免第一帧波动误锚定。

## 锚点重置策略

| 事件 | 行为 |
|---|---|
| 手部暂时丢失（深度=0） | 清除 `motion_anchor[hand_label]` |
| 五指张开重新检测到 | 重新走稳定性等待流程 |
| 手势不再是 Open Hand | 清除锚点，下次重新开始 |

## 左右手分工

| 手 | 控制目标 |
|---|---|
| 左手（hand_label="Left"） | 左臂末端（left_arm_link7） |
| 右手（hand_label="Right"） | 右臂末端（right_arm_link7） |

## 共享状态接口

`HandDetector` 通过以下属性暴露给主循环：

```python
self.shared_status = {}   # {hand_label: {x, y, depth_mm, fingers, gesture}}
self.shared_motion = {}   # {hand_label: {x, y, depth_mm}}
```

主循环只读这两个 dict，主线程安全由 Python GIL + 无并发写入保证（HandDetector 线程独占写入）。

## 改动文件清单

1. 新建 `hand_tracker.py` — 从 `realsense_demo.py` 提取并改造为线程化 `HandDetector`
2. 修改 `main_realsense.py` — 集成 `HandDetector` 后台线程 + polling 逻辑
3. 无需修改 `realsense_demo.py`（保留为独立 demo）

## 风险与约束

- RealSense 断连最长阻塞 5s（后台线程处理），主循环不受影响
- 当前 `STEP_SIZE=0.05` 保持不变，增益在 `move_arm()` 调用前施加
