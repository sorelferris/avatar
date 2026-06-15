# Joy-Con IMU 双臂控制设计

## 概述

使用左右两个 Joy-Con 手柄的 IMU 姿态控制 SO101 双臂机械臂末端位置。按住 L 键激活左臂控制，按住 R 键激活右臂控制。

## 控制映射

| 手柄 | 按键 | 控制目标 | 控制方式 |
|------|------|----------|----------|
| 左 Joy-Con | 按住 L 键 | 左臂末端位置 | IMU 方向变化量映射 |
| 右 Joy-Con | 按住 R 键 | 右臂末端位置 | IMU 方向变化量映射 |

双手可同时激活，独立控制双臂。

## IMU 姿态映射逻辑

1. 读取 `joycon.get_imu()['L']['direction']` — 归一化 3D 方向向量（世界坐标系）
2. 每帧记录上一次的 `direction`，计算 `delta_dir = current - prev`
3. `target_pos += delta_dir * SENSE_SCALE`
4. 末端位置限幅（球形工作空间，半径 0.36m）

## IK 求解

- 使用 `robot.inverse_kinematics()`（skrobot 内置 IK）
- `rotation_mask=False, position_mask=True`（仅位置 IK）
- 目标坐标系 `Coordinates(pos=target_pos.tolist(), rot=[0, 0, 0])`

## 安全参数

| 参数 | 值 | 说明 |
|------|----|------|
| `SENSE_SCALE` | 0.5 | IMU 方向变化量缩放因子（m/frame） |
| `MAX_DELTA` | 0.05 | 单帧最大位置变化量（m） |
| `MAX_RADIUS` | 0.36 | 末端位置限幅半径（m） |
| `calibration_seconds` | 2.0 | 启动时手柄静止校准时间（s） |

## 可视化

- ViserViewer（skrobot 内置 WebSocket 可视化器）
- 每帧调用 `viewer.redraw()` 推送更新

## 新增文件

- `main_joycon_imu.py` — 主程序