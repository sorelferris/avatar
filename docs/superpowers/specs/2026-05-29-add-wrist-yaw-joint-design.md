# 设计规格说明书：为 SO-101 机械臂增加 wrist_yaw 关节达到 6+1 自由度

## 1. 背景与目标
SO-101 机械臂原设计为 5+1 自由度。为了提高手腕的灵活性（例如让夹爪可以在不改变位置的情况下进行偏航姿态调整），我们需要在 `wrist_link`（腕部）与 `gripper_link`（夹爪）之间增加一个偏航（Yaw）自由度 `wrist_yaw`。这不仅能提供完整的 6 自由度机械臂姿态控制（6 DOF arm + 1 DOF gripper），还便于与现有的双臂或高级全身控制器进行对接。

## 2. 方案设计
为了保持原有物理特征的兼容性，我们将原来的 `wrist_link` 挂载到 `gripper_link` 之间的相对距离进行拆分。
* **原状态**：
  `wrist_link` -> `gripper_link` 的相对位置是 `xyz="5.55112e-17 -0.0611 0.0181"`。
* **新状态**：
  引入中间连杆 `wrist_yaw_link`。
  * `wrist_link` -> `wrist_yaw_link` 相对位移设定为原位移的前半部分：`xyz="0 -0.0305 0.009"`。
  * `wrist_yaw` 关节为旋转铰链关节，旋转轴为 `[1, 0, 0]`（在 MuJoCo 坐标系中定义为绕 X 轴，或在 URDF 中与整机坐标系统一）。
  * `wrist_yaw_link` -> `gripper_link` 相对位移设定为原位移的后半部分：`xyz="0 -0.0306 0.0091"`。
  这样当 `wrist_yaw` 关节处于 $0^\circ$ 状态时，`gripper_link` 相对于 `wrist_link` 的总位移与之前完全一致。

## 3. 具体修改项

### A. MuJoCo 模型描述 (`assets/SO101/so101_new_calib.xml`)
1. 在 `wrist` 连杆对应的 `<body>` 节点下，新增子节点 `<body name="wrist_yaw">`。
2. 将 `wrist_yaw` 关节定义在 `wrist_yaw` 连杆中：
   ```xml
   <joint axis="1 0 0" name="wrist_yaw" type="hinge" range="-1.74533 1.74533" class="sts3215"/>
   ```
3. 将 `gripper` 连杆连同其所有的子节点（如 `moving_jaw_so101_v1`）整体嵌套到 `wrist_yaw` 的 `<body>` 下，并修正其 `pos`。
4. 在 `<actuator>` 标签组中，新增 `wrist_yaw` 位置控制器：
   ```xml
   <position class="sts3215" name="wrist_yaw" joint="wrist_yaw" forcerange="-3.35 3.35" ctrlrange="-1.74533 1.74533"/>
   ```

### B. URDF 模型描述 (`assets/SO101/so101_new_calib.urdf`)
1. 新增连杆定义 `<link name="wrist_yaw_link">`。为了简化，使用极其微小的质量块或空连杆，无需添加复杂的网格。
2. 新增关节 `<joint name="wrist_yaw" type="revolute">`：
   * `parent` 连杆为 `wrist_link`。
   * `child` 连杆为 `wrist_yaw_link`。
   * `origin` 设定为 `xyz="0 -0.0305 0.009"`。
   * 旋转轴定义为绕 X 轴 `axis="1 0 0"`。
3. 修改原本的 `wrist_roll` 关节：
   * 将其 `parent` 从 `wrist_link` 修改为 `wrist_yaw_link`。
   * 将其 `origin` 相对位移修正为 `xyz="0 -0.0306 0.0091"`。
4. 为新关节新增传动机构 `<transmission name="wrist_yaw_trans">`。

### C. 核心控制逻辑与环境更新
1. **`src/sim_env.py`**：
   在 `ARM_JOINTS` 类变量列表中加入 `"wrist_yaw"`：
   ```python
   ARM_JOINTS = [
       "shoulder_pan",
       "shoulder_lift",
       "elbow_flex",
       "wrist_flex",
       "wrist_yaw",
       "wrist_roll",
   ]
   ```
2. **`src/ik_solver.py`**：
   在默认初始化中将 `"wrist_yaw"` 插入到 `arm_joint_names` 列表的相应位置。
3. **`assets/SO101/interactive_viewer.py`**：
   在 `JOINT_NAMES` 列表中插入 `"wrist_yaw"`。
4. **`tests/test_ik_solver.py`**：
   更新相关的测试输入，将关节维度从 5 升级为 6（例如 `np.zeros(5)` 变更为 `np.zeros(6)`）。

## 4. 验证方法与成功标准
1. 运行 `tests/test_ik_solver.py`，确保其全部通过，说明逆运动学算法能完美支持 6 自由度，且各关节限位约束表现正常。
2. 在 MuJoCo viewer 中启动该模型，确保模型能正常渲染且不发生关节飘逸、爆破等物理仿真异常，且 6 个关节的滑块全部能被独立操控。
