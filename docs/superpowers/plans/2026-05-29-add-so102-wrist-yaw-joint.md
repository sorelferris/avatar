# SO-102 6+1 自由度机械臂（增加 wrist_yaw 关节）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不影响原有 `so101_new_calib` 模型的前提下，创建全新的 `so102.urdf` 和 `so102.xml` 模型文件，为手腕增加一个偏航关节 `wrist_yaw`，并确保所有测试、仿真环境和 IK 求解器支持全新的 6+1 自由度配置。

**架构：**
1. 将原 `wrist` 到 `gripper` 之间的位移拆分为：`wrist` -> `wrist_yaw_link` 和 `wrist_yaw_link` -> `gripper`。
2. 插入旋转轴为 `axis="1 0 0"` (Yaw) 的 `wrist_yaw` 关节。
3. 新建 `so102.urdf` 与 `so102.xml` 以保证原有 SO-101 文件的独立完整。
4. 在 Python 核心控制代码中提供对 SO-102 的支持，并修改/编写测试用例确保其 100% 正确运行。

**技术栈：** MuJoCo MJCF XML, URDF, Pinocchio, Python, Pytest

---

## 文件结构
- 创建：`assets/SO101/so102.xml` — 基于 `so101_new_calib.xml` 改写的 6+1 自由度 MuJoCo 模型。
- 创建：`assets/SO101/so102.urdf` — 基于 `so101_new_calib.urdf` 改写的 6+1 自由度 URDF 模型。
- 修改：`src/ik_solver.py` — 支持动态加载 5 自由度或 6 自由度关节配置。
- 修改：`src/sim_env.py` — 支持动态加载 5 自由度或 6 自由度仿真配置。
- 修改：`assets/SO101/interactive_viewer.py` — 支持加载 `so102.xml` 对应的 6+1 轴动作及滑块。
- 创建：`tests/test_so102.py` — 专为 SO-102 6+1 自由度编写的完整 IK、FK、关节限位与物理模型校验单元测试。

---

## 详细实施步骤

### 任务 1：创建 `so102.xml` (MuJoCo 描述文件)

**文件：**
- 创建：`assets/SO101/so102.xml`

- [ ] **步骤 1：复制并改写模型文件**
  读取 `assets/SO101/so101_new_calib.xml`，将其复制为 `assets/SO101/so102.xml`。
  
- [ ] **步骤 2：在 XML 中插入 `wrist_yaw` 连杆和关节**
  定位至 `wrist` 的 `<body>` 标签内部，将原来的 `gripper` `<body>` 用全新的 `wrist_yaw` `<body>` 进行嵌套。具体修改如下：
  * 将 `gripper` 的 `pos` 由原来的 `5.55112e-17 -0.0611 0.0181` 拆分为：
    * `wrist_yaw` 的 `pos="0 -0.0305 0.009"`，方向角 `quat="1 0 0 0"`。
    * `gripper` 的 `pos="0 -0.0306 0.0091"`。
  * 在 `wrist_yaw` 内部定义关节 `wrist_yaw`：
    ```xml
    <joint axis="1 0 0" name="wrist_yaw" type="hinge" range="-1.74533 1.74533" class="sts3215"/>
    ```
  
- [ ] **步骤 3：在 `<actuator>` 标签组中加入 `wrist_yaw` 位置驱动器**
  ```xml
  <position class="sts3215" name="wrist_yaw" joint="wrist_yaw" forcerange="-3.35 3.35" ctrlrange="-1.74533 1.74533"/>
  ```

---

### 任务 2：创建 `so102.urdf` (URDF 描述文件)

**文件：**
- 创建：`assets/SO101/so102.urdf`

- [ ] **步骤 1：复制并改写 URDF 模型**
  复制 `assets/SO101/so101_new_calib.urdf` 并重命名为 `assets/SO101/so102.urdf`。
  
- [ ] **步骤 2：添加 `wrist_yaw_link` 和 `wrist_yaw` 关节**
  在 `wrist_link` 节点后新增：
  ```xml
  <link name="wrist_yaw_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.03"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <joint name="wrist_yaw" type="revolute">
    <origin xyz="0 -0.0305 0.009" rpy="0 0 0"/>
    <parent link="wrist_link"/>
    <child link="wrist_yaw_link"/>
    <axis xyz="1 0 0"/>
    <limit effort="10" velocity="10" lower="-1.74533" upper="1.74533"/>
  </joint>
  ```

- [ ] **步骤 3：重构 `wrist_roll` 关节连接**
  将 `wrist_roll` 的 `parent` 改为 `wrist_yaw_link`，并将 `origin` 修正为：
  ```xml
  <origin xyz="0 -0.0306 0.0091" rpy="1.5708 0.0486795 3.14159"/>
  ```

- [ ] **步骤 4：添加 `wrist_yaw_trans` 传动结构**
  在 `wrist_roll_trans` 附近添加：
  ```xml
  <transmission name="wrist_yaw_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="wrist_yaw">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="motor_wrist_yaw">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
  ```

---

### 任务 3：支持代码动态加载 (IK Solver, Sim Env, Viewer)

**文件：**
- 修改：`src/ik_solver.py`
- 修改：`src/sim_env.py`
- 修改：`assets/SO101/interactive_viewer.py`

- [ ] **步骤 1：重构 `src/ik_solver.py` 支持 6 自由度自动探测**
  在 `__init__` 函数中，若 `arm_joint_names` 为 None，则利用 `urdf_path` 是否包含 "so102" 来动态决定使用 5 轴还是 6 轴默认关节：
  ```python
  # line 32 附近
  if arm_joint_names is None:
      if "so102" in urdf_path:
          arm_joint_names = [
              "shoulder_pan",
              "shoulder_lift",
              "elbow_flex",
              "wrist_flex",
              "wrist_yaw",
              "wrist_roll",
          ]
      else:
          arm_joint_names = [
              "shoulder_pan",
              "shoulder_lift",
              "elbow_flex",
              "wrist_flex",
              "wrist_roll",
          ]
  ```

- [ ] **步骤 2：重构 `src/sim_env.py` 以支持动态关节配置**
  重构 `SimEnvironment` 接收 `xml_path` 后，根据实际 XML 包含的关节动态构建 `arm_joints`，而不是使用硬编码的 `ARM_JOINTS`：
  ```python
  # 修改：__init__ 
  # 自适应寻找除 gripper 以外的所有 hinge 类型关节作为 arm_joints
  self.arm_joint_names = []
  for i in range(self.model.njnt):
      jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if jname != "gripper" and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
          self.arm_joint_names.append(jname)
  ```

- [ ] **步骤 3：修改 `assets/SO101/interactive_viewer.py` 以自适应所有关节**
  ```python
  # 将硬编码的 JOINT_NAMES 改为自适应读取 model 中所有的 actuators 名字：
  # 在 SO101Viewer.__init__ 中，不再引用硬编码 JOINT_NAMES，
  # 直接动态填充从 xml 获取的 actuator 和 joint
  ```

---

### 任务 4：编写测试验证 SO-102 关节链、运动学与物理性能

**文件：**
- 创建：`tests/test_so102.py`

- [ ] **步骤 1：编写 6 自由度专属单元测试**
  在 `tests/test_so102.py` 中编写以下测试用例：
  1. `test_so102_joint_count` — 验证 IK 求解器检测到 6 个控制关节，MuJoCo 模型拥有 6 个手臂关节。
  2. `test_so102_fk_at_zero` — 验证全零位下的 Forward Kinematics 正常运行且产生非零 EEF。
  3. `test_so102_ik_convergence` — 验证 6DOF 的 IK 逆运动学求解，随机给一个可达点，确保 DLS IK 完美收敛且误差极小 (< 1mm)。
  4. `test_so102_joint_limits` — 验证 IK 求解时所有 6 个关节始终遵守其在 URDF 中声明的各自限位。

- [ ] **步骤 2：运行测试验证成功**
  运行命令：`pytest tests/test_so102.py -v`
  预期：所有测试全部完美 PASS。

- [ ] **步骤 3：运行原有 SO-101 测试确保无 regression 损害**
  运行命令：`pytest tests/test_ik_solver.py -v`
  预期：原有 5 轴测试依旧 100% PASS，实现完美向前兼容。
