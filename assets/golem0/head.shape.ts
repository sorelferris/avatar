// @ts-ignore
import { drawCircle, drawRectangle } from 'replicad';
// @ts-ignore
import { holes, patterns } from 'shapeitup';

// ---------------------------------------------------------------------------
// 参数 — Atlas-inspired 通用机器人头部
// 所有尺寸单位：毫米
// ---------------------------------------------------------------------------
export const params = {
  // 头部主体
  headRadius: 90,
  headLength: 150,
  coverThick: 2, // 头部中空壁厚

  // 面部特征（贴在头部前端面 Z = 0 上，朝 -Z 方向是"脸"）
  facePanelRadius: 80, // 显示屏（上下半圆）半径
  panelThick: 1, // 显示屏厚度
  ledOuterRadius: 85, // 灯带外圈
  ledInnerRadius: 80, // 灯带内圈（=显示屏外缘，紧密贴合）
  ledThick: 1.5, // 灯带厚度，比显示屏略厚以凸出
  sensorWidth: 140, // 传感器横条 X 方向长度
  sensorHeight: 8, // 传感器横条 Y 方向高度
  sensorThick: 1.5, // 传感器条厚度（同灯带）

  // 颈部 — 从头部侧面 (Y = -headRadius) 沿 -Y 方向伸出
  // Z 位置 = 头部后端附近 (neckAttachZ)，颈部总高 60mm
  neckAttachZ: 75, // 颈部连接点 Z 坐标（头部后端附近）
  neckBaseWidthX: 50, // 颈部底座 X 方向宽度（长方形）
  neckBaseWidthZ: 60, // 颈部底座 Z 方向宽度（长方形）
  neckBaseHeight: 8, // 颈部底座 Y 方向高度
  neckBaseBoltSize: 'M3', // 底座螺孔规格
  neckBaseBoltPitchX: 30, // 螺孔 X 方向间距
  neckBaseBoltPitchZ: 40, // 螺孔 Z 方向间距
  neckLowerRingRadius: 40,
  neckLowerRingHeight: 8,
  neckShaftRadius: 20,
  neckShaftHeight: 20,
  neckUpperRingRadius: 30,
  neckUpperRingHeight: 8,

  // 麦克风阵列 — 6 麦克风 SSL 阵列（声源方向识别）
  // 2 个圆周 × 3 个极点（顶/左/右），避开颈部 Z=45-105 和面部/后端面板
  // Z=30 圆周（接近面部）:  mic_top_front, mic_left_front, mic_right_front
  // Z=120 圆周（接近后端）: mic_top_rear,  mic_left_rear,  mic_right_rear
  // 6 点不共面（X=±90/Y=+90 各占两平面）→ 可定位 3D 声源方向
  micRadius: 5, // 麦克风半径 (直径 10mm)
  micHeight: 3, // 凸出头部外壁高度
  micColor: '#888888', // 金属灰（专业麦克风风格）
  micFrontZ: 30, // 前圆周 Z 位置（接近面部，避开颈部 Z=45）
  micRearZ: 120, // 后圆周 Z 位置（接近后端，避开颈部 Z=105）
};

// ---------------------------------------------------------------------------
// 关键位置计算
//   - 头部圆筒沿 +Z 方向，Z=0 是前端面（面部），Z=110 是后端
//   - 颈部从头部侧面 Y = -headRadius 沿 -Y 方向伸出
//   - sketchOnPlane('XZ', origin).extrude(h) 的 origin 是顶面位置，挤出方向是 -Y
// ---------------------------------------------------------------------------
const HEAD_FRONT_Z = 0; // 头部前端面（贴面部）
const NECK_AXIS_Z = params.neckAttachZ; // 颈部在 Z 方向上的连接点
const NECK_TOP_Y = -params.headRadius; // 颈部上端（紧贴头部侧壁）

// 颈部各段的"顶面 Y"（每段靠近头部的一端）— extrude 沿 -Y 方向挤出
// neckUpperRing 顶面嵌入头部 5mm 形成牢固连接 (Y=-85, 头部内壁 Y=-88, 外壁 Y=-90)
const NECK_UPPER_TOP_Y = NECK_TOP_Y + 5; // -85
const NECK_SHAFT_TOP_Y = NECK_UPPER_TOP_Y - params.neckUpperRingHeight; // -93
const NECK_LOWER_TOP_Y = NECK_SHAFT_TOP_Y - params.neckShaftHeight; // -127
const NECK_BASE_TOP_Y = NECK_LOWER_TOP_Y - params.neckLowerRingHeight; // -135

// 颈部总高 = 8 + 34 + 8 + 10 = 60mm，Y 范围 [-150, -90]

export default () => {
  const parts: Array<{ shape: any; color: string; name: string }> = [];

  // -------------------------------------------------------------------------
  // 1. 头部主体 — 中空圆筒 (#3a3a3a 深灰)
  //    圆筒沿 +Z 方向挤出，从 Z=0（面部）到 Z=110（后端）
  // -------------------------------------------------------------------------
  const headOuter = drawCircle(params.headRadius).sketchOnPlane('XY').extrude(params.headLength);
  const headInner = drawCircle(params.headRadius - params.coverThick)
    .sketchOnPlane('XY')
    .extrude(params.headLength);
  const headBody = headOuter.cut(headInner);
  parts.push({ shape: headBody, color: '#3a3a3a', name: 'head_body' });

  // -------------------------------------------------------------------------
  // 2. 上半圆显示屏（#1a1a1a 哑光黑）— 贴在头部前端面 Z=0
  //    草图技巧：完整圆 cut 下方矩形 = 上半圆
  //    矩形宽 2R 高 R，translate 让其下边正好切到 Y=0
  //    （drawRectangle 默认中心在 (0,0)，所以 translate(0, -R/2) 让矩形中心在 (0, -R/2)）
  //    沿 -Z 方向挤出 1mm（朝外，避免与头部端面重叠）
  // -------------------------------------------------------------------------
  const fullCircle = drawCircle(params.facePanelRadius);
  const lowerRect = drawRectangle(params.facePanelRadius * 2, params.facePanelRadius).translate(
    0,
    -params.facePanelRadius / 2,
  );
  const topHalfDrawing = fullCircle.cut(lowerRect);
  const displayTop = topHalfDrawing
    .sketchOnPlane('XY')
    .extrude(params.panelThick)
    .translate(0, 0, HEAD_FRONT_Z - params.panelThick);
  parts.push({ shape: displayTop, color: '#1a1a1a', name: 'display_top' });

  // -------------------------------------------------------------------------
  // 3. 下半圆显示屏（#1a1a1a 哑光黑，与上半圆同色浑然一体）
  // -------------------------------------------------------------------------
  const upperRect = drawRectangle(params.facePanelRadius * 2, params.facePanelRadius).translate(
    0,
    params.facePanelRadius / 2,
  );
  const bottomHalfDrawing = fullCircle.cut(upperRect);
  const displayBottom = bottomHalfDrawing
    .sketchOnPlane('XY')
    .extrude(params.panelThick)
    .translate(0, 0, HEAD_FRONT_Z - params.panelThick);
  parts.push({ shape: displayBottom, color: '#1a1a1a', name: 'display_bottom' });

  // -------------------------------------------------------------------------
  // 4. 灯带环（#f0f8ff 近透明白）
  //    大圆 cut 小圆得到圆环；Z 偏移 -0.3 让它略微凸出于面部表面
  // -------------------------------------------------------------------------
  const ledOuter = drawCircle(params.ledOuterRadius).sketchOnPlane('XY').extrude(params.ledThick);
  const ledInner = drawCircle(params.ledInnerRadius).sketchOnPlane('XY').extrude(params.ledThick);
  const ledRing = ledOuter.cut(ledInner).translate(0, 0, HEAD_FRONT_Z - params.ledThick - 0.3);
  parts.push({ shape: ledRing, color: '#f0f8ff', name: 'led_ring' });

  // -------------------------------------------------------------------------
  // 5. 传感器横条（#1a1a1a 与显示屏同色）
  //    水平横跨面部中央，Z 偏移让条带凸出于显示屏
  // -------------------------------------------------------------------------
  const sensorBar = drawRectangle(params.sensorWidth, params.sensorHeight)
    .sketchOnPlane('XY')
    .extrude(params.sensorThick)
    .translate(0, 0, HEAD_FRONT_Z - params.sensorThick - 0.5);
  parts.push({ shape: sensorBar, color: '#1a1a1a', name: 'sensor_bar' });

  // -------------------------------------------------------------------------
  // 6. 后端面部（提供 360° FOV）— 镜像复制前端的 4 个面板
  //    头部后端面 Z = headLength，所有面板朝 +Z 方向延伸
  //    （与前端 Z=0 朝 -Z 方向镜像对称）
  // -------------------------------------------------------------------------
  const HEAD_BACK_Z = params.headLength;

  // 后端上半圆显示屏
  const backDisplayTop = topHalfDrawing.sketchOnPlane('XY').extrude(params.panelThick).translate(0, 0, HEAD_BACK_Z);
  parts.push({ shape: backDisplayTop, color: '#1a1a1a', name: 'back_display_top' });

  // 后端下半圆显示屏
  const backDisplayBottom = bottomHalfDrawing
    .sketchOnPlane('XY')
    .extrude(params.panelThick)
    .translate(0, 0, HEAD_BACK_Z);
  parts.push({ shape: backDisplayBottom, color: '#1a1a1a', name: 'back_display_bottom' });

  // 后端灯带环
  const backLedRing = ledOuter.cut(ledInner).translate(0, 0, HEAD_BACK_Z + 0.3);
  parts.push({ shape: backLedRing, color: '#f0f8ff', name: 'back_led_ring' });

  // 后端传感器横条
  const backSensorBar = drawRectangle(params.sensorWidth, params.sensorHeight)
    .sketchOnPlane('XY')
    .extrude(params.sensorThick)
    .translate(0, 0, HEAD_BACK_Z + 0.5);
  parts.push({ shape: backSensorBar, color: '#1a1a1a', name: 'back_sensor_bar' });

  // -------------------------------------------------------------------------
  // 6. 颈部组件 — 从头部侧面 (Y = -headRadius, Z = neckAttachZ) 沿 -Y 方向伸出
  //    用 drawCircle + sketchOnPlane('XZ') + extrude 实现沿 Y 轴的圆柱
  //    origin 设为各段的"顶面 Y"（靠近头部的一端），挤出沿 -Y 方向
  // -------------------------------------------------------------------------
  const neckUpperRing = drawCircle(params.neckUpperRingRadius)
    .sketchOnPlane('XZ', [0, NECK_UPPER_TOP_Y, NECK_AXIS_Z])
    .extrude(params.neckUpperRingHeight);
  parts.push({ shape: neckUpperRing, color: '#3a3a3a', name: 'neck_upper_ring' });

  const neckShaft = drawCircle(params.neckShaftRadius)
    .sketchOnPlane('XZ', [0, NECK_SHAFT_TOP_Y, NECK_AXIS_Z])
    .extrude(params.neckShaftHeight);
  parts.push({ shape: neckShaft, color: '#2a2a2a', name: 'neck_shaft' });

  const neckLowerRing = drawCircle(params.neckLowerRingRadius)
    .sketchOnPlane('XZ', [0, NECK_LOWER_TOP_Y, NECK_AXIS_Z])
    .extrude(params.neckLowerRingHeight);
  parts.push({ shape: neckLowerRing, color: '#3a3a3a', name: 'neck_lower_ring' });

  // -------------------------------------------------------------------------
  // 7. 颈部底座 — 长方形板 (50×60×8mm)，顶面带 4×M3 counterbore 螺孔
  //    用于通过 M3 螺栓固定到身体 (body) 安装板
  //    螺孔 2x2 网格，pitch 30×40mm
  // -------------------------------------------------------------------------
  // 长方形底座：在 XZ 平面画矩形，顶面 Y = NECK_BASE_TOP_Y，沿 -Y 挤出
  const neckBasePlate = drawRectangle(params.neckBaseWidthX, params.neckBaseWidthZ)
    .sketchOnPlane('XZ', [0, NECK_BASE_TOP_Y, NECK_AXIS_Z])
    .extrude(params.neckBaseHeight);

  // 4×M3 counterbore 螺孔，2x2 网格
  // counterbore 默认开口在 Z=0，body 沿 -Z 方向延伸
  // 用 axis:"+Y" 让开口在 Y=0，body 沿 -Y 方向延伸
  // 然后用 patterns.grid 的 origin 把孔放到 (X±pitchX/2, NECK_BASE_TOP_Y, Z±pitchZ/2)
  const neckBaseBoltPlacements = patterns.grid(2, 2, params.neckBaseBoltPitchX, params.neckBaseBoltPitchZ, {
    plane: 'XZ',
    origin: [0, NECK_BASE_TOP_Y, NECK_AXIS_Z],
  });
  const neckBase = patterns.cutAt(
    neckBasePlate,
    () => holes.counterbore(params.neckBaseBoltSize, { plateThickness: params.neckBaseHeight, axis: '+Y' }),
    neckBaseBoltPlacements,
  );
  parts.push({ shape: neckBase, color: '#2a2a2a', name: 'neck_base' });

  // -------------------------------------------------------------------------
  // 8. 麦克风阵列 — 6 个麦克风 SSL 阵列（声源方向识别）
  //    2 个圆周 × 3 个极点（顶/左/右），共 6 个 Ø10×3mm 麦克风
  //    Z=30 圆周（接近面部）: mic_top_front, mic_left_front, mic_right_front
  //    Z=120 圆周（接近后端）: mic_top_rear,  mic_left_rear,  mic_right_rear
  //    6 点不共面 → 360° 声源方向识别 (azimuth + elevation)
  //    全部避开颈部 Z=45-105 范围
  // -------------------------------------------------------------------------
  const micColor = params.micColor;

  // 前圆周（Z=30，接近面部）
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('XZ', [0, params.headRadius, params.micFrontZ])
      .extrude(params.micHeight),
    color: micColor,
    name: 'mic_top_front',
  });
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('YZ', [-params.headRadius, 0, params.micFrontZ])
      .extrude(-params.micHeight),
    color: micColor,
    name: 'mic_left_front',
  });
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('YZ', [params.headRadius, 0, params.micFrontZ])
      .extrude(params.micHeight),
    color: micColor,
    name: 'mic_right_front',
  });

  // 后圆周（Z=120，接近后端）
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('XZ', [0, params.headRadius, params.micRearZ])
      .extrude(params.micHeight),
    color: micColor,
    name: 'mic_top_rear',
  });
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('YZ', [-params.headRadius, 0, params.micRearZ])
      .extrude(-params.micHeight),
    color: micColor,
    name: 'mic_left_rear',
  });
  parts.push({
    shape: drawCircle(params.micRadius)
      .sketchOnPlane('YZ', [params.headRadius, 0, params.micRearZ])
      .extrude(params.micHeight),
    color: micColor,
    name: 'mic_right_rear',
  });

  return parts;
};
