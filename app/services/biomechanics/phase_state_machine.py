# """
# 运动学驱动阶段状态机引擎 v2
# Kinematic-Driven Phase State Machine Engine
# =============================================

# 架构:
#   SignalProcessor    → 滤波 + 求导
#   KinematicDetector  → 零交叉极值检测
#   PhaseStateMachine  → 事件驱动 FSM（只管阶段，不管检测）
#   RepSegmenter       → rep 切分 + 统计

# 核心改进（相比 v1）:
#   1. Bottom/Top 由角速度零交叉决定，不再依赖角度阈值
#   2. FSM 只消费事件，职责单一
#   3. Rep ROM 取区间真实 min/max，不依赖事件时刻
#   4. 滞后带防抖 + 最小持续时间 + 超时保护
#   5. 全面 None-safe
# """

# from __future__ import annotations

# import math
# import re
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass, field
# import numpy as np


# # ================================================================
# #  第一部分：数据模型
# # ================================================================

# @dataclass
# class KinematicEvent:
#     """运动学事件（由信号处理层产生，与 FSM 解耦）"""
#     event_type: str          # "bottom" | "top"
#     frame: int
#     angle: float
#     velocity_at_event: float = 0.0
#     confidence: float = 1.0
#     metadata: Dict[str, Any] = field(default_factory=dict)


# @dataclass
# class FrameState:
#     """逐帧状态快照"""
#     frame_index: int
#     phase: str
#     angle: Optional[float] = None
#     velocity: Optional[float] = None
#     events: List[KinematicEvent] = field(default_factory=list)


# @dataclass
# class PhaseEvent:
#     """rep 内某个阶段的详情"""
#     phase_name: str
#     start_frame: int
#     end_frame: int
#     start_time: float
#     end_time: float
#     duration: float
#     frame_indices: List[int] = field(default_factory=list)
#     metrics: Dict[str, float] = field(default_factory=dict)


# @dataclass
# class RepWithPhases:
#     """一次完整动作"""
#     rep_index: int
#     start_frame: int
#     end_frame: int
#     start_time: float
#     end_time: float
#     duration: float

#     # 事件帧（零交叉时刻）
#     bottom_frame: Optional[int] = None
#     top_frame: Optional[int] = None
#     bottom_angle: Optional[float] = None
#     top_angle: Optional[float] = None

#     # 区间真实极值（统计用，与事件无关）
#     actual_min_angle: Optional[float] = None
#     actual_max_angle: Optional[float] = None
#     actual_rom: Optional[float] = None

#     phases: List[PhaseEvent] = field(default_factory=list)
#     errors_detected: List[Dict] = field(default_factory=list)
#     quality_score: float = 0.0
#     quality_details: Dict[str, float] = field(default_factory=dict)
#     metrics: Dict[str, float] = field(default_factory=dict)

#     def get_phase(self, name: str) -> Optional[PhaseEvent]:
#         return next((p for p in self.phases if p.phase_name == name), None)

#     def get_phase_duration(self, name: str) -> float:
#         p = self.get_phase(name)
#         return p.duration if p else 0.0


# # ================================================================
# #  第二部分：信号处理器
# # ================================================================

# class SignalProcessor:
#     """
#     职责：滤波、求导、滞后带。
#     不涉及任何 FSM 逻辑。
#     """

#     def __init__(
#         self,
#         fps: float = 30.0,
#         sg_window: int = 11,
#         sg_polyorder: int = 3,
#         velocity_smooth_window: int = 7,
#         hysteresis_band: float = 3.0,
#     ):
#         self.fps = fps
#         self.sg_window = sg_window
#         self.sg_polyorder = sg_polyorder
#         self.velocity_smooth_window = velocity_smooth_window
#         self.hysteresis_band = hysteresis_band

#     # ---- Savitzky-Golay 滤波 ----

#     def smooth_angle_series(
#         self, series: List[Optional[float]]
#     ) -> List[Optional[float]]:
#         """
#         对角度序列做 Savitzky-Golay 滤波。
#         None 值先前向+后向填充，滤波后还原 None 位置。
#         """
#         n = len(series)
#         if n < self.sg_window:
#             return list(series)

#         # 记录原始 None 位置
#         none_mask = [v is None for v in series]

#         # 前向填充
#         filled = list(series)
#         last_val = None
#         for i in range(n):
#             if filled[i] is not None:
#                 last_val = filled[i]
#             elif last_val is not None:
#                 filled[i] = last_val

#         # 后向填充
#         next_val = None
#         for i in range(n - 1, -1, -1):
#             if filled[i] is not None:
#                 next_val = filled[i]
#             elif next_val is not None:
#                 filled[i] = next_val

#         # 如果全部 None，直接返回
#         if any(v is None for v in filled):
#             return list(series)

#         # SG 滤波
#         arr = np.array(filled, dtype=np.float64)
#         win = self.sg_window
#         if win % 2 == 0:
#             win += 1
#         if win > n:
#             win = n if n % 2 == 1 else n - 1
#         if win < 3:
#             return list(series)

#         poly = min(self.sg_polyorder, win - 1)
#         smoothed = self._savitzky_golay(arr, win, poly)

#         # 还原 None 位置
#         result = [
#             None if none_mask[i] else float(smoothed[i])
#             for i in range(n)
#         ]
#         return result

#     @staticmethod
#     def _savitzky_golay(y: np.ndarray, window_size: int, order: int) -> np.ndarray:
#         """
#         Savitzky-Golay 滤波器（纯 numpy 实现，无 scipy 依赖）。
#         """
#         half = window_size // 2
#         # 构造 Vandermonde 矩阵
#         order_range = range(order + 1)
#         b = np.array(
#             [[k ** i for i in order_range] for k in range(-half, half + 1)]
#         )
#         # 最小二乘求解
#         m = np.linalg.pinv(b)
#         # 只对中间值做卷积，边界用原始值
#         result = np.copy(y)
#         for i in range(half, len(y) - half):
#             result[i] = np.dot(m[0], y[i - half: i + half + 1])
#         return result

#     # ---- 角速度计算 ----

#     def compute_velocity(
#         self, angles: List[Optional[float]]
#     ) -> List[Optional[float]]:
#         """
#         中心差分求角速度（°/s），再平滑。
#         """
#         n = len(angles)
#         if n < 3:
#             return [None] * n

#         raw_vel = [None] * n
#         for i in range(1, n - 1):
#             a_prev = angles[i - 1]
#             a_next = angles[i + 1]
#             if a_prev is not None and a_next is not None:
#                 dt = 2.0 / self.fps  # 2 帧的时间间隔
#                 raw_vel[i] = (a_next - a_prev) / dt

#         # 首尾用前向/后向差分
#         if angles[0] is not None and angles[1] is not None:
#             raw_vel[0] = (angles[1] - angles[0]) * self.fps
#         if angles[-1] is not None and angles[-2] is not None:
#             raw_vel[-1] = (angles[-1] - angles[-2]) * self.fps

#         # 平滑速度
#         smoothed = self._smooth_velocity(raw_vel, self.velocity_smooth_window)
#         return smoothed

#     @staticmethod
#     def _smooth_velocity(
#         series: List[Optional[float]], window: int
#     ) -> List[Optional[float]]:
#         """对速度序列做简单移动平均，跳过 None。"""
#         n = len(series)
#         if window < 2 or n < window:
#             return list(series)

#         half = window // 2
#         result = list(series)
#         for i in range(n):
#             s, e = max(0, i - half), min(n, i + half + 1)
#             vals = [series[j] for j in range(s, e) if series[j] is not None]
#             if vals:
#                 result[i] = sum(vals) / len(vals)
#         return result

#     # ---- 滞后带速度方向 ----

#     def compute_velocity_sign(
#         self, velocities: List[Optional[float]]
#     ) -> List[Optional[int]]:
#         """
#         带滞后带的速度方向：
#           +1  上升（velocity > +hysteresis_band）
#           -1  下降（velocity < -hysteresis_band）
#            0  过渡区（绝对值 < band，保持上一方向）
#           None  无数据
#         """
#         signs: List[Optional[int]] = []
#         last_confirmed: Optional[int] = None

#         for v in velocities:
#             if v is None:
#                 signs.append(None)
#                 continue

#             if v > self.hysteresis_band:
#                 last_confirmed = 1
#                 signs.append(1)
#             elif v < -self.hysteresis_band:
#                 last_confirmed = -1
#                 signs.append(-1)
#             else:
#                 # 过渡区：保持上一方向
#                 signs.append(last_confirmed if last_confirmed is not None else 0)

#         return signs

#     # ---- 主角度提取 ----

#     @staticmethod
#     def extract_primary_angle(
#         fd: Any,
#         primary_angle: str,
#     ) -> Optional[float]:
#         """从 frame_data 提取主角度值。"""
#         angles = fd.angles if hasattr(fd, "angles") else {}

#         mapping = {
#             "knee_angle_avg": ["left_knee", "right_knee"],
#             "elbow_angle_avg": ["left_elbow", "right_elbow"],
#             "hip_angle": ["left_hip", "right_hip"],
#             "shoulder_flexion_avg": [
#                 "left_shoulder_flexion", "right_shoulder_flexion"
#             ],
#         }

#         keys = mapping.get(primary_angle, [primary_angle])
#         vals = [angles[k] for k in keys if k in angles and angles[k] is not None]
#         if not vals:
#             return None
#         return float(np.mean(vals))


# # ================================================================
# #  第三部分：运动学事件检测器
# # ================================================================

# class KinematicEventDetector:
#     """
#     职责：从角速度序列中检测 bottom/top 事件。
#     核心算法：零交叉检测 + 极值确认 + 最小间距。
#     """

#     def __init__(
#         self,
#         fps: float = 30.0,
#         min_rep_gap_frames: int = 10,
#         bottom_confirm_frames: int = 3,
#         top_confirm_frames: int = 3,
#     ):
#         self.fps = fps
#         self.min_rep_gap_frames = min_rep_gap_frames
#         self.bottom_confirm_frames = bottom_confirm_frames
#         self.top_confirm_frames = top_confirm_frames

#     def detect_events(
#         self,
#         angles: List[Optional[float]],
#         velocities: List[Optional[float]],
#         vel_signs: List[Optional[int]],
#     ) -> List[KinematicEvent]:
#         """
#         扫描全序列，返回所有 bottom/top 事件。

#         Bottom: vel_sign 从 -1 变为 +1（零交叉）
#                  → 在交叉邻域取角度最小值帧作为精确 bottom
#         Top:    vel_sign 从 +1 变为 -1（零交叉）
#                  → 在交叉邻域取角度最大值帧作为精确 top
#         """
#         events: List[KinematicEvent] = []
#         n = len(vel_signs)
#         search_radius = max(3, int(self.fps * 0.15))  # ±150ms

#         # ---- 检测零交叉 ----
#         for i in range(1, n):
#             prev_sign = vel_signs[i - 1]
#             curr_sign = vel_signs[i]

#             if prev_sign is None or curr_sign is None:
#                 continue

#             # Bottom: 从负变正
#             if prev_sign == -1 and curr_sign == 1:
#                 refined = self._refine_extremum(
#                     angles, i, search_radius, "min"
#                 )
#                 if refined is not None:
#                     events.append(KinematicEvent(
#                         event_type="bottom",
#                         frame=refined[0],
#                         angle=refined[1],
#                         velocity_at_event=(
#                             velocities[refined[0]]
#                             if refined[0] < len(velocities)
#                             and velocities[refined[0]] is not None
#                             else 0.0
#                         ),
#                         metadata={"raw_crossing_frame": i},
#                     ))

#             # Top: 从正变负
#             elif prev_sign == 1 and curr_sign == -1:
#                 refined = self._refine_extremum(
#                     angles, i, search_radius, "max"
#                 )
#                 if refined is not None:
#                     events.append(KinematicEvent(
#                         event_type="top",
#                         frame=refined[0],
#                         angle=refined[1],
#                         velocity_at_event=(
#                             velocities[refined[0]]
#                             if refined[0] < len(velocities)
#                             and velocities[refined[0]] is not None
#                             else 0.0
#                         ),
#                         metadata={"raw_crossing_frame": i},
#                     ))

#         # ---- 最小间距过滤（去抖）----
#         events = self._apply_min_gap(events)

#         return events

#     @staticmethod
#     def _refine_extremum(
#         angles: List[Optional[float]],
#         crossing_frame: int,
#         radius: int,
#         mode: str,
#     ) -> Optional[Tuple[int, float]]:
#         """
#         在 zero-crossing 邻域内找精确极值帧。
#         mode="min" → bottom, mode="max" → top
#         """
#         n = len(angles)
#         s = max(0, crossing_frame - radius)
#         e = min(n, crossing_frame + radius + 1)

#         best_frame: Optional[int] = None
#         best_val: Optional[float] = None

#         for j in range(s, e):
#             a = angles[j]
#             if a is None:
#                 continue
#             if best_val is None:
#                 best_frame = j
#                 best_val = a
#             elif mode == "min" and a < best_val:
#                 best_frame = j
#                 best_val = a
#             elif mode == "max" and a > best_val:
#                 best_frame = j
#                 best_val = a

#         if best_frame is not None and best_val is not None:
#             return (best_frame, best_val)
#         return None

#     def _apply_min_gap(
#         self, events: List[KinematicEvent]
#     ) -> List[KinematicEvent]:
#         """
#         同类型事件之间必须间隔 >= min_rep_gap_frames。
#         保留角度更极端的那个。
#         """
#         if len(events) <= 1:
#             return events

#         # 按类型分组过滤
#         bottoms = [e for e in events if e.event_type == "bottom"]
#         tops = [e for e in events if e.event_type == "top"]

#         bottoms = self._filter_same_type_gap(bottoms, "min")
#         tops = self._filter_same_type_gap(tops, "max")

#         merged = sorted(bottoms + tops, key=lambda e: e.frame)
#         return merged

#     def _filter_same_type_gap(
#         self, events: List[KinematicEvent], keep_mode: str
#     ) -> List[KinematicEvent]:
#         """
#         同类型事件最小间距过滤。
#         keep_mode="min" → 保留角度更小的（bottom）
#         keep_mode="max" → 保留角度更大的（top）
#         """
#         if len(events) <= 1:
#             return events

#         events = sorted(events, key=lambda e: e.frame)
#         kept: List[KinematicEvent] = [events[0]]

#         for ev in events[1:]:
#             last = kept[-1]
#             if (ev.frame - last.frame) >= self.min_rep_gap_frames:
#                 kept.append(ev)
#             else:
#                 # 太近，保留更极端的那个
#                 if keep_mode == "min" and ev.angle < last.angle:
#                     kept[-1] = ev
#                 elif keep_mode == "max" and ev.angle > last.angle:
#                     kept[-1] = ev
#                 # 否则丢弃当前

#         return kept


# # ================================================================
# #  第四部分：事件驱动 FSM
# # ================================================================

# class PhaseStateMachine:
#     """
#     纯事件驱动 FSM。
#     只消费 KinematicEvent，不自己做信号检测。

#     状态: setup → eccentric → concentric → lockout → rest → ...
#     事件: bottom（触发 eccentric→concentric）
#            top（触发 concentric→lockout）
#     """

#     # 状态常量
#     SETUP = "setup"
#     ECCENTRIC = "eccentric"
#     CONCENTRIC = "concentric"
#     LOCKOUT = "lockout"
#     REST = "rest"

#     def __init__(
#         self,
#         knowledge: Dict[str, Any],
#         fps: float = 30.0,
#         # 超时保护
#         eccentric_timeout_s: float = 5.0,
#         concentric_timeout_s: float = 5.0,
#         lockout_timeout_s: float = 3.0,
#         # 最小持续时间（防止误触）
#         min_eccentric_duration_s: float = 0.25,
#         min_concentric_duration_s: float = 0.30,
#         # 进入 eccentric 的确认
#         eccentric_entry_min_velocity: float = 10.0,
#         eccentric_entry_min_frames: int = 4,
#     ):
#         self.knowledge = knowledge
#         self.fps = fps

#         # 超时（帧数）
#         self.eccentric_timeout = int(eccentric_timeout_s * fps)
#         self.concentric_timeout = int(concentric_timeout_s * fps)
#         self.lockout_timeout = int(lockout_timeout_s * fps)

#         # 最小持续时间（帧数）
#         self.min_eccentric_frames = int(min_eccentric_duration_s * fps)
#         self.min_concentric_frames = int(min_concentric_duration_s * fps)

#         # eccentric 进入条件
#         self.eccentric_entry_min_vel = eccentric_entry_min_velocity
#         self.eccentric_entry_min_frames = eccentric_entry_min_frames

#         # rep_detection 配置
#         rep_cfg = knowledge.get("rep_detection", {})
#         self.primary_angle: str = rep_cfg.get("primary_angle", "knee_angle_avg")
#         self.min_rom: float = rep_cfg.get("min_rom", 20.0)
#         self.min_duration: float = rep_cfg.get("min_duration", 1.0)
#         self.max_duration: float = rep_cfg.get("max_duration", 12.0)

#     # ================================================================
#     #  主入口：逐帧运行 FSM
#     # ================================================================

#     def run(
#         self,
#         frame_data_list: List[Any],
#         angles: List[Optional[float]],
#         velocities: List[Optional[float]],
#         events: List[KinematicEvent],
#     ) -> List[FrameState]:
#         """
#         输入：逐帧数据 + 预计算的角度/速度 + 预检测的事件
#         输出：List[FrameState]（每帧含状态 + 事件）
#         """
#         n = len(frame_data_list)

#         # 构建事件查找表：frame_index → List[KinematicEvent]
#         event_map: Dict[int, List[KinematicEvent]] = {}
#         for ev in events:
#             event_map.setdefault(ev.frame, []).append(ev)

#         # FSM 状态
#         current_state = self.SETUP
#         frame_states: List[FrameState] = []

#         # 追踪变量
#         phase_enter_frame: int = 0        # 当前阶段的进入帧
#         last_rep_end_frame: int = -999    # 上次 rep 结束帧（冷却用）
#         MIN_REP_GAP = int(self.fps * 0.8) # rep 间最小间隔

#         # eccentric 进入确认（需要连续 N 帧负速度）
#         ecc_confirm_count = 0

#         print(f"[FSM] states: setup→eccentric→concentric→lockout→rest")
#         print(f"[FSM] min_rom={self.min_rom}°  "
#               f"dur=[{self.min_duration}, {self.max_duration}]s")
#         print(f"[FSM] eccentric_timeout={self.eccentric_timeout}f  "
#               f"concentric_timeout={self.concentric_timeout}f")
#         print(f"[FSM] min_ecc_frames={self.min_eccentric_frames}  "
#               f"min_con_frames={self.min_concentric_frames}")
#         print(f"[FSM] events: {len(events)} total "
#               f"({sum(1 for e in events if e.event_type == 'bottom')} bottoms, "
#               f"{sum(1 for e in events if e.event_type == 'top')} tops)")

#         for i in range(n):
#             cur_angle = angles[i] if i < len(angles) else None
#             cur_vel = velocities[i] if i < len(velocities) else None
#             frame_events = event_map.get(i, [])

#             next_state = current_state

#             # ---- 状态转换逻辑 ----

#             if current_state == self.SETUP:
#                 next_state = self._transition_from_setup(
#                     i, cur_angle, cur_vel, frame_events,
#                     velocities, ecc_confirm_count,
#                 )
#                 if next_state == self.ECCENTRIC:
#                     phase_enter_frame = i
#                     ecc_confirm_count = 0
#                     print(f"[FSM] i={i:4d} setup→eccentric")

#             elif current_state == self.ECCENTRIC:
#                 next_state = self._transition_from_eccentric(
#                     i, phase_enter_frame, frame_events,
#                 )
#                 if next_state == self.CONCENTRIC:
#                     phase_enter_frame = i
#                     print(f"[FSM] i={i:4d} eccentric→concentric "
#                           f"(BOTTOM frame={frame_events[0].frame}, "
#                           f"angle={frame_events[0].angle:.1f}°)")
#                 elif next_state == self.REST:
#                     phase_enter_frame = i
#                     last_rep_end_frame = i
#                     print(f"[FSM] i={i:4d} eccentric→rest (timeout)")

#                 # eccentric 进入确认计数
#                 if next_state == self.ECCENTRIC:
#                     if cur_vel is not None and cur_vel < -self.eccentric_entry_min_vel:
#                         ecc_confirm_count += 1
#                     else:
#                         ecc_confirm_count = max(0, ecc_confirm_count - 1)

#             elif current_state == self.CONCENTRIC:
#                 next_state = self._transition_from_concentric(
#                     i, phase_enter_frame, frame_events,
#                 )
#                 if next_state == self.LOCKOUT:
#                     phase_enter_frame = i
#                     print(f"[FSM] i={i:4d} concentric→lockout "
#                           f"(TOP frame={frame_events[0].frame}, "
#                           f"angle={frame_events[0].angle:.1f}°)")
#                 elif next_state == self.REST:
#                     phase_enter_frame = i
#                     last_rep_end_frame = i
#                     print(f"[FSM] i={i:4d} concentric→rest (timeout)")

#             elif current_state == self.LOCKOUT:
#                 next_state = self._transition_from_lockout(
#                     i, phase_enter_frame, last_rep_end_frame,
#                     cur_vel, MIN_REP_GAP,
#                 )
#                 if next_state == self.REST:
#                     phase_enter_frame = i
#                     last_rep_end_frame = i
#                     print(f"[FSM] i={i:4d} lockout→rest")
#                 elif next_state == self.SETUP:
#                     phase_enter_frame = i
#                     print(f"[FSM] i={i:4d} lockout→setup")

#             elif current_state == self.REST:
#                 next_state = self._transition_from_rest(
#                     i, last_rep_end_frame, cur_vel,
#                     velocities, MIN_REP_GAP,
#                     ecc_confirm_count,
#                 )
#                 if next_state == self.ECCENTRIC:
#                     phase_enter_frame = i
#                     ecc_confirm_count = 0
#                     print(f"[FSM] i={i:4d} rest→eccentric")
#                 elif next_state == self.SETUP:
#                     phase_enter_frame = i

#             # ---- 记录帧状态 ----
#             frame_states.append(FrameState(
#                 frame_index=i,
#                 phase=current_state,
#                 angle=cur_angle,
#                 velocity=cur_vel,
#                 events=frame_events,
#             ))

#             current_state = next_state

#         return frame_states

#     # ================================================================
#     #  各状态的转换逻辑
#     # ================================================================

#     def _transition_from_setup(
#         self,
#         i: int,
#         cur_angle: Optional[float],
#         cur_vel: Optional[float],
#         frame_events: List[KinematicEvent],
#         velocities: List[Optional[float]],
#         ecc_confirm_count: int,
#     ) -> str:
#         """setup → eccentric：需要连续负速度确认"""
#         # 检查是否有 bottom 事件已经在之前发生（快速启动场景）
#         # 如果有 bottom 事件，说明已经错过了 eccentric，直接跳到 concentric
#         for ev in frame_events:
#             if ev.event_type == "bottom":
#                 return self.CONCENTRIC

#         # 正常进入：连续负速度
#         if cur_vel is not None and cur_vel < -self.eccentric_entry_min_vel:
#             # 向前回溯确认已经连续下降
#             consecutive_neg = 0
#             for j in range(i, max(-1, i - self.eccentric_entry_min_frames), -1):
#                 if j < 0:
#                     break
#                 v = velocities[j] if j < len(velocities) else None
#                 if v is not None and v < -self.eccentric_entry_min_vel:
#                     consecutive_neg += 1
#                 else:
#                     break
#             if consecutive_neg >= self.eccentric_entry_min_frames:
#                 return self.ECCENTRIC

#         return self.SETUP

#     def _transition_from_eccentric(
#         self,
#         i: int,
#         phase_enter_frame: int,
#         frame_events: List[KinematicEvent],
#     ) -> str:
#         """eccentric → concentric（bottom 事件）或 rest（超时）"""
#         # 超时保护
#         if (i - phase_enter_frame) >= self.eccentric_timeout:
#             return self.REST

#         # 检查 bottom 事件
#         for ev in frame_events:
#             if ev.event_type == "bottom":
#                 return self.CONCENTRIC

#         return self.ECCENTRIC

#     def _transition_from_concentric(
#         self,
#         i: int,
#         phase_enter_frame: int,
#         frame_events: List[KinematicEvent],
#     ) -> str:
#         """concentric → lockout（top 事件）或 rest（超时）"""
#         # 超时保护
#         if (i - phase_enter_frame) >= self.concentric_timeout:
#             return self.REST

#         # 最小持续时间保护
#         if (i - phase_enter_frame) < self.min_concentric_frames:
#             return self.CONCENTRIC

#         # 检查 top 事件
#         for ev in frame_events:
#             if ev.event_type == "top":
#                 return self.LOCKOUT

#         return self.CONCENTRIC

#     def _transition_from_lockout(
#         self,
#         i: int,
#         phase_enter_frame: int,
#         last_rep_end_frame: int,
#         cur_vel: Optional[float],
#         min_gap: int,
#     ) -> str:
#         """lockout → rest（超时或稳定后）"""
#         elapsed = i - phase_enter_frame

#         # 超时 → rest
#         if elapsed >= self.lockout_timeout:
#             return self.REST

#         # 至少停留 0.3s 后才允许退出
#         if elapsed < int(self.fps * 0.3):
#             return self.LOCKOUT

#         # 如果已经开始下降（新 rep），进入 rest
#         if cur_vel is not None and cur_vel < -self.eccentric_entry_min_vel:
#             return self.REST

#         return self.LOCKOUT

#     def _transition_from_rest(
#         self,
#         i: int,
#         last_rep_end_frame: int,
#         cur_vel: Optional[float],
#         velocities: List[Optional[float]],
#         min_gap: int,
#         ecc_confirm_count: int,
#     ) -> str:
#         """rest → eccentric（冷却后 + 连续负速度确认）"""
#         # 冷却期
#         if (i - last_rep_end_frame) < min_gap:
#             return self.REST

#         # 连续负速度确认
#         if cur_vel is not None and cur_vel < -self.eccentric_entry_min_vel:
#             consecutive_neg = 0
#             for j in range(i, max(-1, i - self.eccentric_entry_min_frames), -1):
#                 if j < 0:
#                     break
#                 v = velocities[j] if j < len(velocities) else None
#                 if v is not None and v < -self.eccentric_entry_min_vel:
#                     consecutive_neg += 1
#                 else:
#                     break
#             if consecutive_neg >= self.eccentric_entry_min_frames:
#                 return self.ECCENTRIC

#         return self.REST


# # ================================================================
# #  第五部分：Rep 切分器
# # ================================================================

# class RepSegmenter:
#     """
#     职责：从 FrameState 列表中切分出 rep，计算统计指标。
#     """

#     def __init__(
#         self,
#         knowledge: Dict[str, Any],
#         fps: float = 30.0,
#     ):
#         self.fps = fps
#         rep_cfg = knowledge.get("rep_detection", {})
#         self.min_rom: float = rep_cfg.get("min_rom", 20.0)
#         self.min_duration: float = rep_cfg.get("min_duration", 1.0)
#         self.max_duration: float = rep_cfg.get("max_duration", 12.0)

#     def segment(
#         self,
#         frame_states: List[FrameState],
#         frame_data_list: List[Any],
#     ) -> List[RepWithPhases]:
#         reps: List[RepWithPhases] = []

#         # ---- Step 1: 找 rep 边界 ----
#         # rep 定义：从 eccentric 开始到下一个 eccentric 之前（或 rest）
#         rep_candidates = self._find_rep_candidates(frame_states)

#         # ---- Step 2: 逐个验证 ----
#         for idx, (start_i, end_i) in enumerate(rep_candidates):
#             rfs = frame_states[start_i: end_i + 1]
#             rep = self._build_rep(rfs, frame_data_list, idx + 1)
#             if rep is not None:
#                 reps.append(rep)
#                 # 重新编号
#                 rep.rep_index = len(reps)

#         # ---- Step 3: 阶段详情 ----
#         for rep in reps:
#             rep_fs = [
#                 fs for fs in frame_states
#                 if rep.start_frame <= fs.frame_index <= rep.end_frame
#             ]
#             rep.phases = self._extract_phases(rep, frame_data_list, rep_fs)
#             rep.metrics = self._compute_metrics(rep)

#         return reps

#     def _find_rep_candidates(
#         self, frame_states: List[FrameState]
#     ) -> List[Tuple[int, int]]:
#         """
#         找 rep 候选区间。
#         策略：每个 bottom 事件标记一个 rep 的起点，
#               下一个 bottom 事件（或序列结束）标记终点。
#         """
#         # 收集所有 bottom 事件帧
#         bottom_frames: List[int] = []
#         for fs in frame_states:
#             for ev in fs.events:
#                 if ev.event_type == "bottom":
#                     bottom_frames.append(ev.frame)

#         if not bottom_frames:
#             # 没有 bottom 事件，尝试用 eccentric 阶段
#             return self._find_rep_candidates_fallback(frame_states)

#         candidates: List[Tuple[int, int]] = []

#         for k, bf in enumerate(bottom_frames):
#             # rep 起点：bottom 事件帧向前延伸到 eccentric 开始
#             start_i = self._find_eccentric_start(frame_states, bf)

#             # rep 终点：下一个 bottom 之前，或序列结束
#             if k + 1 < len(bottom_frames):
#                 next_bf = bottom_frames[k + 1]
#                 end_i = self._find_rep_end_before(frame_states, next_bf)
#             else:
#                 end_i = len(frame_states) - 1

#             if start_i <= end_i:
#                 candidates.append((start_i, end_i))

#         return candidates

#     def _find_rep_candidates_fallback(
#         self, frame_states: List[FrameState]
#     ) -> List[Tuple[int, int]]:
#         """无 bottom 事件时的回退策略：用 eccentric→concentric 序列"""
#         candidates = []
#         start = None

#         for fs in frame_states:
#             if fs.phase == "eccentric" and start is None:
#                 start = fs.frame_index
#             elif fs.phase == "rest" and start is not None:
#                 candidates.append((start, fs.frame_index))
#                 start = None

#         if start is not None:
#             candidates.append((start, len(frame_states) - 1))

#         return candidates

#     @staticmethod
#     def _find_eccentric_start(
#         frame_states: List[FrameState], bottom_frame: int
#     ) -> int:
#         """从 bottom 帧向前找 eccentric 阶段的起始帧。"""
#         idx = bottom_frame
#         while idx > 0:
#             if idx < len(frame_states) and frame_states[idx].phase == "eccentric":
#                 idx -= 1
#             else:
#                 break
#         # 回到第一个 eccentric 帧
#         idx = max(0, idx)
#         if idx < len(frame_states) and frame_states[idx].phase != "eccentric":
#             idx += 1
#         return min(idx, bottom_frame)

#     @staticmethod
#     def _find_rep_end_before(
#         frame_states: List[FrameState], next_bottom_frame: int
#     ) -> int:
#         """在下一个 bottom 之前找当前 rep 的结束帧。"""
#         # 结束帧 = 下一个 bottom 之前的最后一个非 eccentric 帧
#         # 或者下一个 rep 的 eccentric 开始之前
#         idx = next_bottom_frame - 1
#         while idx > 0:
#             if idx < len(frame_states):
#                 phase = frame_states[idx].phase
#                 if phase in ("rest", "setup"):
#                     idx -= 1
#                     continue
#                 # 如果还是 eccentric，说明属于下一个 rep
#                 if phase == "eccentric":
#                     idx -= 1
#                     continue
#                 break
#             idx -= 1
#         return max(0, idx)

#     def _build_rep(
#         self,
#         rfs: List[FrameState],
#         frame_data_list: List[Any],
#         rep_index: int,
#     ) -> Optional[RepWithPhases]:
#         """从帧列表构建 RepWithPhases，含完整校验。"""
#         if not rfs:
#             return None

#         sf = rfs[0].frame_index
#         ef = rfs[-1].frame_index

#         # 时间戳
#         st = self._get_timestamp(frame_data_list, sf)
#         et = self._get_timestamp(frame_data_list, ef)
#         dur = et - st

#         # ---- 从事件中提取 bottom/top ----
#         bf, ba = self._find_event(rfs, "bottom")
#         tf, ta = self._find_event(rfs, "top")

#         # ---- 区间真实极值（统计用）----
#         valid_angles = [fs.angle for fs in rfs if fs.angle is not None]
#         actual_min = min(valid_angles) if valid_angles else None
#         actual_max = max(valid_angles) if valid_angles else None
#         actual_rom = (actual_max - actual_min) if (
#             actual_min is not None and actual_max is not None
#         ) else None

#         # ---- 阶段完整性 ----
#         unique_phases = list(dict.fromkeys(fs.phase for fs in rfs))
#         has_ecc = "eccentric" in unique_phases
#         has_con = "concentric" in unique_phases
#         has_complete = has_ecc and has_con

#         # ---- 事件完整性 ----
#         events_ok = (
#             bf is not None and ba is not None
#             and tf is not None and ta is not None
#         )

#         # ---- ROM 校验（用区间真实 ROM）----
#         rom_ok = (actual_rom is not None and actual_rom >= self.min_rom)

#         # ---- 时长校验 ----
#         dur_ok = self.min_duration <= dur <= self.max_duration

#         # ---- 综合判定 ----
#         accepted = dur_ok and rom_ok and has_complete and events_ok

#         if not accepted:
#             reasons = []
#             if not dur_ok:
#                 reasons.append(f"dur={dur:.2f}s")
#             if not rom_ok:
#                 reasons.append(f"rom={actual_rom}")
#             if not has_complete:
#                 reasons.append(f"phases={unique_phases}")
#             if not events_ok:
#                 reasons.append(
#                     f"events: b={'✓' if bf else '✗'} t={'✓' if tf else '✗'}"
#                 )
#             print(f"[REP] ❌ #{rep_index} rejected | {' | '.join(reasons)}")
#             return None

#         rep = RepWithPhases(
#             rep_index=rep_index,
#             start_frame=sf,
#             end_frame=ef,
#             start_time=st,
#             end_time=et,
#             duration=dur,
#             bottom_frame=bf,
#             bottom_angle=ba,
#             top_frame=tf,
#             top_angle=ta,
#             actual_min_angle=actual_min,
#             actual_max_angle=actual_max,
#             actual_rom=actual_rom,
#         )

#         ba_s = f"{ba:.1f}" if ba is not None else "N/A"
#         ta_s = f"{ta:.1f}" if ta is not None else "N/A"
#         rom_s = f"{actual_rom:.1f}" if actual_rom is not None else "N/A"
#         print(
#             f"[REP] ✅ #{rep_index} | {dur:.2f}s | "
#             f"bot_f={bf}({ba_s}°) top_f={tf}({ta_s}°) | "
#             f"ROM={rom_s}° (min={actual_min:.1f}° max={actual_max:.1f}°)"
#         )

#         return rep

#     @staticmethod
#     def _find_event(
#         rfs: List[FrameState], etype: str
#     ) -> Tuple[Optional[int], Optional[float]]:
#         """从帧列表中查找指定类型的事件（取第一个）。"""
#         for fs in rfs:
#             for ev in fs.events:
#                 if ev.event_type == etype:
#                     return ev.frame, ev.angle
#         return None, None

#     def _get_timestamp(
#         self, frame_data_list: List[Any], frame_idx: int
#     ) -> float:
#         if frame_idx < len(frame_data_list):
#             fd = frame_data_list[frame_idx]
#             if hasattr(fd, "timestamp_sec"):
#                 return fd.timestamp_sec
#         return frame_idx / self.fps

#     def _extract_phases(
#         self,
#         rep: RepWithPhases,
#         frame_data_list: List[Any],
#         rfs: List[FrameState],
#     ) -> List[PhaseEvent]:
#         """提取 rep 内各阶段详情。"""
#         # 按阶段分组（保持顺序）
#         groups: Dict[str, List[FrameState]] = {}
#         for fs in rfs:
#             groups.setdefault(fs.phase, []).append(fs)

#         phases: List[PhaseEvent] = []
#         for phase_name, fs_list in groups.items():
#             sf = fs_list[0].frame_index
#             ef = fs_list[-1].frame_index
#             st = self._get_timestamp(frame_data_list, sf)
#             et = self._get_timestamp(frame_data_list, ef)

#             phases.append(PhaseEvent(
#                 phase_name=phase_name,
#                 start_frame=sf,
#                 end_frame=ef,
#                 start_time=st,
#                 end_time=et,
#                 duration=max(0.0, et - st),
#                 frame_indices=[fs.frame_index for fs in fs_list],
#             ))

#         return phases

#     def _compute_metrics(self, rep: RepWithPhases) -> Dict[str, float]:
#         m: Dict[str, float] = {}

#         # 时长
#         m["total_duration"] = rep.duration
#         for p in rep.phases:
#             m[f"{p.phase_name}_duration"] = p.duration

#         # 事件帧
#         if rep.bottom_frame is not None:
#             m["bottom_frame"] = rep.bottom_frame
#         if rep.bottom_angle is not None:
#             m["bottom_angle"] = rep.bottom_angle
#         if rep.top_frame is not None:
#             m["top_frame"] = rep.top_frame
#         if rep.top_angle is not None:
#             m["top_angle"] = rep.top_angle

#         # 区间真实极值
#         if rep.actual_min_angle is not None:
#             m["actual_min_angle"] = rep.actual_min_angle
#         if rep.actual_max_angle is not None:
#             m["actual_max_angle"] = rep.actual_max_angle
#         if rep.actual_rom is not None:
#             m["rom"] = rep.actual_rom

#         # 离心/向心比率
#         ed = rep.get_phase_duration("eccentric")
#         cd = rep.get_phase_duration("concentric")
#         if ed > 0 and cd > 0:
#             m["eccentric_concentric_ratio"] = ed / cd

#         # 平均向心速度
#         if rep.actual_rom is not None and cd > 0:
#             m["avg_concentric_velocity"] = rep.actual_rom / cd

#         return m


# # ================================================================
# #  第六部分：主引擎（门面）
# # ================================================================

# class KinematicPhaseEngine:
#     """
#     主引擎：串联所有组件。

#     用法：
#         engine = KinematicPhaseEngine(knowledge, fps=30)
#         reps = engine.process(frame_data_list)
#     """

#     def __init__(
#         self,
#         knowledge: Dict[str, Any],
#         fps: float = 30.0,
#         # 信号处理参数
#         sg_window: int = 11,
#         sg_polyorder: int = 3,
#         velocity_smooth_window: int = 7,
#         hysteresis_band: float = 3.0,
#         # 事件检测参数
#         min_rep_gap_frames: int = 10,
#         # FSM 参数
#         eccentric_timeout_s: float = 5.0,
#         concentric_timeout_s: float = 5.0,
#         lockout_timeout_s: float = 3.0,
#         min_eccentric_duration_s: float = 0.25,
#         min_concentric_duration_s: float = 0.30,
#         eccentric_entry_min_velocity: float = 10.0,
#         eccentric_entry_min_frames: int = 4,
#     ):
#         self.knowledge = knowledge
#         self.fps = fps

#         rep_cfg = knowledge.get("rep_detection", {})
#         self.primary_angle: str = rep_cfg.get("primary_angle", "knee_angle_avg")

#         # 初始化子组件
#         self.signal = SignalProcessor(
#             fps=fps,
#             sg_window=sg_window,
#             sg_polyorder=sg_polyorder,
#             velocity_smooth_window=velocity_smooth_window,
#             hysteresis_band=hysteresis_band,
#         )

#         self.detector = KinematicEventDetector(
#             fps=fps,
#             min_rep_gap_frames=min_rep_gap_frames,
#         )

#         self.fsm = PhaseStateMachine(
#             knowledge=knowledge,
#             fps=fps,
#             eccentric_timeout_s=eccentric_timeout_s,
#             concentric_timeout_s=concentric_timeout_s,
#             lockout_timeout_s=lockout_timeout_s,
#             min_eccentric_duration_s=min_eccentric_duration_s,
#             min_concentric_duration_s=min_concentric_duration_s,
#             eccentric_entry_min_velocity=eccentric_entry_min_velocity,
#             eccentric_entry_min_frames=eccentric_entry_min_frames,
#         )

#         self.segmenter = RepSegmenter(
#             knowledge=knowledge,
#             fps=fps,
#         )

#     def process(self, frame_data_list: List[Any]) -> List[RepWithPhases]:
#         """主入口。"""
#         if not frame_data_list or len(frame_data_list) < 5:
#             print("[ENGINE] ⚠️ 帧数据不足，跳过")
#             return []

#         n = len(frame_data_list)
#         print(f"[ENGINE] {'='*60}")
#         print(f"[ENGINE] Processing {n} frames @ {self.fps}fps "
#               f"({n / self.fps:.1f}s)")
#         print(f"[ENGINE] primary_angle={self.primary_angle}")
#         print(f"[ENGINE] {'='*60}")

#         # ---- Step 1: 提取原始角度序列 ----
#         raw_angles: List[Optional[float]] = [
#             SignalProcessor.extract_primary_angle(fd, self.primary_angle)
#             for fd in frame_data_list
#         ]

#         valid_count = sum(1 for a in raw_angles if a is not None)
#         print(f"[ENGINE] Step 1: 提取角度  "
#               f"valid={valid_count}/{n} "
#               f"({valid_count/n*100:.1f}%)")

#         if valid_count < 5:
#             print("[ENGINE] ⚠️ 有效角度不足，跳过")
#             return []

#         # ---- Step 2: Savitzky-Golay 滤波 ----
#         smoothed_angles = self.signal.smooth_angle_series(raw_angles)

#         # ---- Step 3: 计算角速度 ----
#         velocities = self.signal.compute_velocity(smoothed_angles)

#         # ---- Step 4: 计算带滞后带的速度方向 ----
#         vel_signs = self.signal.compute_velocity_sign(velocities)

#         # ---- Step 5: 运动学事件检测 ----
#         events = self.detector.detect_events(
#             smoothed_angles, velocities, vel_signs
#         )

#         bottom_events = [e for e in events if e.event_type == "bottom"]
#         top_events = [e for e in events if e.event_type == "top"]
#         print(f"[ENGINE] Step 5: 事件检测  "
#               f"bottoms={len(bottom_events)}  tops={len(top_events)}")

#         for ev in events:
#             print(f"  [{ev.event_type.upper():6s}] frame={ev.frame:4d}  "
#                   f"angle={ev.angle:.1f}°  vel={ev.velocity_at_event:.1f}°/s")

#         # ---- Step 6: FSM 运行 ----
#         frame_states = self.fsm.run(
#             frame_data_list, smoothed_angles, velocities, events
#         )

#         # ---- Step 7: Rep 切分 ----
#         reps = self.segmenter.segment(frame_states, frame_data_list)

#         print(f"[ENGINE] {'='*60}")
#         print(f"[ENGINE] ✅ 完成：{len(reps)} 个有效 rep")
#         for rep in reps:
#             ba_s = f"{rep.bottom_angle:.1f}" if rep.bottom_angle else "N/A"
#             ta_s = f"{rep.top_angle:.1f}" if rep.top_angle else "N/A"
#             rom_s = f"{rep.actual_rom:.1f}" if rep.actual_rom else "N/A"
#             print(
#                 f"  Rep#{rep.rep_index}: {rep.duration:.2f}s | "
#                 f"bot={ba_s}° top={ta_s}° ROM={rom_s}°"
#             )
#         print(f"[ENGINE] {'='*60}")

#         return reps

#     # ---- 调试工具 ----

#     def debug_dump(
#         self,
#         frame_data_list: List[Any],
#         output_path: Optional[str] = None,
#     ) -> Dict[str, Any]:
#         """
#         输出完整的调试数据（可选保存为 JSON）。
#         """
#         raw_angles = [
#             SignalProcessor.extract_primary_angle(fd, self.primary_angle)
#             for fd in frame_data_list
#         ]
#         smoothed = self.signal.smooth_angle_series(raw_angles)
#         velocities = self.signal.compute_velocity(smoothed)
#         vel_signs = self.signal.compute_velocity_sign(velocities)
#         events = self.detector.detect_events(smoothed, velocities, vel_signs)
#         frame_states = self.fsm.run(
#             frame_data_list, smoothed, velocities, events
#         )

#         dump = {
#             "n_frames": len(frame_data_list),
#             "fps": self.fps,
#             "primary_angle": self.primary_angle,
#             "raw_angles": [round(a, 2) if a is not None else None
#                           for a in raw_angles],
#             "smoothed_angles": [round(a, 2) if a is not None else None
#                                for a in smoothed],
#             "velocities": [round(v, 2) if v is not None else None
#                           for v in velocities],
#             "vel_signs": vel_signs,
#             "events": [
#                 {
#                     "type": e.event_type,
#                     "frame": e.frame,
#                     "angle": round(e.angle, 2),
#                     "velocity": round(e.velocity_at_event, 2),
#                 }
#                 for e in events
#             ],
#             "frame_phases": [fs.phase for fs in frame_states],
#         }

#         if output_path:
#             import json
#             with open(output_path, "w") as f:
#                 json.dump(dump, f, indent=2)
#             print(f"[ENGINE] Debug dump saved to {output_path}")

#         return dump  