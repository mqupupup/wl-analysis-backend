"""
exercise_specific_scorer_v2.py (V3 语义修复版)

修复清单：
  1. 删除错误指标：bar_path(肘角std) / elbow_tuck(肘角) / touch_point(肘角std)
  2. power → concentric_speed（角速度不是功率）
  3. joint_stress → depth_control（肘角≠关节应力，且修复不连续）
  4. symmetry 加 bilateral_valid_ratio 门槛，数据不足=N/A 不扣分
  5. ROM 用连续评分，不再 abs(rom-80) 扣分
  6. velocity_smoothness 用 MAD 替代 std，抗噪
  7. Score / Error / Data Quality 三者分离
  8. 缺数据 => N/A，不参与加权
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import warnings

from app.domain.models import RepContext
from app.domain.enums import ValidationStatus, MetricStatus, ErrorStatus
from .butt_contact_analyzer import ButtContactAnalyzer
from .error_detection_engine import detect_bounce


# ═══════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════

@dataclass
class MetricResult:
    key: str
    raw: Optional[float] = None
    score: Optional[float] = None
    status: MetricStatus = MetricStatus.VALID
    detail: str = ""


@dataclass
class LayerResult:
    layer_name: str
    score: Optional[float] = None
    status: MetricStatus = MetricStatus.VALID
    metrics: List[MetricResult] = field(default_factory=list)
    detail: str = ""


@dataclass
class ScoreError:
    code: str
    severity: str
    message: str
    deduction: float = 0.0


@dataclass
class RepScoreResult:
    rep_index: int

    layers: Dict[str, LayerResult] = field(default_factory=dict)

    technique_score: Optional[float] = None
    movement_quality: Optional[float] = None
    safety_score: Optional[float] = None
    performance_score: Optional[float] = None

    overall_score: Optional[float] = None
    # 兼容旧前端
    total_score: Optional[float] = None
    grade: str = "N/A"

    errors: List[ScoreError] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)

    data_quality_score: float = 100.0

    status: MetricStatus = MetricStatus.VALID


# ═══════════════════════════════════════
#  Weights
# ═══════════════════════════════════════

TECHNIQUE_WEIGHTS: Dict[str, float] = {
    "bar_path": 0.18,
    "rom": 0.18,
    "elbow_tuck": 0.15,
    "tempo": 0.12,
    "bottom_control": 0.12,
    "lockout": 0.12,
    "symmetry": 0.13,
}

MOVEMENT_WEIGHTS: Dict[str, float] = {
    "direction_consistency": 0.50,
    "velocity_smoothness": 0.50,
}

SAFETY_WEIGHTS: Dict[str, float] = {
    "depth_control": 0.25,
    "eccentric_control": 0.25,
    "butt_contact": 0.25,
    "bounce_control": 0.25,
}

PERFORMANCE_WEIGHTS: Dict[str, float] = {
    "concentric_speed": 1.0,
}

LAYER_WEIGHTS: Dict[str, float] = {
    "technique": 0.40,
    "movement_quality": 0.25,
    "safety": 0.20,
    "performance": 0.15,
}


# ═══════════════════════════════════════
#  Generic helpers
# ═══════════════════════════════════════

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def _valid_weighted_average(
    metrics: List[MetricResult],
    weights: Dict[str, float],
) -> LayerResult:
    """只对 VALID 且有 score 的指标做加权平均，无效指标自动剔除。"""
    valid = [
        (m, weights.get(m.key, 0.0))
        for m in metrics
        if m.status == MetricStatus.VALID
        and m.score is not None
        and np.isfinite(m.score)
    ]

    if not valid:
        return LayerResult(
            layer_name="unknown",
            score=None,
            status=MetricStatus.INSUFFICIENT_DATA,
            metrics=metrics,
            detail="没有有效指标",
        )

    total_weight = sum(w for _, w in valid)
    if total_weight <= 0:
        return LayerResult(
            layer_name="unknown",
            score=None,
            status=MetricStatus.INSUFFICIENT_DATA,
            metrics=metrics,
        )

    score = sum(
        metric.score * (weight / total_weight)
        for metric, weight in valid
    )

    return LayerResult(
        layer_name="unknown",
        score=round(score, 1),
        status=MetricStatus.VALID,
        metrics=metrics,
        detail=f"{len(valid)}/{len(metrics)} 指标有效",
    )


# ═══════════════════════════════════════
#  1. ROM — 连续评分，不再 abs(rom-80)
# ═══════════════════════════════════════

def compute_rom(rep: RepContext) -> MetricResult:
    rom = float(rep.actual_rom)

    if not np.isfinite(rom) or rom <= 0:
        return MetricResult("rom", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "ROM 无效")

    # 卧推肘角 ROM：
    #   70~100°  优秀（满分）
    #   60~70°   偏低但可接受
    #   100~110° 偏大但可接受
    #   <60°     幅度不足，线性扣分
    #   >110°    可能检测异常，轻度扣分
    if 70.0 <= rom <= 100.0:
        score = 100.0
    elif 60.0 <= rom < 70.0:
        score = 85.0 + (rom - 60.0) * 1.5   # 60→85, 70→100
    elif 100.0 < rom <= 110.0:
        score = 100.0 - (rom - 100.0) * 1.0  # 100→100, 110→90
    elif rom < 60.0:
        score = max(50.0, 85.0 - (60.0 - rom) * 2.0)
    else:  # > 110
        score = max(60.0, 90.0 - (rom - 110.0) * 1.5)

    return MetricResult("rom", raw=rom, score=_clamp(score))


# ═══════════════════════════════════════
#  2. Tempo — ecc/con 比例
# ═══════════════════════════════════════

def compute_tempo(rep: RepContext) -> MetricResult:
    ecc = float(rep.eccentric_duration)
    con = float(rep.concentric_duration)

    if ecc <= 0 or con <= 0:
        return MetricResult("tempo", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "阶段时长不足")

    ratio = ecc / con

    # 卧推常见目标：eccentric >= concentric
    #   1.2~2.5  理想
    #   1.0~1.2  偏快离心
    #   0.75~1.0 离心明显偏短
    #   2.5~3.0  离心偏长
    #   其他     60 分基线
    if 1.2 <= ratio <= 2.5:
        score = 100.0
    elif 1.0 <= ratio < 1.2:
        score = 90.0
    elif 0.75 <= ratio < 1.0:
        score = 75.0
    elif 2.5 < ratio <= 3.0:
        score = 90.0
    else:
        score = 60.0

    return MetricResult("tempo", raw=ratio, score=score)


# ═══════════════════════════════════════
#  3. Bottom Control — dwell time
# ═══════════════════════════════════════

def compute_bottom_control(rep: RepContext) -> MetricResult:
    """
    V3.1: 底部控制评分，与 Bounce 语义解耦。

    注意：Bounce（砸胸弹震）由错误检测层独立判断，这里只评底部停留控制。
    极短停留 (<0.03s) 可能是 bounce，但不直接等同——需要结合 approach_speed 等信号。

    评分区间：
      <0.03s   极短停留，可能弹胸 → 70 分
      0.03~0.10s  Touch & Go，可接受 → 90 分
      0.10~0.35s  正常控制 → 100 分
      0.35~0.50s  稍长停顿 → 95 分
      >0.50s   过长（可能失败/粘滞）→ 扣分
    """
    dwell = float(rep.bottom_dwell_time)

    if not np.isfinite(dwell):
        return MetricResult("bottom_control", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    if dwell < 0.03:
        score = 70.0
    elif dwell < 0.10:
        score = 90.0
    elif dwell <= 0.35:
        score = 100.0
    elif dwell <= 0.50:
        score = 95.0
    else:
        score = max(60.0, 95.0 - (dwell - 0.50) * 40.0)

    return MetricResult("bottom_control", raw=dwell, score=_clamp(score))


# ═══════════════════════════════════════
#  3b. Elbow Tuck — 真正的 upper_arm_torso_angle
# ═══════════════════════════════════════

def compute_elbow_tuck(rep: RepContext) -> MetricResult:
    """
    V3: 真正的 elbow tuck 检测，使用 upper_arm_torso_angle（上臂与躯干夹角）。
    不是 elbow joint angle！

    评分区间（卧推）：
      45~70°   favorable（自然内收）
      30~45°   偏夹紧但可接受
      70~85°   偏外展但可接受
      85~90°   明显外展
      >90°     T型，高肩负荷
    """
    # 取 bottom 帧附近的 upper_arm_torso_angle
    bf_rel = rep.bottom_frame - rep.start_frame
    window = max(1, int(0.1 * rep.fps))  # bottom 前后 0.1s

    values = []
    for side_arr in [rep.left_upper_arm_torso, rep.right_upper_arm_torso]:
        if side_arr is None:
            continue
        start = max(0, bf_rel - window)
        end = min(len(side_arr), bf_rel + window + 1)
        if end > start:
            seg = np.asarray(side_arr[start:end], dtype=float)
            seg = seg[np.isfinite(seg)]
            if len(seg) > 0:
                values.append(float(np.median(seg)))

    if not values:
        return MetricResult("elbow_tuck", None, None,
                            MetricStatus.INSUFFICIENT_DATA,
                            "缺少 upper_arm_torso 数据")

    angle = float(np.mean(values))

    # 连续评分
    if 45.0 <= angle <= 70.0:
        score = 100.0
    elif 30.0 <= angle < 45.0:
        score = 85.0 + (angle - 30.0) * 1.0  # 30→85, 45→100
    elif 70.0 < angle <= 85.0:
        score = 100.0 - (angle - 70.0) * 1.0  # 70→100, 85→85
    elif 85.0 < angle <= 90.0:
        score = 85.0 - (angle - 85.0) * 4.0  # 85→85, 90→65
    elif angle < 30.0:
        score = max(60.0, 85.0 - (30.0 - angle) * 1.5)
    else:  # > 90
        score = max(40.0, 65.0 - (angle - 90.0) * 2.0)

    return MetricResult("elbow_tuck", raw=angle, score=_clamp(score),
                        detail=f"upper_arm_torso={angle:.1f}°")


# ═══════════════════════════════════════
#  4. Lockout — 顶部角度 + 平台稳定性
# ═══════════════════════════════════════

def compute_lockout(rep: RepContext) -> MetricResult:
    """
    V3: 锁定评分 = 顶部角度 + 顶部平台稳定性，不再用单一阈值。
    """
    top = float(rep.actual_max_angle)

    if not np.isfinite(top) or top <= 0:
        return MetricResult("lockout", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    # 如果有 concentric 段数据，计算顶部平台稳定性
    plateau_bonus = 0.0
    if rep.bilateral_elbow is not None and rep.has_concentric:
        cs = max(0, rep.concentric_start - rep.start_frame)
        ce = min(len(rep.bilateral_elbow), rep.concentric_end - rep.start_frame + 1)
        if ce - cs >= 5:
            con_seg = np.asarray(rep.bilateral_elbow[cs:ce], dtype=float)
            con_seg = con_seg[np.isfinite(con_seg)]
            if len(con_seg) >= 5:
                window_size = max(3, int(len(con_seg) * 0.15))
                top_window = con_seg[-window_size:]
                top_std = float(np.nanstd(top_window))
                near_top = float(np.mean(top_window >= top - 3.0))
                # 稳定平台期加分
                if top_std <= 3.0 and near_top >= 0.6:
                    plateau_bonus = 5.0

    # 基础分：顶部角度
    if top >= 165:
        score = 100.0
    elif top >= 155:
        score = 92.0
    elif top >= 145:
        score = 80.0
    elif top >= 135:
        score = 65.0
    else:
        score = 50.0

    score = min(100.0, score + plateau_bonus)

    return MetricResult("lockout", raw=top, score=score,
                        detail=f"plateau_bonus={plateau_bonus:.0f}")


# ═══════════════════════════════════════
#  4b. Bar Path — 真正的 wrist 轨迹检测
# ═══════════════════════════════════════

def compute_bar_path(rep: RepContext) -> MetricResult:
    """
    V3: 真正的 Bar Path 检测，使用 wrist midpoint 作为代理。
    第一版只评估轨迹平滑度，不评判水平偏移量（缺乏校准数据）。

    数据来源：
      - 双侧 wrist 都有 → midpoint
      - 只有左侧 → left wrist
      - 只有右侧 → right wrist
      - 都没有 → INSUFFICIENT_DATA
    """
    if not rep.has_concentric:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "无 concentric 阶段")

    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(
        rep.concentric_end - rep.start_frame + 1,
        len(rep.left_wrist) if rep.left_wrist is not None else 0,
        len(rep.right_wrist) if rep.right_wrist is not None else 0,
    )

    if ce - cs < 5:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "concentric 帧数不足")

    # 收集可用的 wrist 轨迹
    trajectories = []
    if rep.left_wrist is not None:
        lw = np.asarray(rep.left_wrist[cs:ce], dtype=float)
        if lw.ndim == 2 and lw.shape[1] >= 2:
            trajectories.append(("left", lw))
    if rep.right_wrist is not None:
        rw = np.asarray(rep.right_wrist[cs:ce], dtype=float)
        if rw.ndim == 2 and rw.shape[1] >= 2:
            trajectories.append(("right", rw))

    if not trajectories:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "缺少 wrist 坐标数据")

    # 选择轨迹：双侧都有 → midpoint；否则用可用侧
    if len(trajectories) == 2:
        lw = trajectories[0][1]
        rw = trajectories[1][1]
        valid = np.isfinite(lw[:, 0]) & np.isfinite(rw[:, 0])
        if valid.sum() < 5:
            # 双侧同时有效帧不足，退化为单侧
            for name, traj in trajectories:
                v = np.isfinite(traj[:, 0])
                if v.sum() >= 5:
                    x = traj[v, 0]
                    y = traj[v, 1]
                    break
            else:
                return MetricResult("bar_path", None, None,
                                    MetricStatus.INSUFFICIENT_DATA, "wrist 有效帧不足")
        else:
            x = (lw[valid, 0] + rw[valid, 0]) / 2.0
            y = (lw[valid, 1] + rw[valid, 1]) / 2.0
    else:
        name, traj = trajectories[0]
        v = np.isfinite(traj[:, 0])
        if v.sum() < 5:
            return MetricResult("bar_path", None, None,
                                MetricStatus.INSUFFICIENT_DATA, "wrist 有效帧不足")
        x = traj[v, 0]
        y = traj[v, 1]

    if len(x) < 5:
        return MetricResult("bar_path", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "过滤后有效帧不足")

    # 归一化：以起点为原点
    x0, y0 = x[0], y[0]
    dx = x - x0
    dy = y - y0

    # 尺度归一化：用轨迹的总位移作为参考尺度
    scale = max(float(np.ptp(dx)), float(np.ptp(dy)), 1.0)

    # 轨迹平滑度：二阶差分的 median magnitude（抗异常帧）
    if len(dx) >= 3:
        ddx = np.diff(dx, n=2)
        ddy = np.diff(dy, n=2)
        curvature = float(np.median(np.sqrt(ddx**2 + ddy**2)))
    else:
        curvature = 0.0

    # 归一化曲率噪声
    curvature_norm = curvature / scale if scale > 0 else curvature

    # 平滑度评分：curvature_norm 越小越平滑
    # 这些阈值是工程经验值，未来需要用标注数据校准
    if curvature_norm <= 0.02:
        smooth_score = 100.0
    elif curvature_norm <= 0.05:
        smooth_score = 90.0
    elif curvature_norm <= 0.10:
        smooth_score = 75.0
    elif curvature_norm <= 0.20:
        smooth_score = 60.0
    else:
        smooth_score = max(30.0, 60.0 - (curvature_norm - 0.20) * 150.0)

    # 水平位移比（仅输出，不参与评分——因为理想 J-path 的偏移量因人而异）
    horizontal_ratio = float(np.ptp(dx)) / scale if scale > 0 else 0.0

    detail = (f"curvature_norm={curvature_norm:.4f}, "
              f"horizontal_ratio={horizontal_ratio:.3f}, "
              f"points={len(x)}")

    return MetricResult(
        "bar_path",
        raw=curvature_norm,
        score=_clamp(smooth_score),
        detail=detail,
    )


# ═══════════════════════════════════════
#  5. Symmetry — 双侧数据门槛
# ═══════════════════════════════════════

def compute_symmetry(rep: RepContext) -> MetricResult:
    # 非双侧数据，绝对不能判定不对称
    if rep.left_elbow is None or rep.right_elbow is None:
        return MetricResult("symmetry", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "缺少双侧数据")

    # 关键：bilateral_valid_ratio 不足时直接 N/A
    bilateral = float(getattr(rep, "bilateral_valid_ratio", 0.0))
    if bilateral < 0.65:
        return MetricResult(
            "symmetry", None, None,
            MetricStatus.INSUFFICIENT_DATA,
            f"双侧有效比例不足: {bilateral:.2f} (<0.65)",
        )

    left = np.asarray(rep.left_elbow, dtype=float)
    right = np.asarray(rep.right_elbow, dtype=float)

    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(len(left), rep.concentric_end - rep.start_frame)
    if ce - cs < 5:
        return MetricResult("symmetry", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "concentric 帧数不足")

    seg_l = left[cs:ce]
    seg_r = right[cs:ce]
    valid = np.isfinite(seg_l) & np.isfinite(seg_r)

    if valid.sum() < 5:
        return MetricResult("symmetry", None, None,
                            MetricStatus.INSUFFICIENT_DATA, "有效帧不足")

    diff = np.abs(seg_l[valid] - seg_r[valid])
    asym = float(np.nanmedian(diff))

    # <=3° 优秀，<=5° 良好，<=8° 可接受，<=12° 轻度不对称
    if asym <= 3:
        score = 100.0
    elif asym <= 5:
        score = 95.0
    elif asym <= 8:
        score = 85.0
    elif asym <= 12:
        score = 70.0
    else:
        score = max(40.0, 70.0 - (asym - 12.0) * 3.0)

    return MetricResult("symmetry", raw=asym, score=_clamp(score))


# ═══════════════════════════════════════
#  6. Direction Consistency
# ═══════════════════════════════════════

def compute_direction_consistency(rep: RepContext) -> MetricResult:
    if rep.eccentric_velocity is None or rep.concentric_velocity is None:
        return MetricResult("direction_consistency", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    ecc = np.asarray(rep.eccentric_velocity, dtype=float)
    con = np.asarray(rep.concentric_velocity, dtype=float)
    ecc = ecc[np.isfinite(ecc)]
    con = con[np.isfinite(con)]

    if len(ecc) < 5 or len(con) < 5:
        return MetricResult("direction_consistency", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    # eccentric 应主要为负（角度减小），concentric 应主要为正（角度增大）
    # 用 ±5°/s 容差带
    ecc_correct = float(np.mean(ecc <= 5.0))
    con_correct = float(np.mean(con >= -5.0))
    quality = 0.5 * ecc_correct + 0.5 * con_correct

    return MetricResult("direction_consistency", raw=quality,
                        score=_clamp(quality * 100.0))


# ═══════════════════════════════════════
#  7. Velocity Smoothness — MAD 替代 std
# ═══════════════════════════════════════

def compute_velocity_smoothness(rep: RepContext) -> MetricResult:
    if rep.concentric_velocity is None:
        return MetricResult("velocity_smoothness", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    v = np.asarray(rep.concentric_velocity, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) < 6:
        return MetricResult("velocity_smoothness", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    dv = np.diff(v)
    # robust MAD，抗异常帧
    med = np.median(dv)
    mad = float(np.median(np.abs(dv - med)))

    if mad <= 3:
        score = 100.0
    elif mad <= 6:
        score = 95.0
    elif mad <= 10:
        score = 85.0
    elif mad <= 15:
        score = 75.0
    elif mad <= 25:
        score = 60.0
    else:
        score = max(30.0, 60.0 - (mad - 25.0) * 1.5)

    return MetricResult("velocity_smoothness", raw=mad, score=_clamp(score))


# ═══════════════════════════════════════
#  8. Depth Control — 修复 joint_stress 不连续
# ═══════════════════════════════════════

def compute_depth_control(rep: RepContext) -> MetricResult:
    bottom = float(rep.actual_min_angle)

    if not np.isfinite(bottom) or bottom <= 0:
        return MetricResult("depth_control", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    # 只评价底部深度是否进入极端区间，不声称是"医学关节压力"
    #   >=75  安全深度
    #   65~75 可接受
    #   55~65 偏深
    #   45~55 较深
    #   <45   过深
    if bottom >= 75:
        score = 100.0
    elif bottom >= 65:
        score = 95.0
    elif bottom >= 55:
        score = 85.0
    elif bottom >= 45:
        score = 70.0
    else:
        score = 55.0

    return MetricResult("depth_control", raw=bottom, score=score)


# ═══════════════════════════════════════
#  9. Eccentric Control
# ═══════════════════════════════════════

def compute_eccentric_control(rep: RepContext) -> MetricResult:
    if rep.eccentric_velocity is None:
        return MetricResult("eccentric_control", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    v = np.asarray(rep.eccentric_velocity, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) < 5:
        return MetricResult("eccentric_control", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    peak = abs(float(np.nanmin(v)))

    # 不把高速度直接等价为危险
    if peak <= 120:
        score = 100.0
    elif peak <= 180:
        score = 95.0
    elif peak <= 240:
        score = 90.0
    elif peak <= 320:
        score = 80.0
    else:
        score = max(55.0, 80.0 - (peak - 320.0) * 0.10)

    return MetricResult("eccentric_control", raw=peak, score=_clamp(score))


# ═══════════════════════════════════════
#  9b. Butt Contact — 臀部是否离凳（V1.1 Pose-only）
# ═══════════════════════════════════════

def compute_butt_contact(rep: RepContext) -> MetricResult:
    """
    V1.1: 臀部接触状态评分，调用 ButtContactAnalyzer。

    评分映射：
      normal_arch     → 100（正常起桥，臀部接触凳面）
      suspected_lift  → 70（疑似臀部离凳，置信度 0.55~0.79）
      confirmed_lift  → 40（高置信度臀部离凳模式，置信度 >=0.80）
      insufficient_data → N/A（不扣分，自动从加权平均中剔除）

    注意：这是 Safety 层的一个指标，不是错误检测。
    错误检测层（detect_butt_off_bench）独立输出，不重复扣分。
    """
    analyzer = ButtContactAnalyzer()
    result = analyzer.analyze(rep)

    if result.status == "insufficient_data":
        return MetricResult(
            "butt_contact", None, None,
            MetricStatus.INSUFFICIENT_DATA,
            result.detail or "数据不足，无法判断臀部接触状态",
        )

    if result.status == "normal_arch":
        score = 100.0
    elif result.status == "suspected_lift":
        # suspected 置信度 0.55~0.79，分数 60~80 线性映射
        score = 60.0 + (result.confidence - 0.55) / (0.79 - 0.55) * 20.0
    elif result.status == "confirmed_lift":
        # confirmed 置信度 0.80~0.85，分数 30~50 线性映射
        score = 30.0 + (result.confidence - 0.80) / (0.85 - 0.80) * 20.0
    else:
        score = 50.0  # 未知状态，给中间分

    return MetricResult(
        "butt_contact",
        raw=result.max_relative_lift,
        score=_clamp(score),
        detail=(
            f"status={result.status}, "
            f"confidence={result.confidence:.2f}, "
            f"max_lift={result.max_relative_lift}px, "
            f"persistence={result.separated_frames}frames, "
            f"reason={result.reason}"
        ),
    )


# ═══════════════════════════════════════
#  9c. Bounce Control — 弹胸控制（Safety 层）
# ═══════════════════════════════════════

def compute_bounce_control(rep: RepContext) -> MetricResult:
    """
    V3.1: 弹胸控制评分，调用 detect_bounce（verbose=False 避免重复日志）。

    评分映射：
      Normal (NOT_DETECTED, 非 Touch&Go)  → 100
      Touch & Go (NOT_DETECTED, detail含"Touch & Go") → 90（合法变式，轻度扣分）
      Moderate Bounce (DETECTED + MODERATE) → 65
      Severe Bounce (DETECTED + SEVERE) → 35
      Insufficient Data → N/A（不扣分，自动从加权平均中剔除）

    注意：这是 Safety 层的一个指标，和错误检测层的 detect_bounce 是独立通道。
    错误检测层输出"检测到的问题"，评分层输出 0~100 分，不会重复扣分。
    """
    result = detect_bounce(rep, verbose=False)

    if result.status == ErrorStatus.INSUFFICIENT_DATA:
        return MetricResult(
            "bounce_control", None, None,
            MetricStatus.INSUFFICIENT_DATA,
            result.detail or "数据不足，无法判断弹胸",
        )

    if result.status == ErrorStatus.NOT_DETECTED:
        # 区分 Touch & Go 和 Normal
        if result.detail and "Touch & Go" in result.detail:
            score = 90.0
            label = "touch_and_go"
        else:
            score = 100.0
            label = "normal"
    elif result.status == ErrorStatus.DETECTED:
        if result.severity and result.severity.value == "severe":
            score = 35.0
            label = "severe_bounce"
        elif result.severity and result.severity.value == "moderate":
            score = 65.0
            label = "moderate_bounce"
        else:
            score = 50.0
            label = "bounce"
    else:
        score = 50.0
        label = "unknown"

    return MetricResult(
        "bounce_control",
        raw=result.value,
        score=_clamp(score),
        detail=f"{label}: {result.detail}",
    )


# ═══════════════════════════════════════
#  10. Concentric Speed（原 power，语义修正）
# ═══════════════════════════════════════

def compute_concentric_speed(rep: RepContext) -> MetricResult:
    if rep.concentric_velocity is None:
        return MetricResult("concentric_speed", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    v = np.asarray(rep.concentric_velocity, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) < 5:
        return MetricResult("concentric_speed", None, None,
                            MetricStatus.INSUFFICIENT_DATA)

    peak = float(np.nanmax(v))

    # 只是角速度表现，不叫 power
    if peak >= 250:
        score = 100.0
    elif peak >= 200:
        score = 95.0
    elif peak >= 150:
        score = 85.0
    elif peak >= 100:
        score = 75.0
    elif peak >= 50:
        score = 60.0
    else:
        score = 45.0

    return MetricResult("concentric_speed", raw=peak, score=score)


# ═══════════════════════════════════════
#  Layer aggregator
# ═══════════════════════════════════════

def aggregate_layer(
    layer_name: str,
    metrics: List[MetricResult],
    weights: Dict[str, float],
) -> LayerResult:
    result = _valid_weighted_average(metrics, weights)
    result.layer_name = layer_name
    return result


# ═══════════════════════════════════════
#  Main Scorer
# ═══════════════════════════════════════

class ExerciseSpecificScorerV2:

    def score_rep(self, rep: RepContext) -> RepScoreResult:
        result = RepScoreResult(rep_index=rep.rep_index)

        if rep.validation_status != ValidationStatus.VALID:
            result.status = MetricStatus.INSUFFICIENT_DATA
            return result

        # ── Technique ──
        technique_metrics = [
            compute_bar_path(rep),
            compute_rom(rep),
            compute_elbow_tuck(rep),
            compute_tempo(rep),
            compute_bottom_control(rep),
            compute_lockout(rep),
            compute_symmetry(rep),
        ]
        technique = aggregate_layer("technique", technique_metrics, TECHNIQUE_WEIGHTS)
        result.layers["technique"] = technique
        result.technique_score = technique.score

        # ── Movement Quality ──
        movement_metrics = [
            compute_direction_consistency(rep),
            compute_velocity_smoothness(rep),
        ]
        movement = aggregate_layer("movement_quality", movement_metrics, MOVEMENT_WEIGHTS)
        result.layers["movement_quality"] = movement
        result.movement_quality = movement.score

        # ── Safety / Control ──
        safety_metrics = [
            compute_depth_control(rep),
            compute_eccentric_control(rep),
            compute_butt_contact(rep),
            compute_bounce_control(rep),
        ]
        safety = aggregate_layer("safety", safety_metrics, SAFETY_WEIGHTS)
        result.layers["safety"] = safety
        result.safety_score = safety.score

        # ── Performance ──
        perf_metrics = [
            compute_concentric_speed(rep),
        ]
        performance = aggregate_layer("performance", perf_metrics, PERFORMANCE_WEIGHTS)
        result.layers["performance"] = performance
        result.performance_score = performance.score

        # ── Overall（只对有效层加权，0~100 → 0~10）──
        valid_layers = []
        for name, weight in LAYER_WEIGHTS.items():
            layer = result.layers.get(name)
            if (layer is not None and layer.score is not None
                    and np.isfinite(layer.score)):
                valid_layers.append((layer.score, weight))

        if not valid_layers:
            result.overall_score = None
            result.total_score = None
            result.status = MetricStatus.INSUFFICIENT_DATA
            return result

        total_weight = sum(w for _, w in valid_layers)
        score_100 = sum(s * (w / total_weight) for s, w in valid_layers)
        score_10 = round(score_100 / 10.0, 1)

        result.overall_score = score_10
        result.total_score = score_10

        # ── Data Quality（独立，不扣动作分）──
        result.data_quality_score = self._compute_data_quality(rep)

        # ── Metrics for frontend ──
        result.metrics = {
            "rom": float(rep.actual_rom) if np.isfinite(rep.actual_rom) else None,
            "tempo_ratio": (
                rep.eccentric_duration / rep.concentric_duration
                if rep.concentric_duration > 0 else None
            ),
            "peak_concentric_velocity": (
                float(rep.peak_concentric_velocity)
                if rep.peak_concentric_velocity is not None
                and np.isfinite(rep.peak_concentric_velocity) else None
            ),
            "bottom_dwell_time": (
                float(rep.bottom_dwell_time)
                if np.isfinite(rep.bottom_dwell_time) else None
            ),
        }

        # ── Errors（只做解释，不直接扣分）──
        result.errors = self._detect_errors(rep)

        # ── Grade ──
        result.grade = self._grade(score_10)

        return result

    # ═══════════════════════════════════════
    #  Data quality
    # ═══════════════════════════════════════

    @staticmethod
    def _compute_data_quality(rep: RepContext) -> float:
        score = 100.0

        bilateral = float(getattr(rep, "bilateral_valid_ratio", 1.0))
        if bilateral < 0.20:
            score -= 20
        elif bilateral < 0.50:
            score -= 10
        elif bilateral < 0.65:
            score -= 5

        if rep.eccentric_velocity is None:
            score -= 10
        if rep.concentric_velocity is None:
            score -= 10

        return _clamp(score)

    # ═══════════════════════════════════════
    #  Error detection（解释层，不直接扣分）
    # ═══════════════════════════════════════

    @staticmethod
    def _detect_errors(rep: RepContext) -> List[ScoreError]:
        errors: List[ScoreError] = []

        # ROM 不足
        if rep.actual_rom > 0 and rep.actual_rom < 60:
            errors.append(ScoreError(
                code="INSUFFICIENT_ROM",
                severity="moderate",
                message="动作幅度偏小",
            ))

        # Tempo 失衡
        if rep.concentric_duration > 0:
            ratio = rep.eccentric_duration / rep.concentric_duration
            if ratio < 0.75:
                errors.append(ScoreError(
                    code="TEMPO_UNBALANCED",
                    severity="minor",
                    message="离心阶段明显短于向心阶段",
                ))

        # Symmetry：只有数据充足时才检测
        bilateral = float(getattr(rep, "bilateral_valid_ratio", 0.0))
        if bilateral >= 0.65 and rep.left_elbow is not None and rep.right_elbow is not None:
            left = np.asarray(rep.left_elbow, dtype=float)
            right = np.asarray(rep.right_elbow, dtype=float)
            cs = max(0, rep.concentric_start - rep.start_frame)
            ce = min(len(left), rep.concentric_end - rep.start_frame)
            if ce - cs >= 5:
                seg_l, seg_r = left[cs:ce], right[cs:ce]
                valid = np.isfinite(seg_l) & np.isfinite(seg_r)
                if valid.sum() >= 5:
                    asym = float(np.nanmedian(np.abs(seg_l[valid] - seg_r[valid])))
                    if asym > 10:
                        errors.append(ScoreError(
                            code="ASYMMETRY",
                            severity="moderate",
                            message="左右侧动作存在明显差异",
                        ))

        return errors

    # ═══════════════════════════════════════
    #  Grade
    # ═══════════════════════════════════════

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 9.0:
            return "A"
        if score >= 8.0:
            return "B"
        if score >= 7.0:
            return "C"
        if score >= 6.0:
            return "D"
        return "E"


# ═══════════════════════════════════════
#  Frontend formatter
# ═══════════════════════════════════════

def format_v2_results_for_frontend(
    rep_scores: list,
    set_errors: list = None,
    fatigue_result=None,
    exercise_type: str = "unknown",
) -> dict:
    """将 V2 引擎原始输出转为前端可直接消费的 JSON 结构。"""
    reps_out = []
    for rs in rep_scores:
        reps_out.append({
            "rep_index": rs.rep_index,
            "score": rs.total_score,
            "grade": rs.grade,
            "data_quality": getattr(rs, "data_quality_score", 100.0),
            "errors": [
                {
                    "code": e.code,
                    "severity": e.severity,
                    "message": e.message,
                    "deduction": e.deduction,
                }
                for e in (rs.errors or [])
            ],
            "metrics": {
                "rom": rs.metrics.get("rom"),
                "tempo_ratio": rs.metrics.get("tempo_ratio"),
                "peak_concentric_velocity": rs.metrics.get("peak_concentric_velocity"),
                "bottom_dwell_time": rs.metrics.get("bottom_dwell_time"),
            },
            "layers": {
                ln: {
                    "score": lr.score,
                    "status": lr.status.value,
                    "metrics": [
                        {"key": m.key, "raw": m.raw, "score": m.score,
                         "status": m.status.value, "detail": m.detail}
                        for m in lr.metrics
                    ],
                }
                for ln, lr in (rs.layers or {}).items()
            },
        })

    set_errors_out = [
        {
            "code": e.error_id if hasattr(e, "error_id") else e.get("code"),
            "severity": (e.worst_severity.value if hasattr(e, "worst_severity") and e.worst_severity
                         else e.get("severity", "moderate")),
            "message": (e.display_name if hasattr(e, "display_name")
                        else e.get("message", "")),
        }
        for e in (set_errors or [])
    ]

    fatigue_out = None
    if fatigue_result and getattr(fatigue_result, "status", "") == "valid":
        fatigue_out = {
            "velocity_loss_pct": fatigue_result.velocity_loss_pct,
            "fatigue_level": fatigue_result.fatigue_level,
            "estimated_rir": fatigue_result.estimated_rir,
            "trend": fatigue_result.trend,
        }

    valid_scores = [r["score"] for r in reps_out if r["score"] is not None]
    avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0.0

    return {
        "exercise_type": exercise_type,
        "total_reps": len(reps_out),
        "average_score": avg_score,
        "reps": reps_out,
        "set_errors": set_errors_out,
        "fatigue": fatigue_out,
    }
