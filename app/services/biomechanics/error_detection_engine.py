"""
error_detection_engine.py (v2)

修正：
  - bounce: 多信号联合判断
  - elbow_flare: 无肩外展数据 → INSUFFICIENT_DATA
  - hip_lift: 无髋部数据 → INSUFFICIENT_DATA
  - Set 级别聚合
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import warnings

from app.domain.models import RepContext
from app.domain.enums import ValidationStatus, ErrorStatus, ErrorSeverity
from .butt_contact_analyzer import ButtContactAnalyzer


@dataclass
class ErrorDetection:
    error_id: str
    rep_index: int
    status: ErrorStatus
    severity: Optional[ErrorSeverity] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    confidence: float = 0.0
    detail: str = ""


@dataclass
class SetLevelError:
    error_id: str
    display_name: str
    occurrences: List[int] = field(default_factory=list)
    frequency: float = 0.0
    worst_severity: Optional[ErrorSeverity] = None
    avg_value: Optional[float] = None


# ═══════════════════════════════════════
#  规则函数
# ═══════════════════════════════════════

def detect_incomplete_rom(rep: RepContext, min_rom: float = 60.0) -> ErrorDetection:
    eid = "bench_incomplete_rom"
    if rep.actual_rom <= 0:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="ROM 不可用")
    if rep.actual_rom < min_rom:
        sev = ErrorSeverity.SEVERE if rep.actual_rom < min_rom * 0.6 else ErrorSeverity.MODERATE
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED, sev,
                              value=rep.actual_rom, threshold=min_rom, confidence=0.9,
                              detail=f"actual_rom={rep.actual_rom:.1f}° < {min_rom}°")
    return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                          value=rep.actual_rom, threshold=min_rom, confidence=0.9)


def detect_incomplete_lockout(rep: RepContext, threshold: float = 160.0) -> ErrorDetection:
    """
    V3: 锁定检测 = top extension + top plateau + stability，不再用单一 160° 阈值。

    判断逻辑：
      1. 取 concentric 段末尾 15%（至少 3 帧）作为顶部窗口
      2. 计算顶部窗口的 std、range、near_top_ratio
      3. 如果进入平台期（稳定）→ 锁定确认，NOT_DETECTED
      4. 如果 top >= 165° 且基本稳定 → NOT_DETECTED
      5. 否则 → DETECTED（confidence 降低，因为可能是个体差异）
    """
    eid = "bench_incomplete_lockout"

    if rep.top2_angle <= 0 or not np.isfinite(rep.top2_angle):
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="顶部角度无效")

    if rep.bilateral_elbow is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少肘角信号")

    # concentric 段在 rep-relative 数组中的索引
    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(len(rep.bilateral_elbow), rep.concentric_end - rep.start_frame + 1)

    if ce - cs < 5:
        # concentric 太短，退化为单一角度判断
        top = float(rep.top2_angle)
        if top >= threshold:
            return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                                  value=top, threshold=threshold, confidence=0.7,
                                  detail=f"concentric 过短，退化为角度判断 top={top:.1f}°")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.MODERATE, value=top, threshold=threshold,
                              confidence=0.6, detail=f"concentric 过短，top={top:.1f}°")

    con_seg = np.asarray(rep.bilateral_elbow[cs:ce], dtype=float)
    con_seg = con_seg[np.isfinite(con_seg)]

    if len(con_seg) < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="concentric 有效帧不足")

    top = float(np.nanmax(con_seg))

    # 顶部窗口：concentric 末尾 15%，至少 3 帧
    window_size = max(3, int(len(con_seg) * 0.15))
    top_window = con_seg[-window_size:]

    top_std = float(np.nanstd(top_window))
    top_range = float(np.ptp(top_window))
    near_top_ratio = float(np.mean(top_window >= top - 3.0))

    # 末尾趋势：最后 3 帧的角度变化，如果还在明显上升说明没到平台
    if len(top_window) >= 3:
        tail_trend = float(top_window[-1] - top_window[-3])
    else:
        tail_trend = 0.0

    detail = (f"top={top:.1f}°, top_std={top_std:.1f}, "
              f"top_range={top_range:.1f}, plateau={near_top_ratio:.2f}, "
              f"tail_trend={tail_trend:+.1f}°")

    # ── 判断 1：已经进入稳定平台期 → 锁定确认 ──
    stable_plateau = (
        top_std <= 3.0
        and top_range <= 6.0
        and near_top_ratio >= 0.60
        and abs(tail_trend) <= 2.0
    )

    if stable_plateau:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=top, threshold=threshold, confidence=0.9,
            detail=f"顶部平台期确认，{detail}",
        )

    # ── 判断 2：角度充分高（>=165°）即使平台不完美也认为锁定 ──
    if top >= 165.0 and near_top_ratio >= 0.40:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=top, threshold=threshold, confidence=0.8,
            detail=f"顶部角度充分，{detail}",
        )

    # ── 判断 3：顶部角度极低（<145°）且无平台 → 明确未锁定 ──
    if top < 145.0:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED,
            ErrorSeverity.MODERATE, value=top, threshold=threshold, confidence=0.85,
            detail=f"顶部角度明显不足，{detail}",
        )

    # ── 判断 4：中间地带（145~165° 且无稳定平台）→ 报出但 confidence 低 ──
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.DETECTED,
        ErrorSeverity.MODERATE, value=top, threshold=threshold, confidence=0.55,
        detail=f"顶部未形成稳定平台（可能个体差异），{detail}",
    )


def detect_elbow_flare(rep: RepContext) -> ErrorDetection:
    """
    V3: 真正的 elbow flare 检测，使用 upper_arm_torso_angle（上臂与躯干夹角）。
    不是 elbow joint angle！

    取 bottom 帧附近的 upper_arm_torso_angle，>80° 判定为明显外展。

    V3.1 修复：加单侧视角/数据质量可信度控制。
    - 双侧都有数据 → confidence=0.8
    - 只有单侧数据（bilateral_ratio<0.65）→ confidence 降至 0.55~0.65（透视可能影响角度）
    """
    eid = "bench_elbow_flare"

    # 取 bottom 帧附近的 upper_arm_torso_angle
    bf_rel = rep.bottom_frame - rep.start_frame
    window = max(1, int(0.1 * rep.fps))

    left_values = []
    right_values = []
    for side_arr, side_list in [
        (rep.left_upper_arm_torso, left_values),
        (rep.right_upper_arm_torso, right_values),
    ]:
        if side_arr is None:
            continue
        start = max(0, bf_rel - window)
        end = min(len(side_arr), bf_rel + window + 1)
        if end > start:
            seg = np.asarray(side_arr[start:end], dtype=float)
            seg = seg[np.isfinite(seg)]
            if len(seg) > 0:
                side_list.append(float(np.median(seg)))

    # 统计双侧数据可用性
    left_has = len(left_values) > 0
    right_has = len(right_values) > 0
    bilateral_ratio = float(getattr(rep, "bilateral_valid_ratio", 0.0))

    if not left_has and not right_has:
        return ErrorDetection(
            eid, rep.rep_index,
            ErrorStatus.INSUFFICIENT_DATA,
            detail="缺少 upper_arm_torso 数据（feature_extractor 可能未计算）",
            confidence=0.0,
        )

    # 取较大的一侧（更外展的一侧）
    all_values = left_values + right_values
    angle = float(np.max(all_values))

    # 可信度控制：单侧视角降低 confidence
    # 双侧都有数据 → 0.8
    # 只有单侧数据 或 bilateral_ratio<0.65 → 0.55（透视可能影响角度准确性）
    if left_has and right_has and bilateral_ratio >= 0.65:
        base_confidence = 0.8
        view_note = "双侧数据"
    else:
        base_confidence = 0.55
        view_note = f"单侧视角(bilateral_ratio={bilateral_ratio:.3f})，透视可能影响角度准确性"

    if angle > 90.0:
        sev = ErrorSeverity.SEVERE
    elif angle > 80.0:
        sev = ErrorSeverity.MODERATE
    elif angle > 75.0:
        sev = ErrorSeverity.MILD
    else:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=angle, threshold=75.0, confidence=base_confidence,
            detail=f"upper_arm_torso={angle:.1f}° ({view_note})",
        )

    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.DETECTED, sev,
        value=angle, threshold=75.0, confidence=base_confidence,
        detail=f"upper_arm_torso={angle:.1f}° ({view_note})",
    )


def detect_bounce(rep: RepContext) -> ErrorDetection:
    """砸胸检测 — 多信号联合判断"""
    eid = "bench_bounce"

    if rep.pre_bottom_velocity is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="无 bottom 前速度数据")

    PRE_VEL_THRESH = 180.0
    DWELL_THRESH = 0.08
    REVERSAL_THRESH = 2
    ACCEL_THRESH = 3000.0

    signals_hit = 0
    details = []

    fast_descent = rep.pre_bottom_velocity > PRE_VEL_THRESH
    if fast_descent:
        signals_hit += 1
        details.append(f"pre_vel={rep.pre_bottom_velocity:.0f}°/s")

    short_dwell = rep.bottom_dwell_time < DWELL_THRESH
    if short_dwell:
        signals_hit += 1
        details.append(f"dwell={rep.bottom_dwell_time * 1000:.0f}ms")

    fast_reversal = rep.direction_reversal_frames <= REVERSAL_THRESH
    if fast_reversal:
        signals_hit += 1
        details.append(f"reversal={rep.direction_reversal_frames}frames")

    high_accel = (rep.bottom_acceleration is not None
                  and abs(rep.bottom_acceleration) > ACCEL_THRESH)
    if high_accel:
        signals_hit += 1
        details.append(f"accel={rep.bottom_acceleration:.0f}°/s²")

    if signals_hit >= 3:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.SEVERE, value=rep.pre_bottom_velocity,
                              threshold=PRE_VEL_THRESH, confidence=0.9,
                              detail="; ".join(details))
    elif signals_hit == 2 and (fast_descent and (short_dwell or fast_reversal)):
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.MODERATE, value=rep.pre_bottom_velocity,
                              threshold=PRE_VEL_THRESH, confidence=0.75,
                              detail="; ".join(details))
    else:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                              value=rep.pre_bottom_velocity, confidence=0.7,
                              detail="; ".join(details) if details else "正常触胸")


def detect_butt_off_bench(rep: RepContext) -> ErrorDetection:
    """
    V1: 臀部离凳检测（Pose-only）。

    使用 ButtContactAnalyzer 分析骨盆相对肩部的抬升模式。
    - normal_arch → NOT_DETECTED
    - suspected_lift → SUSPECTED（置信度 0.55~0.80）
    - confirmed_lift → DETECTED（置信度 >=0.80，V1 很难达到）
    - insufficient_data → INSUFFICIENT_DATA
    """
    eid = "bench_butt_off_bench"

    analyzer = ButtContactAnalyzer()
    result = analyzer.analyze(rep)

    print(f"\n   🍑 [最终检测] Rep {rep.rep_index}: status={result.status}, "
          f"confidence={result.confidence:.2f}, max_lift={result.max_relative_lift}, "
          f"separated_frames={result.separated_frames}, reason={result.reason}")
    print(f"   🍑 [最终检测] Rep {rep.rep_index}: detail={result.detail}")

    if result.status == "insufficient_data":
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
            detail=result.detail or "缺少髋部坐标数据",
            confidence=0.0,
        )

    if result.status == "normal_arch":
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=result.max_relative_lift,
            confidence=result.confidence,
            detail=result.detail or "正常起桥，未检测到臀部离凳",
        )

    if result.status == "suspected_lift":
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.SUSPECTED,
            ErrorSeverity.MODERATE,
            value=result.max_relative_lift,
            confidence=result.confidence,
            detail=f"疑似臀部离凳: {result.detail}",
        )

    # confirmed_lift
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.DETECTED,
        ErrorSeverity.SEVERE,
        value=result.max_relative_lift,
        confidence=result.confidence,
        detail=f"确认臀部离凳: {result.detail}",
    )


def detect_asymmetric_push(rep: RepContext, max_asym: float = 12.0) -> ErrorDetection:
    eid = "bench_asymmetric_push"

    # 关键修复：双侧有效比例不足时，绝对不能判定"左右不对称"
    # 这是"测不到"，不是"做得差"
    bilateral = float(getattr(rep, "bilateral_valid_ratio", 0.0))
    if bilateral < 0.65:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
            detail=f"双侧有效比例不足: {bilateral:.2f} (<0.65)，跳过对称性判断",
            confidence=0.0,
        )

    if rep.left_elbow is None or rep.right_elbow is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少单侧数据")

    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(len(rep.left_elbow), rep.concentric_end - rep.start_frame)
    if ce - cs < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="concentric 有效帧不足")

    seg_l = np.asarray(rep.left_elbow[cs: ce], dtype=float)
    seg_r = np.asarray(rep.right_elbow[cs: ce], dtype=float)
    valid = np.isfinite(seg_l) & np.isfinite(seg_r)

    if valid.sum() < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="双侧同时有效帧不足")

    # 用 median 替代 mean，抗异常帧
    asym = float(np.nanmedian(np.abs(seg_l[valid] - seg_r[valid])))

    if asym > max_asym:
        sev = ErrorSeverity.SEVERE if asym > max_asym * 2 else ErrorSeverity.MODERATE
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED, sev,
                              value=asym, threshold=max_asym, confidence=0.8,
                              detail=f"asym={asym:.1f}°")
    return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                          value=asym, threshold=max_asym, confidence=0.8)


BENCH_RULES = [
    detect_incomplete_rom,
    detect_incomplete_lockout,
    detect_elbow_flare,
    detect_bounce,
    detect_butt_off_bench,
    detect_asymmetric_push,
]


class ErrorDetectionEngineFixed:

    def detect_set(self, reps: List[RepContext]) -> List[SetLevelError]:
        all_det: Dict[str, List[ErrorDetection]] = {}

        for rep in reps:
            if rep.validation_status != ValidationStatus.VALID:
                continue
            for fn in BENCH_RULES:
                d = fn(rep)
                all_det.setdefault(d.error_id, []).append(d)

        valid_count = sum(1 for r in reps if r.validation_status == ValidationStatus.VALID)
        results: List[SetLevelError] = []

        for eid, dets in all_det.items():
            # 统计 DETECTED 和 SUSPECTED 的错误
            hits = [d for d in dets if d.status in (ErrorStatus.DETECTED, ErrorStatus.SUSPECTED)]
            if not hits:
                continue
            vals = [d.value for d in hits if d.value is not None]
            sevs = [d.severity for d in hits if d.severity]
            worst = max(sevs, key=lambda s: list(ErrorSeverity).index(s)) if sevs else None

            results.append(SetLevelError(
                error_id=eid,
                display_name=self._name(eid),
                occurrences=[d.rep_index for d in hits],
                frequency=len(hits) / max(1, valid_count),
                worst_severity=worst,
                avg_value=float(np.mean(vals)) if vals else None,
            ))

        return results

    @staticmethod
    def _name(eid: str) -> str:
        return {
            "bench_incomplete_rom": "动作幅度不足",
            "bench_incomplete_lockout": "锁定不完全",
            "bench_elbow_flare": "肘部外展",
            "bench_bounce": "砸胸/弹胸",
            "bench_butt_off_bench": "臀部离凳",
            "bench_asymmetric_push": "左右发力不对称",
        }.get(eid, eid)