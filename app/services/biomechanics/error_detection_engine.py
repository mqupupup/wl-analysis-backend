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


def detect_bounce(rep: RepContext, verbose: bool = True) -> ErrorDetection:
    """
    V3.4 Bounce Detector — 多证据评分：高速接近 + 高速恢复 + 快速反弹 + 短停留。

    V3.4 核心修复（解决 V3.3 漏检）：
      1. recovery 门槛从 0.75 降到 0.50（high）/ 0.65（extreme）。
         V3.3 要求 recovery>=0.75 太苛刻，导致 462°/s+0ms+1f 被判成 Touch&Go。
      2. approach 门槛从 180/250 改成 300/400（用 peak speed）。
      3. 新增 rebound_latency_frames（bottom→向心速度达pre_peak*40%所需帧数），
         比 direction_reversal_frames（只判断方向是否反转）强很多。
      4. 用 impact_score 多证据评分，不再依赖单一 recovery ratio 硬门槛。
      5. 产品文案用"疑似砸胸/弹胸"而非"确认砸胸"（单目2D无法真正证明撞击）。

    判定逻辑：
      Severe   = fast_approach(>=400) + high_recovery(>=0.50) + (rapid_rebound or (short_dwell and rapid_reversal)) + impact_score>=6
      Moderate = moderate_approach(>=300) + high_recovery(>=0.50) + (rapid_rebound or (short_dwell and rapid_reversal)) + impact_score>=4
      Touch&Go = short_dwell + rapid_reversal + NOT high_recovery + approach<400
      Normal   = 其他

    Args:
        rep: RepContext
        verbose: 是否输出诊断日志（评分层调用时传 False 避免重复输出）
    """
    eid = "bench_bounce"

    # ── 数据可用性检查：优先用 V3.4 peak speed ──
    if rep.pre_bottom_peak_speed is None or not np.isfinite(rep.pre_bottom_peak_speed):
        # fallback 到旧字段
        if rep.pre_bottom_velocity is None or not np.isfinite(rep.pre_bottom_velocity):
            return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                                  detail="缺少 bottom 前接近速度")
        approach = float(rep.pre_bottom_velocity)
    else:
        approach = float(rep.pre_bottom_peak_speed)

    post_peak = (
        float(rep.post_bottom_peak_speed)
        if rep.post_bottom_peak_speed is not None and np.isfinite(rep.post_bottom_peak_speed)
        else None
    )
    recovery = (
        float(rep.impact_recovery_ratio)
        if rep.impact_recovery_ratio is not None and np.isfinite(rep.impact_recovery_ratio)
        else None
    )
    latency = (
        int(rep.rebound_latency_frames)
        if getattr(rep, "rebound_latency_frames", None) is not None
        else None
    )
    dwell = (
        float(rep.bottom_dwell_time)
        if rep.bottom_dwell_time is not None and np.isfinite(rep.bottom_dwell_time)
        else np.nan
    )
    reversal_frames = (
        int(rep.direction_reversal_frames)
        if rep.direction_reversal_frames is not None else 999
    )

    # ── 阈值定义（V3.4） ──
    FAST_APPROACH = 400.0
    MODERATE_APPROACH = 300.0
    SHORT_DWELL = 0.08
    RAPID_REVERSAL = 2
    HIGH_RECOVERY = 0.50
    EXTREME_RECOVERY = 0.65
    RAPID_REBOUND_LATENCY = 3

    # ── 基础信号 ──
    fast_approach = approach >= FAST_APPROACH
    moderate_approach = approach >= MODERATE_APPROACH
    short_dwell = np.isfinite(dwell) and dwell <= SHORT_DWELL
    rapid_reversal = reversal_frames <= RAPID_REVERSAL
    high_recovery = recovery is not None and recovery >= HIGH_RECOVERY
    extreme_recovery = recovery is not None and recovery >= EXTREME_RECOVERY
    rapid_rebound = latency is not None and latency <= RAPID_REBOUND_LATENCY

    # ── V3.4: 多证据 impact_score ──
    impact_score = 0
    if fast_approach:
        impact_score += 2
    elif moderate_approach:
        impact_score += 1
    if high_recovery:
        impact_score += 2
    if extreme_recovery:
        impact_score += 1
    if rapid_rebound:
        impact_score += 2
    if short_dwell:
        impact_score += 1
    if rapid_reversal:
        impact_score += 1

    # ── 诊断日志 ──
    if verbose:
        print(f"\n   🔻 [Bounce V3.4] Rep {rep.rep_index}")
        approach_label = "[FAST]" if fast_approach else "[MODERATE]" if moderate_approach else "[NORMAL]"
        print(f"   🔻   approach={approach:.1f}°/s {approach_label}")
        if post_peak is not None:
            print(f"   🔻   post_peak={post_peak:.1f}°/s")
        else:
            print(f"   🔻   post_peak=N/A")
        if recovery is not None:
            recovery_label = "[EXTREME]" if extreme_recovery else "[HIGH]" if high_recovery else "[NORMAL]"
            print(f"   🔻   recovery={recovery:.2f} {recovery_label}")
        else:
            print(f"   🔻   recovery=N/A")
        if latency is not None:
            print(f"   🔻   rebound_latency={latency}f {'[RAPID]' if rapid_rebound else '[NORMAL]'}")
        else:
            print(f"   🔻   rebound_latency=N/A")
        if np.isfinite(dwell):
            print(f"   🔻   dwell={dwell*1000:.0f}ms {'[SHORT]' if short_dwell else '[NORMAL]'}")
        else:
            print(f"   🔻   dwell=N/A")
        print(f"   🔻   reversal={reversal_frames}f {'[RAPID]' if rapid_reversal else '[NORMAL]'}")
        print(f"   🔻   impact_score={impact_score}")

    # ═══════════════════════════════════════
    # 第一层：Context Gate
    # 没有足够的接近速度，绝不能判 Bounce
    # ═══════════════════════════════════════
    if not moderate_approach:
        if verbose:
            print("   🔻   → NORMAL (接近速度不足以支持 Bounce)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=approach, threshold=MODERATE_APPROACH, confidence=0.93,
            detail=f"接近速度不足以支持 Bounce: approach={approach:.1f}°/s (<{MODERATE_APPROACH})",
        )

    # ═══════════════════════════════════════
    # Severe Bounce
    # fast_approach + high_recovery + (rapid_rebound or (short_dwell and rapid_reversal)) + impact_score>=6
    # ═══════════════════════════════════════
    if (fast_approach and high_recovery
            and (rapid_rebound or (short_dwell and rapid_reversal))
            and impact_score >= 6):
        if verbose:
            print("   🔻   → SEVERE BOUNCE (疑似)")
        detail_parts = [f"approach={approach:.0f}°/s"]
        if post_peak is not None:
            detail_parts.append(f"post={post_peak:.0f}°/s")
        if recovery is not None:
            detail_parts.append(f"recovery={recovery:.2f}")
        if latency is not None:
            detail_parts.append(f"latency={latency}f")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED,
            ErrorSeverity.SEVERE, value=approach,
            threshold=FAST_APPROACH, confidence=0.90,
            detail="疑似高速砸胸反弹: " + ", ".join(detail_parts),
        )

    # ═══════════════════════════════════════
    # Moderate Bounce
    # moderate_approach + high_recovery + (rapid_rebound or (short_dwell and rapid_reversal)) + impact_score>=4
    # ═══════════════════════════════════════
    if (moderate_approach and high_recovery
            and (rapid_rebound or (short_dwell and rapid_reversal))
            and impact_score >= 4):
        if verbose:
            print("   🔻   → MODERATE BOUNCE (疑似)")
        recovery_text = f"{recovery:.2f}" if recovery is not None else "N/A"
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED,
            ErrorSeverity.MODERATE, value=approach,
            threshold=MODERATE_APPROACH, confidence=0.82,
            detail=(
                f"疑似底部弹震: approach={approach:.0f}°/s, "
                f"recovery={recovery_text}, impact_score={impact_score}"
            ),
        )

    # ═══════════════════════════════════════
    # Touch & Go
    # short_dwell + rapid_reversal + NOT high_recovery + approach<400
    # ═══════════════════════════════════════
    if short_dwell and rapid_reversal and not high_recovery and approach < FAST_APPROACH:
        if verbose:
            print("   🔻   → TOUCH & GO (正常)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
            value=approach, threshold=MODERATE_APPROACH, confidence=0.90,
            detail=f"Touch & Go: approach={approach:.0f}°/s (无高恢复证据)",
        )

    # ═══════════════════════════════════════
    # Normal
    # ═══════════════════════════════════════
    if verbose:
        print("   🔻   → NORMAL")
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=approach, threshold=MODERATE_APPROACH, confidence=0.85,
        detail="正常底部控制",
    )


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


# ═══════════════════════════════════════════════════════════════
# 新增检测项（第一批：离心速度/杠铃路径/触胸漂移/握距一致性）
# ═══════════════════════════════════════════════════════════════

def detect_eccentric_speed(rep: RepContext) -> ErrorDetection:
    """
    离心阶段速度失控检测。

    与 detect_bounce 的区别：
    - bounce 关注 bottom 前的 approach speed（是否撞到底部）
    - 本函数关注整个 eccentric phase 的速度控制（是否全程下放过快）

    分析整个 eccentric 段的 elbow angle |velocity| 的 mean 和 90th percentile。
    """
    eid = "bench_eccentric_speed"

    if rep.bilateral_elbow is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少肘角信号")

    angles = np.asarray(rep.bilateral_elbow, dtype=float)
    ecc_start = max(0, rep.eccentric_start - rep.start_frame)
    ecc_end = min(len(angles), rep.eccentric_end - rep.start_frame + 1)

    if ecc_end - ecc_start < 3:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="eccentric段数据不足")

    ecc_angles = angles[ecc_start:ecc_end]
    ecc_vel = np.gradient(ecc_angles, 1.0 / rep.fps)
    ecc_vel = ecc_vel[np.isfinite(ecc_vel)]

    if len(ecc_vel) < 3:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="eccentric速度有效点不足")

    # 卧推 eccentric 时 elbow angle 下降，velocity 为负，取绝对值
    mean_speed = float(np.nanmean(np.abs(ecc_vel)))
    peak_speed = float(np.percentile(np.abs(ecc_vel), 90))

    # 工程初始阈值（°/s）
    MODERATE_PEAK = 200.0
    SEVERE_PEAK = 300.0
    MODERATE_MEAN = 120.0
    SEVERE_MEAN = 180.0

    if peak_speed >= SEVERE_PEAK or mean_speed >= SEVERE_MEAN:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.SEVERE,
            value=peak_speed, threshold=SEVERE_PEAK, confidence=0.85,
            detail=f"离心速度失控: peak={peak_speed:.0f}°/s, mean={mean_speed:.0f}°/s",
        )

    if peak_speed >= MODERATE_PEAK or mean_speed >= MODERATE_MEAN:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.MODERATE,
            value=peak_speed, threshold=MODERATE_PEAK, confidence=0.75,
            detail=f"离心速度偏快: peak={peak_speed:.0f}°/s, mean={mean_speed:.0f}°/s",
        )

    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=peak_speed, threshold=MODERATE_PEAK, confidence=0.80,
        detail=f"离心速度正常: peak={peak_speed:.0f}°/s, mean={mean_speed:.0f}°/s",
    )


def detect_bar_path(rep: RepContext) -> ErrorDetection:
    """
    V6: 杠铃轨迹几何 + 稳定性检测（完整rep + 中值滤波 + 速度门控）。

    核心修复（解决 V5 数据不足 + 几何定义问题）：
    1. 分析完整 rep [start→end]，而不是只 concentric（规格要求 eccentric+concentric）
    2. mixed proxy 后加 3 点中值滤波，去除 left/right 切换产生的人为跳变
    3. 速度门控：去除不连续跳点（位移 > 0.20*shoulder_width），线性插值恢复
    4. reference path corridor：用完整 rep 的二次拟合作为参考曲线，计算 residual
    5. p90_offset 保留作为辅助证据（检测整体偏移），residual 作为主要路径偏差指标
    6. 上游配合：pose_estimation.py 对 wrist(15,16) 使用更低 visibility 阈值(0.15)

    V5 的问题：只分析 concentric（31帧），mixed proxy 左右切换产生跳变，
    wrist 上游 visibility>0.3 门控太严导致只有 5/31 帧可用。
    """
    eid = "bench_bar_path"

    # ═══════════════════════════════════════════════════════════
    # 0. 诊断日志（函数最开头，确保所有早退原因都能看到）
    # ═══════════════════════════════════════════════════════════
    print(f"\n   🛤️ [BarPath V6] Rep {rep.rep_index}")
    left_shape = None if rep.left_wrist is None else np.asarray(rep.left_wrist).shape
    right_shape = None if rep.right_wrist is None else np.asarray(rep.right_wrist).shape
    print(f"   🛤️   left_wrist={left_shape}")
    print(f"   🛤️   right_wrist={right_shape}")
    print(f"   🛤️   rep_abs=[{rep.start_frame}, {rep.end_frame}]")
    print(f"   🛤️   concentric_abs=[{rep.concentric_start}, {rep.concentric_end}]")
    if rep.bottom_frame is not None:
        print(f"   🛤️   bottom_frame={rep.bottom_frame}")

    # ═══════════════════════════════════════════════════════════
    # 1. 基础数据（允许单侧为 None）
    # ═══════════════════════════════════════════════════════════
    if rep.left_wrist is None and rep.right_wrist is None:
        print("   ❌ [BarPath] EARLY RETURN: 两侧 wrist 都是 None")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              confidence=0.0, detail="左右 wrist 均不可用")

    left = np.asarray(rep.left_wrist, dtype=float) if rep.left_wrist is not None else None
    right = np.asarray(rep.right_wrist, dtype=float) if rep.right_wrist is not None else None

    # ═══════════════════════════════════════════════════════════
    # 2. 完整 rep 范围（eccentric + concentric）
    # ═══════════════════════════════════════════════════════════
    start = 0
    end_candidates = []
    if left is not None:
        end_candidates.append(len(left))
    if right is not None:
        end_candidates.append(len(right))
    end = min(end_candidates)

    # 去掉前后各 1 帧边界噪声
    if end - start > 10:
        start += 1
        end -= 1

    print(f"   🛤️   full_rep=[{start}:{end}] len={end - start}")

    if end - start < 8:
        print(f"   ❌ [BarPath] EARLY RETURN: full rep len={end - start} < 8")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              confidence=0.0, detail=f"完整 rep 轨迹长度不足: {end - start}")

    # ═══════════════════════════════════════════════════════════
    # 3. 逐帧构造 bar proxy（mixed proxy）
    # ═══════════════════════════════════════════════════════════
    l_seg = left[start:end] if left is not None else None
    r_seg = right[start:end] if right is not None else None

    n = end - start
    bar = np.full((n, 2), np.nan, dtype=float)

    left_valid = np.zeros(n, dtype=bool)
    right_valid = np.zeros(n, dtype=bool)

    if l_seg is not None:
        left_valid = np.isfinite(l_seg[:, 0]) & np.isfinite(l_seg[:, 1])
    if r_seg is not None:
        right_valid = np.isfinite(r_seg[:, 0]) & np.isfinite(r_seg[:, 1])

    # 双腕 -> midpoint
    both = left_valid & right_valid
    if np.any(both):
        bar[both] = (l_seg[both] + r_seg[both]) / 2.0

    # 只有左腕
    left_only = left_valid & ~right_valid
    if np.any(left_only):
        bar[left_only] = l_seg[left_only]

    # 只有右腕
    right_only = right_valid & ~left_valid
    if np.any(right_only):
        bar[right_only] = r_seg[right_only]

    bar_valid = np.isfinite(bar[:, 0]) & np.isfinite(bar[:, 1])
    bar_valid_ratio = float(np.mean(bar_valid))
    both_ratio = float(np.mean(both))
    left_ratio = float(np.mean(left_valid))
    right_ratio = float(np.mean(right_valid))

    print(f"   🛤️   left={left_ratio:.3f} right={right_ratio:.3f} "
          f"both={both_ratio:.3f} bar_proxy={bar_valid_ratio:.3f} "
          f"valid_frames={int(bar_valid.sum())}/{n}")

    if bar_valid.sum() < 8:
        print(f"   ❌ [BarPath] EARLY RETURN: 有效 bar proxy={int(bar_valid.sum())} < 8")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA, confidence=0.0,
            detail=f"有效 bar proxy 帧不足: {int(bar_valid.sum())}/{n}",
        )

    # proxy_mode 三档
    if both_ratio >= 0.50:
        proxy_mode = "bilateral_midpoint"
        quality = 0.90
    elif bar_valid_ratio >= 0.65:
        proxy_mode = "mixed_wrist_proxy"
        quality = 0.68
    else:
        proxy_mode = "sparse_wrist_proxy"
        quality = 0.50

    # ═══════════════════════════════════════════════════════════
    # 3.5 中值滤波 + 速度门控（去除 mixed proxy 左右切换跳变）
    # ═══════════════════════════════════════════════════════════
    # 先线性插值填充 NaN（用于中值滤波和速度门控）
    bar_interp = bar.copy()
    for col in range(2):
        valid = np.isfinite(bar_interp[:, col])
        if np.sum(valid) >= 2:
            bar_interp[:, col] = np.interp(
                np.arange(n), np.where(valid)[0], bar_interp[valid, col]
            )

    # 3 点中值滤波（去除左右切换跳变）
    def _median_filter_1d(x, kernel=3):
        result = x.copy()
        half = kernel // 2
        for i in range(len(x)):
            lo = max(0, i - half)
            hi = min(len(x), i + half + 1)
            window = x[lo:hi]
            if len(window) > 0:
                result[i] = np.median(window)
        return result

    bar_filtered = np.column_stack([
        _median_filter_1d(bar_interp[:, 0], 3),
        _median_filter_1d(bar_interp[:, 1], 3),
    ])

    # 速度门控：去除不连续跳点（位移 > 0.20 * 估计尺度）
    # 先用 bar x 的 IQR 估计尺度（避免依赖 shoulder_width）
    x_iqr = float(np.percentile(bar_filtered[:, 0], 75) -
                   np.percentile(bar_filtered[:, 0], 25))
    speed_threshold = max(0.20 * x_iqr, 5.0)  # 至少 5 像素

    jump_count = 0
    for i in range(1, n):
        if abs(bar_filtered[i, 0] - bar_filtered[i-1, 0]) > speed_threshold:
            # 用前后值的平均替换跳点
            bar_filtered[i, 0] = (bar_filtered[i-1, 0] + bar_filtered[min(i+1, n-1), 0]) / 2
            bar_filtered[i, 1] = (bar_filtered[i-1, 1] + bar_filtered[min(i+1, n-1), 1]) / 2
            jump_count += 1

    print(f"   🛤️   median_filter + speed_gate: jumps_removed={jump_count}, "
          f"speed_threshold={speed_threshold:.1f}px")

    # ═══════════════════════════════════════════════════════════
    # 4. 逐帧肩部参考（shoulder_mid_x 数组）
    # ═══════════════════════════════════════════════════════════
    shoulder_mid_x = None
    shoulder_width = None

    if rep.left_shoulder is not None and rep.right_shoulder is not None:
        ls = np.asarray(rep.left_shoulder, dtype=float)
        rs = np.asarray(rep.right_shoulder, dtype=float)

        if (ls.ndim == 2 and rs.ndim == 2
                and ls.shape[1] >= 2 and rs.shape[1] >= 2):
            ls_seg = ls[start:end]
            rs_seg = rs[start:end]
            shoulder_valid = (
                np.isfinite(ls_seg[:, 0]) & np.isfinite(ls_seg[:, 1]) &
                np.isfinite(rs_seg[:, 0]) & np.isfinite(rs_seg[:, 1])
            )
            if np.sum(shoulder_valid) >= 8:
                shoulder_mid_x = (ls_seg[:, 0] + rs_seg[:, 0]) / 2.0
                shoulder_width_signal = np.sqrt(
                    (rs_seg[:, 0] - ls_seg[:, 0]) ** 2 +
                    (rs_seg[:, 1] - ls_seg[:, 1]) ** 2
                )
                shoulder_width_signal[~shoulder_valid] = np.nan
                shoulder_width = shoulder_width_signal

    # fallback：肩部数据不足时退化
    if shoulder_mid_x is None:
        shoulder_mid_x = np.full(n, np.nanmedian(bar_filtered[:, 0]))

    if shoulder_width is None:
        if left is not None and right is not None:
            grip = np.sqrt(
                (left[start:end, 0] - right[start:end, 0]) ** 2 +
                (left[start:end, 1] - right[start:end, 1]) ** 2
            )
            width = float(np.nanmedian(grip))
        else:
            width = x_iqr * 2.0
        width = max(width, 10.0)
        shoulder_width = np.full(n, width)
    else:
        shoulder_width = np.asarray(shoulder_width, dtype=float)
        fallback_w = np.nanmedian(shoulder_width)
        shoulder_width[~np.isfinite(shoulder_width) | (shoulder_width < 5.0)] = fallback_w
        shoulder_width[~np.isfinite(shoulder_width)] = 10.0

    # ═══════════════════════════════════════════════════════════
    # 5. 逐帧 body-relative position + 真实时间轴
    # ═══════════════════════════════════════════════════════════
    x_relative = (bar_filtered[:, 0] - shoulder_mid_x) / np.maximum(shoulder_width, 10.0)

    # 保留 frame index，用于真实时间轴
    frame_idx = np.arange(start, end)
    valid_idx = frame_idx  # bar_filtered 已经插值，全部有效
    x = x_relative  # 已经全部有效（插值后）

    # 真实时间轴
    t_full = (valid_idx - valid_idx[0]) / max(valid_idx[-1] - valid_idx[0], 1)

    print(f"   🛤️   time_axis: {len(x)} points, span={valid_idx[-1] - valid_idx[0]} frames")

    # ═══════════════════════════════════════════════════════════
    # 6. 7点移动平均平滑
    # ═══════════════════════════════════════════════════════════
    if len(x) >= 7:
        kernel = np.ones(7, dtype=float) / 7.0
        padded = np.pad(x, (3, 3), mode="edge")
        x_smooth = np.convolve(padded, kernel, mode="valid")
    else:
        x_smooth = x.copy()

    # ═══════════════════════════════════════════════════════════
    # 7. Path Offset（相对于肩中线的绝对偏移，辅助证据）
    # ═══════════════════════════════════════════════════════════
    abs_x = np.abs(x_smooth)
    p90_offset = float(np.percentile(abs_x, 90))
    max_offset = float(np.max(abs_x))

    OFFSET_MODERATE = 0.30
    OFFSET_SEVERE = 0.40
    offset_moderate = p90_offset >= OFFSET_MODERATE
    offset_severe = p90_offset >= OFFSET_SEVERE

    # ═══════════════════════════════════════════════════════════
    # 8. Wobble（横向反复摆动）
    # ═══════════════════════════════════════════════════════════
    dx = np.diff(x_smooth)
    if len(dx) < 3:
        print("   ❌ [BarPath] EARLY RETURN: 轨迹点不足")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              confidence=0.0, detail="轨迹点不足")

    total_variation = float(np.sum(np.abs(dx)))
    net_shift = float(abs(x_smooth[-1] - x_smooth[0]))
    wobble = max(0.0, total_variation - net_shift)

    WOBBLE_MODERATE = 0.08
    WOBBLE_SEVERE = 0.14
    wobble_moderate = wobble >= WOBBLE_MODERATE
    wobble_severe = wobble >= WOBBLE_SEVERE

    # ═══════════════════════════════════════════════════════════
    # 9. 方向反转计数
    # ═══════════════════════════════════════════════════════════
    DIFF_EPS = 0.008
    signs = np.zeros_like(dx)
    signs[dx > DIFF_EPS] = 1
    signs[dx < -DIFF_EPS] = -1
    nz = signs[signs != 0]
    reversal_count = 0
    if len(nz) >= 2:
        reversal_count = int(np.sum(nz[1:] != nz[:-1]))

    REVERSAL_MODERATE = 4
    REVERSAL_SEVERE = 7
    reversal_moderate = reversal_count >= REVERSAL_MODERATE
    reversal_severe = reversal_count >= REVERSAL_SEVERE

    # ═══════════════════════════════════════════════════════════
    # 10. Reference Path Corridor（二次拟合残差，使用真实时间轴）
    # ═══════════════════════════════════════════════════════════
    try:
        coef = np.polyfit(t_full, x_smooth, deg=2)
        fitted = np.polyval(coef, t_full)
        residual_rms = float(np.sqrt(np.mean((x_smooth - fitted) ** 2)))
        residual_peak = float(np.max(np.abs(x_smooth - fitted)))
    except Exception:
        residual_rms = np.inf
        residual_peak = np.inf

    RESIDUAL_MODERATE = 0.045
    RESIDUAL_SEVERE = 0.075
    residual_moderate = residual_rms >= RESIDUAL_MODERATE
    residual_severe = residual_rms >= RESIDUAL_SEVERE

    # ═══════════════════════════════════════════════════════════
    # 11. 多证据综合
    # ═══════════════════════════════════════════════════════════
    moderate_evidence = sum([offset_moderate, wobble_moderate, reversal_moderate, residual_moderate])
    severe_evidence = sum([offset_severe, wobble_severe, reversal_severe, residual_severe])

    other_dynamic = wobble_moderate or reversal_moderate or residual_moderate
    other_severe = wobble_severe or reversal_severe or residual_severe

    # ═══════════════════════════════════════════════════════════
    # 12. 诊断日志（详细指标）
    # ═══════════════════════════════════════════════════════════
    print(f"   🛤️   proxy={proxy_mode} quality={quality:.2f}")
    print(f"   🛤️   p90_offset={p90_offset:.3f} "
          f"{'[SEVERE]' if offset_severe else '[MODERATE]' if offset_moderate else '[NORMAL]'}")
    print(f"   🛤️   max_offset={max_offset:.3f}")
    print(f"   🛤️   net_shift={net_shift:.3f}")
    print(f"   🛤️   wobble={wobble:.3f} "
          f"{'[SEVERE]' if wobble_severe else '[MODERATE]' if wobble_moderate else '[NORMAL]'}")
    print(f"   🛤️   residual={residual_rms:.3f} (peak={residual_peak:.3f}) "
          f"{'[SEVERE]' if residual_severe else '[MODERATE]' if residual_moderate else '[NORMAL]'}")
    print(f"   🛤️   reversals={reversal_count} "
          f"{'[SEVERE]' if reversal_severe else '[MODERATE]' if reversal_moderate else '[NORMAL]'}")
    print(f"   🛤️   evidence={moderate_evidence}/{severe_evidence}")

    # ═══════════════════════════════════════════════════════════
    # 13. 判定
    # ═══════════════════════════════════════════════════════════
    detail = (
        f"proxy={proxy_mode}, p90_offset={p90_offset:.3f}, "
        f"max_offset={max_offset:.3f}, net_shift={net_shift:.3f}, "
        f"wobble={wobble:.3f}, residual={residual_rms:.3f}, "
        f"reversals={reversal_count}, bar_valid={bar_valid_ratio:.2f}, "
        f"jumps_removed={jump_count}"
    )

    # Severe：至少两类强证据
    if severe_evidence >= 2:
        print("   🛤️   → SEVERE (多类强证据)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.SEVERE,
            value=p90_offset, threshold=OFFSET_SEVERE, confidence=quality,
            detail=f"杠铃轨迹明显偏离/不稳定: {detail}",
        )

    # Severe（备选）：offset_severe + 任何其他严重证据
    if offset_severe and other_severe:
        print("   🛤️   → SEVERE (大偏移+严重动态证据)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.SEVERE,
            value=p90_offset, threshold=OFFSET_SEVERE, confidence=quality,
            detail=f"杠铃轨迹大偏移+不稳定: {detail}",
        )

    # Moderate：offset_severe 单独可触发（避免纯粹平滑偏移漏检）
    if offset_severe:
        print("   🛤️   → MODERATE (大偏移，平滑)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.MODERATE,
            value=p90_offset, threshold=OFFSET_SEVERE, confidence=quality,
            detail=f"杠铃轨迹整体偏移较大: {detail}",
        )

    # Moderate：offset_moderate + 任何其他动态/形状证据
    if offset_moderate and other_dynamic:
        print("   🛤️   → MODERATE (中等偏移+动态证据)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.MODERATE,
            value=p90_offset, threshold=OFFSET_MODERATE, confidence=quality,
            detail=f"杠铃轨迹偏移+不稳定: {detail}",
        )

    # Moderate：至少两类中等证据
    if moderate_evidence >= 2:
        print("   🛤️   → MODERATE (多类中等证据)")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.MODERATE,
            value=p90_offset, threshold=OFFSET_MODERATE, confidence=quality,
            detail=f"杠铃轨迹存在不稳定: {detail}",
        )

    # Normal
    print("   🛤️   → NORMAL")
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=p90_offset, threshold=OFFSET_MODERATE, confidence=quality,
        detail=f"杠铃轨迹未发现明显偏移: {detail}",
    )


def detect_touch_point_drift(rep: RepContext) -> ErrorDetection:
    """
    触胸位置漂移检测（rep-level 只计算指标，set-level 判断 drift）。

    单 rep 计算 bottom_frame 对应的 wrist midpoint x，
    归一化到 shoulder_width（相对于 shoulder 中线）。
    返回 NOT_DETECTED，value 存归一化触胸点。

    真正的 drift 判断在 ErrorDetectionEngineFixed.detect_set 末尾做，
    比较多 rep 的 touch_x_norm 的 std。
    """
    eid = "bench_touch_point_drift"

    if rep.left_wrist is None or rep.right_wrist is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少wrist坐标")

    bf_rel = rep.bottom_frame - rep.start_frame
    left = np.asarray(rep.left_wrist)
    right = np.asarray(rep.right_wrist)

    if bf_rel < 0 or bf_rel >= len(left) or bf_rel >= len(right):
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="bottom frame超出wrist数组范围")

    lx = left[bf_rel, 0]
    rx = right[bf_rel, 0]

    if not np.isfinite(lx) or not np.isfinite(rx):
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="bottom帧wrist x坐标无效")

    bar_x = float((lx + rx) / 2.0)

    # 归一化：相对于 shoulder 中线，尺度为 shoulder_width
    shoulder_mid_x = bar_x
    shoulder_width = 1.0
    if rep.left_shoulder is not None and rep.right_shoulder is not None:
        ls = np.asarray(rep.left_shoulder)
        rs = np.asarray(rep.right_shoulder)
        if bf_rel < len(ls) and bf_rel < len(rs):
            if np.isfinite(ls[bf_rel, 0]) and np.isfinite(rs[bf_rel, 0]):
                shoulder_mid_x = float((ls[bf_rel, 0] + rs[bf_rel, 0]) / 2.0)
                shoulder_width = float(np.abs(ls[bf_rel, 0] - rs[bf_rel, 0]))
        else:
            svalid = (np.isfinite(ls[:, 0]) & np.isfinite(rs[:, 0]))
            if svalid.sum() >= 3:
                shoulder_mid_x = float(np.nanmedian((ls[svalid, 0] + rs[svalid, 0]) / 2.0))
                shoulder_width = float(np.nanmedian(np.abs(ls[svalid, 0] - rs[svalid, 0])))

    if shoulder_width < 1.0:
        shoulder_width = 1.0

    touch_x_norm = float((bar_x - shoulder_mid_x) / shoulder_width)

    # 单 rep 不判断 drift，把值存在 value 里供 set-level 使用
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=touch_x_norm, threshold=None, confidence=0.0,
        detail=f"touch_x_norm={touch_x_norm:.3f}",
    )


def detect_grip_width(rep: RepContext) -> ErrorDetection:
    """
    握距一致性检测（rep-level 只计算指标，set-level 判断 consistency）。

    单 rep 计算归一化握距：median(|left_wrist_x - right_wrist_x|) / shoulder_width。
    返回 NOT_DETECTED，value 存归一化握距。

    真正的 consistency 判断在 ErrorDetectionEngineFixed.detect_set 末尾做，
    比较多 rep 的 grip_width_norm 的变异系数 (CV)。
    """
    eid = "bench_grip_width"

    if rep.left_wrist is None or rep.right_wrist is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少wrist坐标")

    left = np.asarray(rep.left_wrist)
    right = np.asarray(rep.right_wrist)

    if left.ndim != 2 or right.ndim != 2:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="wrist坐标维度错误")

    valid = (np.isfinite(left[:, 0]) & np.isfinite(right[:, 0]))
    if valid.sum() < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="wrist x坐标有效帧不足")

    grip_width = float(np.nanmedian(np.abs(left[valid, 0] - right[valid, 0])))

    # 归一化到 shoulder_width
    shoulder_width = None
    if rep.left_shoulder is not None and rep.right_shoulder is not None:
        ls = np.asarray(rep.left_shoulder)
        rs = np.asarray(rep.right_shoulder)
        svalid = (np.isfinite(ls[:, 0]) & np.isfinite(rs[:, 0]))
        if svalid.sum() >= 3:
            shoulder_width = float(np.nanmedian(np.abs(ls[svalid, 0] - rs[svalid, 0])))

    if shoulder_width is None or shoulder_width < 1.0:
        shoulder_width = max(grip_width, 1.0)

    grip_norm = float(grip_width / shoulder_width)

    # 单 rep 不判断一致性，把值存在 value 里供 set-level 使用
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=grip_norm, threshold=None, confidence=0.0,
        detail=f"grip_width_norm={grip_norm:.3f}",
    )


# ═══════════════════════════════════════════════════════════════
# 规则分组（4层体系：完成度/路径/稳定性/动力学）
# ═══════════════════════════════════════════════════════════════

BENCH_RULE_GROUPS = {
    "completion": [
        detect_incomplete_rom,
        detect_incomplete_lockout,
    ],
    "path": [
        detect_elbow_flare,
        detect_bar_path,
    ],
    "stability": [
        detect_butt_off_bench,
    ],
    "dynamics": [
        detect_bounce,
        detect_eccentric_speed,
        detect_asymmetric_push,
    ],
}

# 扁平化列表（保持向后兼容；touch_point_drift 和 grip_width 是 set-level 检测，不在此列）
BENCH_RULES = [fn for group in BENCH_RULE_GROUPS.values() for fn in group]


class ErrorDetectionEngineFixed:

    def detect_set(self, reps: List[RepContext]) -> List[SetLevelError]:
        all_det: Dict[str, List[ErrorDetection]] = {}

        for rep in reps:
            if rep.validation_status != ValidationStatus.VALID:
                continue
            for fn in BENCH_RULES:
                print(f"   🧪 [RuleExecute] rep={rep.rep_index} rule={fn.__name__}")
                d = fn(rep)
                print(f"   🧪 [RuleResult] {fn.__name__} → status={d.status} severity={d.severity} value={d.value}")
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

        # ═══════════════════════════════════════════════════════
        # Set-level 检测（跨 rep 一致性：触胸漂移 + 握距一致性）
        # ═══════════════════════════════════════════════════════
        valid_reps = [r for r in reps if r.validation_status == ValidationStatus.VALID]

        # --- 触胸位置漂移 ---
        touch_points = []
        for rep in valid_reps:
            d = detect_touch_point_drift(rep)
            if d.status == ErrorStatus.NOT_DETECTED and d.value is not None:
                touch_points.append(d.value)

        if len(touch_points) >= 3:
            touch_std = float(np.std(touch_points))
            if touch_std >= 0.08:  # 归一化触胸点的 std 阈值
                sev = ErrorSeverity.SEVERE if touch_std >= 0.12 else ErrorSeverity.MODERATE
                results.append(SetLevelError(
                    error_id="bench_touch_point_drift",
                    display_name=self._name("bench_touch_point_drift"),
                    occurrences=list(range(1, len(touch_points) + 1)),
                    frequency=1.0,
                    worst_severity=sev,
                    avg_value=touch_std,
                ))

        # --- 握距一致性 ---
        grip_widths = []
        for rep in valid_reps:
            d = detect_grip_width(rep)
            if d.status == ErrorStatus.NOT_DETECTED and d.value is not None:
                grip_widths.append(d.value)

        if len(grip_widths) >= 3:
            grip_mean = float(np.mean(grip_widths))
            grip_std = float(np.std(grip_widths))
            grip_cv = grip_std / max(grip_mean, 1e-6)
            if grip_cv >= 0.08:  # 变异系数阈值
                sev = ErrorSeverity.SEVERE if grip_cv >= 0.12 else ErrorSeverity.MODERATE
                results.append(SetLevelError(
                    error_id="bench_grip_width",
                    display_name=self._name("bench_grip_width"),
                    occurrences=list(range(1, len(grip_widths) + 1)),
                    frequency=1.0,
                    worst_severity=sev,
                    avg_value=grip_cv,
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
            "bench_eccentric_speed": "离心速度失控",
            "bench_bar_path": "杠铃轨迹不稳定",
            "bench_touch_point_drift": "触胸位置漂移",
            "bench_grip_width": "握距不一致",
        }.get(eid, eid)


# 模块加载确认日志（用于排查 Uvicorn reload 是否真正加载了新代码）
print(f"🔥 ErrorDetectionEngine loaded: {__file__}")