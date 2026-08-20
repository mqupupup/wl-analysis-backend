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
    V8: Phase-Aware Bar Path Detector（修复 V7 正常 J 型轨迹被误判）。

    核心修复（解决 V7 正常轨迹被误判 SEVERE）：
    1. 删除 wobble = total_variation - net_shift：正常 J 型轨迹天然有大量"非净位移"，
       这个公式会把正常 J 型当成摆动。
    2. 删除全局 reversal 计数：正常 J 型本身就有一次方向反转，MediaPipe 抖动会被算成 reversal。
    3. 删除 quadratic residual：真实轨迹不像二次函数不代表左右晃。
    4. 分 eccentric / concentric 阶段分析：每阶段横向趋势是否总体一致。
    5. 用 path_amplitude 归一化（而不是 shoulder_width）：侧视视频中 shoulder_width 很小
       （躯干长度仅12px），1px抖动就会被放大成0.08。
    6. dead-zone 过滤低幅度 pose jitter：|dx| < EPS 视为 0。
    7. 只统计持续 ≥3 帧的反向运动：避免 MediaPipe 单帧抖动被算成 reversal。

    V7 的问题：wobble=2.513, residual=0.123, reversals=15 全部 SEVERE，
    但实际是正常 J 型轨迹 + MediaPipe 抖动 + shoulder_width 太小导致的假阳性。
    """
    eid = "bench_bar_path"

    # ═══════════════════════════════════════════════════════════
    # 0. 诊断日志
    # ═══════════════════════════════════════════════════════════
    print(f"\n   🛤️ [BarPath V8] Rep {rep.rep_index}")
    left_shape = None if rep.left_wrist is None else np.asarray(rep.left_wrist).shape
    right_shape = None if rep.right_wrist is None else np.asarray(rep.right_wrist).shape
    print(f"   🛤️   left_wrist={left_shape}")
    print(f"   🛤️   right_wrist={right_shape}")
    print(f"   🛤️   rep_abs=[{rep.start_frame}, {rep.end_frame}]")

    # ═══════════════════════════════════════════════════════════
    # 1. 基础数据
    # ═══════════════════════════════════════════════════════════
    if rep.left_wrist is None and rep.right_wrist is None:
        print("   ❌ [BarPath] EARLY RETURN: 两侧 wrist 都是 None")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              confidence=0.0, detail="左右 wrist 均不可用")

    left = np.asarray(rep.left_wrist, dtype=float) if rep.left_wrist is not None else None
    right = np.asarray(rep.right_wrist, dtype=float) if rep.right_wrist is not None else None

    # ═══════════════════════════════════════════════════════════
    # 2. 完整 rep 范围
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

    n = end - start
    print(f"   🛤️   full_rep=[{start}:{end}] len={n}")

    if n < 8:
        print(f"   ❌ [BarPath] EARLY RETURN: full rep len={n} < 8")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              confidence=0.0, detail=f"完整 rep 轨迹长度不足: {n}")

    # ═══════════════════════════════════════════════════════════
    # 3. 计算各侧有效率，决定 proxy 模式（和 V7 一样）
    # ═══════════════════════════════════════════════════════════
    l_seg = left[start:end] if left is not None else None
    r_seg = right[start:end] if right is not None else None

    left_valid = np.zeros(n, dtype=bool)
    right_valid = np.zeros(n, dtype=bool)

    if l_seg is not None:
        left_valid = np.isfinite(l_seg[:, 0]) & np.isfinite(l_seg[:, 1])
    if r_seg is not None:
        right_valid = np.isfinite(r_seg[:, 0]) & np.isfinite(r_seg[:, 1])

    both = left_valid & right_valid
    left_ratio = float(np.mean(left_valid))
    right_ratio = float(np.mean(right_valid))
    both_ratio = float(np.mean(both))

    print(f"   🛤️   left={left_ratio:.3f} right={right_ratio:.3f} both={both_ratio:.3f}")

    # ═══════════════════════════════════════════════════════════
    # 4. 根据 proxy 模式构建 bar 轨迹（和 V7 一样）
    # ═══════════════════════════════════════════════════════════
    bar = np.full((n, 2), np.nan, dtype=float)

    if left_ratio >= 0.80 and l_seg is not None:
        bar = l_seg.copy()
        bar[~left_valid] = np.nan
        proxy_mode = "left_wrist_only"
        quality = 0.78
        print(f"   🛤️   proxy=left_wrist_only (left={left_ratio:.3f}>=0.80)")
    elif right_ratio >= 0.80 and r_seg is not None:
        bar = r_seg.copy()
        bar[~right_valid] = np.nan
        proxy_mode = "right_wrist_only"
        quality = 0.78
        print(f"   🛤️   proxy=right_wrist_only (right={right_ratio:.3f}>=0.80)")
    elif both_ratio >= 0.50:
        bar[both] = (l_seg[both] + r_seg[both]) / 2.0
        proxy_mode = "bilateral_midpoint"
        quality = 0.90
        print(f"   🛤️   proxy=bilateral_midpoint (both={both_ratio:.3f}>=0.50)")
    else:
        if np.any(both):
            bar[both] = (l_seg[both] + r_seg[both]) / 2.0
        left_only = left_valid & ~right_valid
        if np.any(left_only) and l_seg is not None:
            bar[left_only] = l_seg[left_only]
        right_only = right_valid & ~left_valid
        if np.any(right_only) and r_seg is not None:
            bar[right_only] = r_seg[right_only]
        proxy_mode = "mixed_wrist_proxy"
        quality = 0.60
        print(f"   🛤️   proxy=mixed_wrist_proxy")

    bar_valid = np.isfinite(bar[:, 0]) & np.isfinite(bar[:, 1])
    bar_valid_ratio = float(np.mean(bar_valid))
    print(f"   🛤️   bar_valid={int(bar_valid.sum())}/{n} ({bar_valid_ratio:.3f})")

    if bar_valid.sum() < 8:
        print(f"   ❌ [BarPath] EARLY RETURN: 有效 bar proxy={int(bar_valid.sum())} < 8")
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA, confidence=0.0,
                              detail=f"有效 bar proxy 帧不足: {int(bar_valid.sum())}/{n}")

    # ═══════════════════════════════════════════════════════════
    # 5. 线性插值填充 NaN
    # ═══════════════════════════════════════════════════════════
    bar_interp = bar.copy()
    for col in range(2):
        valid = np.isfinite(bar_interp[:, col])
        if np.sum(valid) >= 2:
            bar_interp[:, col] = np.interp(
                np.arange(n), np.where(valid)[0], bar_interp[valid, col]
            )

    # ═══════════════════════════════════════════════════════════
    # 6. 用 path_amplitude 归一化（而不是 shoulder_width）
    #
    # 侧视视频中 shoulder_width 很小（躯干长度仅12px），
    # 1px MediaPipe 抖动就会被放大成 0.08。
    # 用卧推本身的水平运动幅度作为尺度更稳定。
    # ═══════════════════════════════════════════════════════════
    x_raw = bar_interp[:, 0]
    path_amplitude = float(
        np.percentile(x_raw, 95) - np.percentile(x_raw, 5)
    )

    # 如果水平运动幅度太小（<10px），可能是正面视频，用 shoulder_width fallback
    if path_amplitude < 10.0:
        if rep.left_shoulder is not None and rep.right_shoulder is not None:
            ls = np.asarray(rep.left_shoulder, dtype=float)
            rs = np.asarray(rep.right_shoulder, dtype=float)
            if ls.ndim == 2 and rs.ndim == 2 and ls.shape[1] >= 2:
                sw = np.median(np.sqrt(
                    (rs[start:end, 0] - ls[start:end, 0]) ** 2 +
                    (rs[start:end, 1] - ls[start:end, 1]) ** 2
                ))
                if np.isfinite(sw) and sw > 5.0:
                    path_amplitude = sw
                    print(f"   🛤️   path_amplitude太小，用shoulder_width={sw:.1f}px替代")

    path_amplitude = max(path_amplitude, 10.0)

    # 归一化到 [-0.5, 0.5] 范围（median 为 0）
    x_median = float(np.median(x_raw))
    x_norm = (x_raw - x_median) / path_amplitude

    print(f"   🛤️   path_amplitude={path_amplitude:.1f}px")
    print(f"   🛤️   x_norm range=[{x_norm.min():.3f}, {x_norm.max():.3f}]")

    # ═══════════════════════════════════════════════════════════
    # 7. 7点移动平均平滑
    # ═══════════════════════════════════════════════════════════
    if len(x_norm) >= 7:
        kernel = np.ones(7, dtype=float) / 7.0
        padded = np.pad(x_norm, (3, 3), mode="edge")
        x_smooth = np.convolve(padded, kernel, mode="valid")
    else:
        x_smooth = x_norm.copy()

    # ═══════════════════════════════════════════════════════════
    # 8. 分阶段：eccentric / concentric
    # ═══════════════════════════════════════════════════════════
    # 相对索引（相对于 rep 起始，已经去掉了前后各1帧，所以需要调整）
    # start 已经是去掉前1帧后的索引，所以相对索引 = 原始索引 - start
    # 但 rep.start_frame 是原始帧号，bar 数组是从 rep.start_frame 开始的
    # 所以 bar[i] 对应帧号 = rep.start_frame + i + (1 if 去掉了前1帧 else 0)

    # 简化：用 bottom_frame 和 concentric_start 作为分界
    bottom_rel = None
    if hasattr(rep, 'bottom_frame') and rep.bottom_frame is not None:
        bottom_rel = rep.bottom_frame - rep.start_frame - start
        if bottom_rel < 1 or bottom_rel >= n - 1:
            bottom_rel = None

    concentric_start_rel = None
    if hasattr(rep, 'concentric_start') and rep.concentric_start is not None:
        concentric_start_rel = rep.concentric_start - rep.start_frame - start
        if concentric_start_rel < 1 or concentric_start_rel >= n - 1:
            concentric_start_rel = None

    # 用 bottom_frame 作为分界点（更准确）
    if bottom_rel is not None:
        ecc_end = bottom_rel
        con_start = bottom_rel
    elif concentric_start_rel is not None:
        ecc_end = concentric_start_rel
        con_start = concentric_start_rel
    else:
        # fallback：用中间点
        ecc_end = n // 2
        con_start = n // 2

    ecc_end = max(2, min(ecc_end, n - 2))
    con_start = max(2, min(con_start, n - 2))

    print(f"   🛤️   eccentric=[0:{ecc_end}] len={ecc_end}")
    print(f"   🛤️   concentric=[{con_start}:{n}] len={n - con_start}")

    # ═══════════════════════════════════════════════════════════
    # 9. dead-zone 过滤 + 阶段分析
    #
    # EPS = max(0.015, 0.03 * path_amplitude_norm)
    # 但 x_norm 已经归一化到 path_amplitude，所以 EPS 用相对值
    # ═══════════════════════════════════════════════════════════
    EPS = 0.015  # 1.5% of path_amplitude，小于这个幅度的变化视为 pose noise

    def analyze_phase(x_phase, phase_name):
        """分析一个阶段的横向运动稳定性。

        Returns:
            dict: {
                'dominant_dir': 1 or -1,
                'reverse_ratio': 反向帧占比,
                'sustained_reversals': 持续≥3帧的反向运动段数,
                'max_sustained_reverse': 最大持续反向段的累计位移,
                'high_freq_wobble': 高频摆动次数（方向持续≥3帧的反转）,
            }
        """
        if len(x_phase) < 4:
            return {
                'dominant_dir': 0, 'reverse_ratio': 0.0,
                'sustained_reversals': 0, 'max_sustained_reverse': 0.0,
                'high_freq_wobble': 0,
            }

        dx = np.diff(x_phase)

        # dead-zone 过滤
        dx_filtered = dx.copy()
        dx_filtered[np.abs(dx_filtered) < EPS] = 0.0

        # 主导方向（median sign）
        nonzero_dx = dx_filtered[dx_filtered != 0]
        if len(nonzero_dx) == 0:
            dominant_dir = 0
        else:
            dominant_dir = 1 if np.median(nonzero_dx) > 0 else -1

        # 反向帧占比
        if dominant_dir == 0:
            reverse_ratio = 0.0
        else:
            reverse_frames = np.sum(dx_filtered * dominant_dir < 0)
            reverse_ratio = float(reverse_frames / max(len(dx_filtered), 1))

        # 持续反向运动段（连续 ≥3 帧反向）
        sustained_reversals = 0
        max_sustained_reverse = 0.0
        if dominant_dir != 0:
            is_reverse = dx_filtered * dominant_dir < 0
            # 找连续段
            current_len = 0
            current_sum = 0.0
            for i in range(len(is_reverse)):
                if is_reverse[i]:
                    current_len += 1
                    current_sum += abs(dx_filtered[i])
                else:
                    if current_len >= 3:
                        sustained_reversals += 1
                        max_sustained_reverse = max(max_sustained_reverse, current_sum)
                    current_len = 0
                    current_sum = 0.0
            if current_len >= 3:
                sustained_reversals += 1
                max_sustained_reverse = max(max_sustained_reverse, current_sum)

        # 高频摆动：方向持续 ≥3 帧的反转次数
        high_freq_wobble = 0
        signs = np.sign(dx_filtered)
        signs = signs[signs != 0]
        if len(signs) >= 4:
            # 压缩连续相同符号
            compressed = []
            current_sign = signs[0]
            current_len = 1
            for i in range(1, len(signs)):
                if signs[i] == current_sign:
                    current_len += 1
                else:
                    if current_len >= 3:
                        compressed.append(current_sign)
                    current_sign = signs[i]
                    current_len = 1
            if current_len >= 3:
                compressed.append(current_sign)

            # 统计反转次数
            for i in range(1, len(compressed)):
                if compressed[i] != compressed[i-1]:
                    high_freq_wobble += 1

        return {
            'dominant_dir': dominant_dir,
            'reverse_ratio': reverse_ratio,
            'sustained_reversals': sustained_reversals,
            'max_sustained_reverse': max_sustained_reverse,
            'high_freq_wobble': high_freq_wobble,
        }

    ecc_result = analyze_phase(x_smooth[:ecc_end], "eccentric")
    con_result = analyze_phase(x_smooth[con_start:], "concentric")

    print(f"   🛤️   [eccentric] dominant_dir={ecc_result['dominant_dir']} "
          f"reverse_ratio={ecc_result['reverse_ratio']:.3f} "
          f"sustained_rev={ecc_result['sustained_reversals']} "
          f"max_reverse={ecc_result['max_sustained_reverse']:.3f} "
          f"hf_wobble={ecc_result['high_freq_wobble']}")
    print(f"   🛤️   [concentric] dominant_dir={con_result['dominant_dir']} "
          f"reverse_ratio={con_result['reverse_ratio']:.3f} "
          f"sustained_rev={con_result['sustained_reversals']} "
          f"max_reverse={con_result['max_sustained_reverse']:.3f} "
          f"hf_wobble={con_result['high_freq_wobble']}")

    # ═══════════════════════════════════════════════════════════
    # 10. 三个综合指标
    # ═══════════════════════════════════════════════════════════

    # 指标1：phase_monotonicity（阶段单调性）
    # 两个阶段的反向帧占比都 < 30% 为正常
    ecc_monotonic = ecc_result['reverse_ratio'] < 0.30
    con_monotonic = con_result['reverse_ratio'] < 0.30
    phase_monotonicity_ok = ecc_monotonic and con_monotonic
    phase_monotonicity_severe = (
        ecc_result['reverse_ratio'] >= 0.50 or
        con_result['reverse_ratio'] >= 0.50
    )

    # 指标2：sustained_lateral_excursion（持续横向偏移）
    # 两个阶段都没有持续 ≥3帧的反向运动为正常
    ecc_sustained_ok = ecc_result['sustained_reversals'] == 0
    con_sustained_ok = con_result['sustained_reversals'] == 0
    sustained_ok = ecc_sustained_ok and con_sustained_ok
    sustained_severe = (
        ecc_result['max_sustained_reverse'] >= 0.15 or
        con_result['max_sustained_reverse'] >= 0.15
    )

    # 指标3：high_frequency_wobble（高频摆动）
    # 两个阶段的高频摆动都 < 2 为正常
    ecc_hf_ok = ecc_result['high_freq_wobble'] < 2
    con_hf_ok = con_result['high_freq_wobble'] < 2
    hf_wobble_ok = ecc_hf_ok and con_hf_ok
    hf_wobble_severe = (
        ecc_result['high_freq_wobble'] >= 3 or
        con_result['high_freq_wobble'] >= 3
    )

    # ═══════════════════════════════════════════════════════════
    # 11. 诊断日志
    # ═══════════════════════════════════════════════════════════
    print(f"   🛤️   phase_monotonicity={'OK' if phase_monotonicity_ok else 'BAD'} "
          f"(ecc_rev={ecc_result['reverse_ratio']:.2f}, con_rev={con_result['reverse_ratio']:.2f})")
    print(f"   🛤️   sustained_excursion={'OK' if sustained_ok else 'BAD'} "
          f"(ecc_sus={ecc_result['sustained_reversals']}, con_sus={con_result['sustained_reversals']})")
    print(f"   🛤️   high_freq_wobble={'OK' if hf_wobble_ok else 'BAD'} "
          f"(ecc_hf={ecc_result['high_freq_wobble']}, con_hf={con_result['high_freq_wobble']})")

    # ═══════════════════════════════════════════════════════════
    # 12. 判定
    # ═══════════════════════════════════════════════════════════
    abnormal_count = sum([
        not phase_monotonicity_ok,
        not sustained_ok,
        not hf_wobble_ok,
    ])
    severe_count = sum([
        phase_monotonicity_severe,
        sustained_severe,
        hf_wobble_severe,
    ])

    detail = (
        f"proxy={proxy_mode}, path_amp={path_amplitude:.0f}px, "
        f"ecc_rev_ratio={ecc_result['reverse_ratio']:.2f}, "
        f"con_rev_ratio={con_result['reverse_ratio']:.2f}, "
        f"ecc_sustained={ecc_result['sustained_reversals']}, "
        f"con_sustained={con_result['sustained_reversals']}, "
        f"ecc_hf={ecc_result['high_freq_wobble']}, "
        f"con_hf={con_result['high_freq_wobble']}"
    )

    # Severe：至少 2 个强异常
    if severe_count >= 2:
        print(f"   🛤️   → SEVERE (severe_count={severe_count})")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.SEVERE,
            value=float(abnormal_count), threshold=2.0, confidence=quality,
            detail=f"杠铃轨迹明显不稳定: {detail}",
        )

    # Moderate：至少 2 个异常
    if abnormal_count >= 2:
        print(f"   🛤️   → MODERATE (abnormal_count={abnormal_count})")
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.DETECTED, ErrorSeverity.MODERATE,
            value=float(abnormal_count), threshold=2.0, confidence=quality,
            detail=f"杠铃轨迹存在不稳定: {detail}",
        )

    # Normal
    print(f"   🛤️   → NORMAL (abnormal_count={abnormal_count})")
    return ErrorDetection(
        eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
        value=float(abnormal_count), threshold=2.0, confidence=quality,
        detail=f"杠铃轨迹稳定（正常J型）: {detail}",
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