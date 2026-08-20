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